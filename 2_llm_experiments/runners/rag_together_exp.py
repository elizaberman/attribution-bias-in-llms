#!/usr/bin/env python3
"""Evidence-conditioned (RAG) attribution runner for open-weight models on Together AI.

The evidence-conditioned arm of the paper's evaluation: each quote is presented
alongside retrieved neighbor quotes that include their authors, so the correct
attribution is available in the prompt. Suppression is the failure to use it.
The no-evidence counterpart is `parallel_together_exp.py`.

Pipeline
--------
1. Embed every quote with an OpenAI embedding model (default text-embedding-3-small),
   caching vectors to disk (--embedding-cache-path) so reruns skip re-embedding.
2. Retrieve the --top-k (default 5) nearest neighbors by cosine similarity
   (`build_topk_neighbors`); --exclude-self-match drops the quote itself.
3. Render them into the {context} slot of RAG_PROMPTS (`format_context`) and submit
   all requests through the Together **Batch API** (JSONL upload, then poll).

Because it uses the batch API rather than live calls, this runs at lower cost but
returns only once the whole batch completes.

Usage
-----
  python rag_together_exp.py --help

Requires OPENAI_API_KEY (embeddings only) and TOGETHER_API_KEY (generation).
Output: one CSV row per quote x prompt type x run.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from together import Together

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

_ATTRIBENCH_ROOT = os.environ.get(
    "ATTRIBENCH_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_PATH = os.path.join(
    _ATTRIBENCH_ROOT, "1_dataset_construction", "datasets", "multirace_with_quotes.csv"
)

RAG_PROMPTS = {
    "direct": (
        "Please identify the author of the following quote.\n"
        "Quote:\n\"{passage}\"\n\n"
        "Retrieved examples:\n{context}"
    ),
    "indirect": (
        "Briefly summarize the context in which the following quote was written.\n"
        "Quote:\n\"{passage}\"\n\n"
        "Retrieved examples:\n{context}"
    ),
    "indirect_overt": (
        "Briefly summarize the context in which the following quote was written and mention the author's name if relevant.\n"
        "Quote:\n\"{passage}\"\n\n"
        "Retrieved examples:\n{context}"
    ),
}

TOGETHER_MODELS = [
    "moonshotai/Kimi-K2.5",
    "deepseek-ai/DeepSeek-V3.1",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3.5-397B-A17B",
    "zai-org/GLM-5",
    "openai/gpt-oss-120b",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

DEFAULT_BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_POLL_EVERY_S = 60
DEFAULT_MAX_REQUESTS_PER_BATCH = 48000
DEFAULT_GPT_OSS_REASONING_EFFORT = "low"
SYSTEM_PROMPT = (
    "You are a helpful assistant that provides concise, factual responses. "
    "Respond with at most one sentence."
)


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def sanitize_model_tag(model_name: str) -> str:
    return model_name.replace("/", "_").replace(" ", "_").replace(".", "_")


def quote_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_checkpoint(results: List[str], checkpoint_path: str):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "completed": len(results),
            },
            f,
        )
    flush_print(f"\nCheckpoint saved: {len(results)} completed")


def load_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        completed = checkpoint.get("completed", len(checkpoint.get("results", [])))
        flush_print(f"Resuming from checkpoint: {completed} already completed")
        return checkpoint.get("results", []), completed
    return [], 0


def pick_first_populated_column(df, candidates):
    for col in candidates:
        if col in df.columns and df[col].fillna("").astype(str).str.strip().ne("").any():
            return col
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_columns(df):
    author_col = pick_first_populated_column(df, ["author", "author_clean"])
    race_col = pick_first_populated_column(df, ["race_ethnicity", "race", "chatgpt_race", "perplexity_race"])
    gender_col = pick_first_populated_column(df, ["gender", "chatgpt_gender", "perplexity_gender"])
    if author_col is None or race_col is None:
        raise ValueError(
            "Missing required columns. Need author/author_clean and race_ethnicity/race/chatgpt_race/perplexity_race."
        )
    if "quote" not in df.columns:
        raise ValueError("Missing required column: quote")
    return author_col, race_col, gender_col


def load_embedding_cache(cache_path):
    if cache_path.endswith(".json"):
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cache = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
            flush_print(f" Loaded legacy embedding cache: {len(cache)} vectors")
            return cache
        return {}

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        hashes = data["hashes"]
        embeddings = data["embeddings"]
        cache = {str(h): embeddings[i] for i, h in enumerate(hashes.tolist())}
        flush_print(f" Loaded embedding cache: {len(cache)} vectors")
        return cache
    return {}


def save_embedding_cache(cache, cache_path):
    directory = os.path.dirname(cache_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    hashes = np.array(list(cache.keys()), dtype="<U64")
    embeddings = np.vstack([np.asarray(cache[h], dtype=np.float32) for h in hashes.tolist()])
    np.savez(cache_path, hashes=hashes, embeddings=embeddings)
    flush_print(f" Saved embedding cache: {len(cache)} vectors")


async def embed_texts_batch(client: AsyncOpenAI, texts, embedding_model, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = await client.embeddings.create(model=embedding_model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
            else:
                raise


async def build_embedding_matrix(df, quote_col, embedding_model, cache_path, client: AsyncOpenAI, batch_size=128):
    quotes = df[quote_col].fillna("").astype(str).tolist()
    cache = load_embedding_cache(cache_path)

    unique_quotes = list(dict.fromkeys(quotes))
    missing = []
    for q in unique_quotes:
        h = quote_hash(q)
        if h not in cache:
            missing.append((h, q))

    if missing:
        flush_print(f" Embedding {len(missing)} new quotes with {embedding_model}...")
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            batch_texts = [text for _, text in batch]
            vectors = await embed_texts_batch(client, batch_texts, embedding_model=embedding_model)
            for (h, _), vec in zip(batch, vectors):
                cache[h] = np.asarray(vec, dtype=np.float32)
            flush_print(f"  Embedded {min(start + batch_size, len(missing))}/{len(missing)}")
        save_embedding_cache(cache, cache_path)
    else:
        flush_print(" All quote embeddings found in cache.")

    matrix = np.array([cache[quote_hash(q)] for q in quotes], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-12, None)
    return matrix


def build_topk_neighbors(matrix, top_k=5, max_full_sims_n=15000, exclude_self=False):
    n = matrix.shape[0]
    if n > max_full_sims_n:
        raise RuntimeError(
            f"N={n} is too large for full NxN similarity matrix in-memory; "
            f"use FAISS/NearestNeighbors or increase --max-full-sims-n."
        )
    sims = matrix @ matrix.T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)

    k = min(top_k, sims.shape[1] - (1 if exclude_self else 0))
    if k <= 0:
        return np.empty((sims.shape[0], 0), dtype=np.int32), np.empty((sims.shape[0], 0), dtype=np.float32)

    top_idx_unsorted = np.argpartition(sims, -k, axis=1)[:, -k:]
    top_scores_unsorted = np.take_along_axis(sims, top_idx_unsorted, axis=1)
    order = np.argsort(top_scores_unsorted, axis=1)[:, ::-1]
    top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1).astype(np.int32)
    top_scores = np.take_along_axis(top_scores_unsorted, order, axis=1).astype(np.float32)
    return top_idx, top_scores


def retrieve_neighbors(query_idx, top_idx, top_scores, df, author_col):
    items = []
    for idx, score in zip(top_idx[query_idx], top_scores[query_idx]):
        score = float(score)
        if score <= 0:
            continue
        row = df.iloc[int(idx)]
        items.append(
            {
                "index": int(idx),
                "author": str(row[author_col]),
                "quote": str(row["quote"]),
                "score": score,
            }
        )
    return items


def format_context(items, include_author=True):
    if not items:
        return "No strong embedding-similarity neighbors found in retrieval set."
    lines = []
    for i, item in enumerate(items, start=1):
        if include_author:
            lines.append(
                f"{i}. author: {item['author']} | similarity: {item['score']:.4f} | quote: \"{item['quote']}\""
            )
        else:
            lines.append(
                f"{i}. similarity: {item['score']:.4f} | quote: \"{item['quote']}\""
            )
    return "\n".join(lines)


def ensure_together_client() -> Together:
    api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY")
    if not api_key:
        raise SystemExit("TOGETHER_API_KEY (or TOGETHERAI_API_KEY) not set")
    return Together(api_key=api_key)


def unwrap_batch(resp):
    return getattr(resp, "job", resp)


def _as_dict(obj) -> Dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            return {}
    return {}


def extract_batch_failure_details(batch) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for attr in [
        "status",
        "errors",
        "error",
        "error_message",
        "failed_at",
        "cancelled_at",
        "completed_at",
        "request_counts",
        "error_file_id",
        "output_file_id",
    ]:
        value = getattr(batch, attr, None)
        if value is not None:
            out[attr] = _as_dict(value) if not isinstance(value, (str, int, float, bool, list, dict)) else value
    return out


def submit_batch(client: Together, batch_input_jsonl: Path) -> Tuple[str, str]:
    file_resp = client.files.upload(
        file=str(batch_input_jsonl),
        purpose="batch-api",
        check=False,
    )
    batch_resp = client.batches.create(
        input_file_id=file_resp.id,
        endpoint=DEFAULT_BATCH_ENDPOINT,
    )
    batch = unwrap_batch(batch_resp)
    return file_resp.id, batch.id


def get_batch(client: Together, batch_id: str):
    resp = client.batches.retrieve(batch_id)
    return unwrap_batch(resp)


def wait_for_batch(client: Together, batch_id: str, poll_every_s: int = 60):
    while True:
        batch = get_batch(client, batch_id)
        status = getattr(batch, "status", None)
        flush_print(f"Batch {batch_id}: {status}")
        if status in {"COMPLETED", "FAILED", "CANCELLED"}:
            return batch
        time.sleep(poll_every_s)


def download_batch_outputs(
    client: Together,
    batch_id: str,
    results_jsonl: Path,
    errors_jsonl: Optional[Path] = None,
) -> Dict:
    batch = get_batch(client, batch_id)
    status = getattr(batch, "status", None)

    if status != "COMPLETED":
        raise RuntimeError(f"Batch {batch_id} is not COMPLETED (status={status})")

    output_file_id = getattr(batch, "output_file_id", None)
    error_file_id = getattr(batch, "error_file_id", None)

    if not output_file_id:
        raise RuntimeError(f"Batch {batch_id} completed but has no output_file_id")

    results_jsonl.parent.mkdir(parents=True, exist_ok=True)

    result_content = client.files.content(output_file_id)
    results_jsonl.write_bytes(result_content.read())

    if errors_jsonl and error_file_id:
        error_content = client.files.content(error_file_id)
        errors_jsonl.write_bytes(error_content.read())

    return {
        "status": status,
        "output_file_id": output_file_id,
        "error_file_id": error_file_id,
    }


def build_batch_input_from_prompts(
    prompts: List[str],
    jsonl_path: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    global_start_idx: int,
) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0

    with jsonl_path.open("w", encoding="utf-8") as f:
        for local_idx, prompt in enumerate(prompts):
            global_idx = global_start_idx + local_idx
            record = {
                "custom_id": f"req_{global_idx}",
                "method": "POST",
                "url": DEFAULT_BATCH_ENDPOINT,
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
            if model.startswith("openai/gpt-oss"):
                record["body"]["reasoning_effort"] = DEFAULT_GPT_OSS_REASONING_EFFORT
            else:
                record["body"]["reasoning"] = {"enabled": False}

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    return n


def extract_text_from_batch_response(obj: Dict) -> str:
    try:
        return str(obj["response"]["body"]["choices"][0]["message"]["content"]).strip()
    except Exception:
        return ""


def load_batch_results(results_jsonl: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            if not custom_id.startswith("req_"):
                continue
            idx = int(custom_id.replace("req_", ""))
            out[idx] = extract_text_from_batch_response(obj)
    return out


def generate_with_together_batch(
    prompts: List[str],
    checkpoint_path: str,
    artifacts_dir: str,
    model: str,
    save_every: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    poll_every_s: int,
    max_requests_per_batch: int,
):
    completed_results, completed_count = load_checkpoint(checkpoint_path)
    outputs = completed_results.copy()

    remaining_prompts = prompts[completed_count:]
    if completed_count > 0:
        flush_print(f"Resuming from index {completed_count}/{len(prompts)}")
    flush_print(f" Processing {len(remaining_prompts)} requests with Together Batch API...")

    if save_every <= 0:
        raise ValueError("save_every must be > 0")

    block_size = min(save_every, max_requests_per_batch)
    if block_size <= 0:
        raise ValueError("Invalid block size")

    client = ensure_together_client()
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    total_remaining = len(remaining_prompts)
    n_blocks = (total_remaining + block_size - 1) // block_size if total_remaining else 0

    for block_idx, local_start in enumerate(range(0, total_remaining, block_size), start=1):
        local_end = min(local_start + block_size, total_remaining)
        global_start = completed_count + local_start
        global_end = completed_count + local_end
        block_prompts = remaining_prompts[local_start:local_end]

        shard_prefix = f"batch_{global_start:09d}_{global_end - 1:09d}"
        input_jsonl = artifacts / f"{shard_prefix}_input.jsonl"
        output_jsonl = artifacts / f"{shard_prefix}_output.jsonl"
        error_jsonl = artifacts / f"{shard_prefix}_errors.jsonl"

        req_count = build_batch_input_from_prompts(
            prompts=block_prompts,
            jsonl_path=input_jsonl,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            global_start_idx=global_start,
        )

        flush_print(f"\n Processing block {block_idx}/{n_blocks}")
        flush_print(f"    Items {global_start} to {global_end - 1}")
        flush_print(f"    Wrote {req_count} requests -> {input_jsonl}")

        start_time = time.time()
        file_id, batch_id = submit_batch(client, input_jsonl)
        flush_print(f"    Uploaded file_id={file_id}")
        flush_print(f"    Created batch_id={batch_id}")

        batch = wait_for_batch(client, batch_id, poll_every_s=poll_every_s)
        final_status = getattr(batch, "status", None)
        if final_status != "COMPLETED":
            failure_details = extract_batch_failure_details(batch)
            if failure_details:
                flush_print(f"    Failure details: {json.dumps(failure_details, ensure_ascii=False)}")
            raise RuntimeError(f"Together batch failed with status={final_status} (batch_id={batch_id})")

        meta = download_batch_outputs(
            client=client,
            batch_id=batch_id,
            results_jsonl=output_jsonl,
            errors_jsonl=error_jsonl,
        )
        flush_print(f"    Downloaded outputs: {meta}")

        indexed_results = load_batch_results(output_jsonl)
        block_outputs = []
        missing = 0
        for idx in range(global_start, global_end):
            text = indexed_results.get(idx, "")
            if idx not in indexed_results:
                missing += 1
            block_outputs.append(text)

        outputs.extend(block_outputs)
        elapsed = time.time() - start_time
        rate = (len(block_prompts) / elapsed) if elapsed > 0 else 0
        flush_print(f"     Completed in {elapsed:.1f}s ({rate:.2f} req/s)")
        if missing > 0:
            flush_print(f" {missing} missing results in block (filled with empty strings)")

        save_checkpoint(outputs, checkpoint_path)

    return outputs


async def generate_outputs_async(
    csv_path,
    subset_size=None,
    random_subset_size=None,
    random_subset_seed=None,
    quote_range=None,
    runs=1,
    model="deepseek-ai/DeepSeek-V3.1",
    out_dir="results/prompt_attribution_rag_together",
    save_every=50,
    resume_from=None,
    prompts=None,
    top_k=5,
    embedding_model="text-embedding-3-small",
    embedding_cache_path=None,
    embedding_batch_size=128,
    max_full_sims_n=15000,
    context_label_mode="both",
    exclude_self_match=False,
    max_tokens=128,
    temperature=0.7,
    top_p=0.95,
    poll_every_s=DEFAULT_POLL_EVERY_S,
    max_requests_per_batch=DEFAULT_MAX_REQUESTS_PER_BATCH,
):
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    flush_print(
        f"Using OPENAI_API_KEY: {openai_api_key[:4]}...{openai_api_key[-4:]}"
        if openai_api_key
        else "No OPENAI_API_KEY found"
    )
    if not openai_api_key:
        raise SystemExit("OPENAI_API_KEY not set (required for embeddings)")

    df = pd.read_csv(csv_path)
    flush_print(f" Loaded {len(df)} quotes")

    author_col, race_col, gender_col = infer_columns(df)

    if "quote_id" not in df.columns:
        df["quote_id"] = range(len(df))

    if quote_range:
        start_idx, end_idx = quote_range
        df = df.iloc[start_idx - 1 : end_idx].reset_index(drop=True)
        flush_print(f" Using quotes {start_idx} to {end_idx} ({len(df)} quotes)")
    elif random_subset_size:
        n = min(random_subset_size, len(df))
        df = df.sample(n=n, random_state=random_subset_seed).reset_index(drop=True)
        flush_print(
            f" Using random subset of {n} quotes"
            + (f" (seed={random_subset_seed})" if random_subset_seed is not None else "")
        )
    elif subset_size:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.head(subset_size)

    if resume_from:
        out_dir = resume_from
        flush_print(f"Resuming from: {out_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"{out_dir}_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = f"{out_dir}/checkpoint.json"
    artifacts_dir = f"{out_dir}/batch_artifacts"
    prompts_to_use = prompts if prompts is not None else RAG_PROMPTS

    if embedding_cache_path is None:
        safe_model = embedding_model.replace("/", "_")
        embedding_cache_path = f"results/rag_embedding_cache_{safe_model}.npz"

    embed_client = AsyncOpenAI(api_key=openai_api_key)
    flush_print(f" Building embedding retriever over dataset ({embedding_model})...")
    matrix = await build_embedding_matrix(
        df=df,
        quote_col="quote",
        embedding_model=embedding_model,
        cache_path=embedding_cache_path,
        client=embed_client,
        batch_size=embedding_batch_size,
    )
    top_idx, top_scores = build_topk_neighbors(
        matrix,
        top_k=top_k,
        max_full_sims_n=max_full_sims_n,
        exclude_self=exclude_self_match,
    )

    all_prompts, metadata = [], []
    if context_label_mode not in {"labeled", "unlabeled", "both"}:
        raise ValueError("context_label_mode must be one of: labeled, unlabeled, both")

    if context_label_mode == "both":
        context_variants = [("labeled", True), ("unlabeled", False)]
    elif context_label_mode == "labeled":
        context_variants = [("labeled", True)]
    else:
        context_variants = [("unlabeled", False)]

    for idx, row in df.iterrows():
        retrieved = retrieve_neighbors(
            query_idx=idx,
            top_idx=top_idx,
            top_scores=top_scores,
            df=df,
            author_col=author_col,
        )
        for context_variant, include_author in context_variants:
            context = format_context(retrieved, include_author=include_author)
            for prompt_name, template in prompts_to_use.items():
                for run_id in range(runs):
                    all_prompts.append(template.format(passage=row["quote"], context=context))
                    metadata.append(
                        {
                            "quote_id": row["quote_id"],
                            "prompt_type": f"{prompt_name}_{context_variant}",
                            "base_prompt_type": prompt_name,
                            "context_variant": context_variant,
                            "run": run_id,
                            "author": row[author_col],
                            "gender": row[gender_col] if gender_col is not None else "",
                            "race_ethnicity": row[race_col],
                            "quote": row["quote"],
                            "rag_top_k": top_k,
                            "rag_context": context,
                        }
                    )

    flush_print(f" Generating {len(all_prompts)} outputs...")
    flush_print(f"    Model: {model}")
    flush_print("    Backend: together_batch")
    flush_print(f"    Checkpointing every: {save_every}")
    flush_print(f"    Poll every: {poll_every_s}s")

    start_time = time.time()
    all_outputs = generate_with_together_batch(
        all_prompts,
        checkpoint_path=checkpoint_path,
        artifacts_dir=artifacts_dir,
        model=model,
        save_every=save_every,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        poll_every_s=poll_every_s,
        max_requests_per_batch=max_requests_per_batch,
    )
    total_time = time.time() - start_time

    if len(metadata) != len(all_outputs):
        raise RuntimeError("Checkpoint/output length mismatch. Ensure resume uses identical args/input ordering.")

    records = [{**meta, "llm_output": output} for meta, output in zip(metadata, all_outputs)]
    results_df = pd.DataFrame(records)

    output_file = f"{out_dir}/raw_outputs.csv"
    results_df.to_csv(output_file, index=False)

    config = {
        "timestamp": datetime.now().isoformat(),
        "csv_path": csv_path,
        "subset_size": subset_size,
        "random_subset_size": random_subset_size,
        "random_subset_seed": random_subset_seed,
        "quote_range": quote_range,
        "runs": runs,
        "model": model,
        "backend": "together_batch",
        "total_quotes": len(df),
        "total_outputs": len(all_outputs),
        "top_k": top_k,
        "author_col": author_col,
        "gender_col": gender_col,
        "race_col": race_col,
        "retriever": "openai_embedding_cosine",
        "embedding_model": embedding_model,
        "embedding_cache_path": embedding_cache_path,
        "embedding_batch_size": embedding_batch_size,
        "max_full_sims_n": max_full_sims_n,
        "context_label_mode": context_label_mode,
        "exclude_self_match": exclude_self_match,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "poll_every_s": poll_every_s,
        "max_requests_per_batch": max_requests_per_batch,
        "batch_artifacts_dir": artifacts_dir,
        "total_time_minutes": total_time / 60,
    }
    with open(f"{out_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    rate = (len(all_prompts) / total_time) if total_time > 0 else 0
    flush_print(f"\nComplete. Outputs saved to: {output_file}")
    flush_print(f"    Total time: {total_time/60:.1f} minutes ({rate:.2f} req/s)")
    return out_dir


def generate_outputs(
    csv_path,
    subset_size=None,
    random_subset_size=None,
    random_subset_seed=None,
    quote_range=None,
    runs=1,
    model="deepseek-ai/DeepSeek-V3.1",
    out_dir="results/prompt_attribution_rag_together",
    save_every=50,
    resume_from=None,
    prompts=None,
    top_k=5,
    embedding_model="text-embedding-3-small",
    embedding_cache_path=None,
    embedding_batch_size=128,
    max_full_sims_n=15000,
    context_label_mode="both",
    exclude_self_match=False,
    max_tokens=128,
    temperature=0.7,
    top_p=0.95,
    poll_every_s=DEFAULT_POLL_EVERY_S,
    max_requests_per_batch=DEFAULT_MAX_REQUESTS_PER_BATCH,
):
    return asyncio.run(
        generate_outputs_async(
            csv_path,
            subset_size,
            random_subset_size,
            random_subset_seed,
            quote_range,
            runs,
            model,
            out_dir,
            save_every,
            resume_from,
            prompts,
            top_k,
            embedding_model,
            embedding_cache_path,
            embedding_batch_size,
            max_full_sims_n,
            context_label_mode,
            exclude_self_match,
            max_tokens,
            temperature,
            top_p,
            poll_every_s,
            max_requests_per_batch,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG attribution pipeline with Together Batch API.")
    parser.add_argument("--csv-path", default=DATA_PATH)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--random-subset-size", type=int, default=None)
    parser.add_argument("--random-subset-seed", type=int, default=None)
    parser.add_argument("--quote-start", type=int, default=None)
    parser.add_argument("--quote-end", type=int, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default=TOGETHER_MODELS[0], choices=TOGETHER_MODELS)
    parser.add_argument("--out-dir", default="results/prompt_attribution_rag_together")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Existing output directory to resume from (must contain checkpoint.json).",
    )
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-cache-path", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--max-full-sims-n", type=int, default=15000)
    parser.add_argument(
        "--context-label-mode",
        choices=["labeled", "unlabeled", "both"],
        default="labeled",
        help="Whether retrieved context includes author labels.",
    )
    parser.add_argument(
        "--exclude-self-match",
        action="store_true",
        help="Exclude the exact quote from its own retrieved neighbors (default: include self-match).",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--poll-every-s", type=int, default=DEFAULT_POLL_EVERY_S)
    parser.add_argument("--max-requests-per-batch", type=int, default=DEFAULT_MAX_REQUESTS_PER_BATCH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quote_range = None
    if args.quote_start is not None and args.quote_end is not None:
        quote_range = (args.quote_start, args.quote_end)

    flush_print("=" * 60)
    flush_print("STARTING TOGETHER BATCH RAG PROMPT ATTRIBUTION")
    flush_print("=" * 60)

    output_dir = generate_outputs(
        csv_path=args.csv_path,
        subset_size=args.subset_size,
        random_subset_size=args.random_subset_size,
        random_subset_seed=args.random_subset_seed,
        quote_range=quote_range,
        runs=args.runs,
        model=args.model,
        out_dir=args.out_dir,
        save_every=args.save_every,
        resume_from=args.resume_from,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        embedding_cache_path=args.embedding_cache_path,
        embedding_batch_size=args.embedding_batch_size,
        max_full_sims_n=args.max_full_sims_n,
        context_label_mode=args.context_label_mode,
        exclude_self_match=args.exclude_self_match,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        poll_every_s=args.poll_every_s,
        max_requests_per_batch=args.max_requests_per_batch,
    )

    flush_print("\n" + "=" * 60)
    flush_print("GENERATION COMPLETE!")
    flush_print("=" * 60)
    flush_print("\nTo analyze results, run:")
    flush_print(f"  python analyze_outputs.py {output_dir}")
