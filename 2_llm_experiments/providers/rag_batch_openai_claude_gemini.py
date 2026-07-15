#!/usr/bin/env python3
"""Evidence-conditioned (RAG) attribution runner for the closed frontier APIs.

The RAG arm for GPT-5.1, Claude 4.6 Sonnet, and Gemini 2.5 Flash-Lite, selected with
--provider. Open-weight models go through `../runners/rag_together_exp.py`; the
no-evidence counterpart for these same providers is `batch_experiments.py`.

Each quote is presented with retrieved neighbor quotes that name their authors, so
the correct attribution is available in the prompt — failure to use it is suppression.

Pipeline
--------
1. Embed every quote with an OpenAI embedding model (default text-embedding-3-small),
   cached to disk (--embedding-cache-path).
2. Retrieve the --top-k (default 5) nearest neighbors by cosine similarity and render
   them into the {context} slot of RAG_PROMPTS (`format_context`).
3. Dispatch per provider:
     openai  -> Batch API (client.batches), ~24h turnaround
     claude  -> Message Batches API (client.beta.messages.batches)
     gemini  -> live concurrent calls with a rate limiter, NOT the batch API
                (`generate_with_gemini_concurrent`)

Usage
-----
  python rag_batch_openai_claude_gemini.py --provider openai \
      --csv-path 1_dataset_construction/datasets/intersectional_with_quotes.csv --runs 3

Use --subset-size / --random-subset-size for a smoke test before a full run. Requires
OPENAI_API_KEY (always, for embeddings) plus the selected provider's key.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI, OpenAI

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

SYSTEM_PROMPT = (
    "You are a helpful assistant that provides concise, factual responses. "
    "Answer concisely. Do NOT repeat the quote. Answer in maximum 1 sentence."
)

OPENAI_DEFAULT_MODEL = "gpt-5.1"
CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-6"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-lite"

DEFAULT_POLL_EVERY_S = 60
DEFAULT_MAX_REQUESTS_PER_BATCH = 20000
DEFAULT_GEMINI_MAX_CONCURRENCY = 20
DEFAULT_GEMINI_RATE_LIMIT_RPM = 25.0

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
GEMINI_MAX_RETRIES = 8
GEMINI_BASE_BACKOFF_SECONDS = 1.5
GEMINI_MAX_BACKOFF_SECONDS = 60.0
GEMINI_REQUEST_TIMEOUT_SECONDS = 120


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def quote_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_model_tag(model_name: str) -> str:
    return model_name.replace("/", "_").replace(" ", "_").replace(".", "_")


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


def load_embedding_cache(cache_path: str):
    if cache_path.endswith(".json"):
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cache = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
            flush_print(f"Loaded legacy embedding cache: {len(cache)} vectors")
            return cache
        return {}

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        hashes = data["hashes"]
        embeddings = data["embeddings"]
        cache = {str(h): embeddings[i] for i, h in enumerate(hashes.tolist())}
        flush_print(f"Loaded embedding cache: {len(cache)} vectors")
        return cache
    return {}


def save_embedding_cache(cache, cache_path: str):
    directory = os.path.dirname(cache_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    hashes = np.array(list(cache.keys()), dtype="<U64")
    embeddings = np.vstack([np.asarray(cache[h], dtype=np.float32) for h in hashes.tolist()])
    np.savez(cache_path, hashes=hashes, embeddings=embeddings)
    flush_print(f"Saved embedding cache: {len(cache)} vectors")


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
        flush_print(f"Embedding {len(missing)} new quotes with {embedding_model}...")
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            batch_texts = [text for _, text in batch]
            vectors = await embed_texts_batch(client, batch_texts, embedding_model=embedding_model)
            for (h, _), vec in zip(batch, vectors):
                cache[h] = np.asarray(vec, dtype=np.float32)
            flush_print(f"Embedded {min(start + batch_size, len(missing))}/{len(missing)}")
        save_embedding_cache(cache, cache_path)
    else:
        flush_print("All quote embeddings found in cache.")

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


def normalize_cell(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def quote_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_cell(text).encode("utf-8")).hexdigest()


def build_retrieval_row_lookup(df: pd.DataFrame) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    by_quote_id: Dict[str, List[int]] = {}
    by_quote_hash: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        qid = normalize_cell(row.get("quote_id"))
        qhash = quote_text_hash(row.get("quote"))
        by_quote_id.setdefault(qid, []).append(int(idx))
        by_quote_hash.setdefault(qhash, []).append(int(idx))
    return by_quote_id, by_quote_hash


def choose_retrieval_idx_for_target_row(
    row_like,
    retrieval_df: pd.DataFrame,
    retrieval_by_quote_id: Dict[str, List[int]],
    retrieval_by_quote_hash: Dict[str, List[int]],
) -> Optional[int]:
    qid = normalize_cell(row_like.get("quote_id"))
    quote = normalize_cell(row_like.get("quote"))
    qhash = quote_text_hash(quote)

    # Prefer quote_id match when present.
    cand = retrieval_by_quote_id.get(qid, [])
    if cand:
        if len(cand) == 1:
            return cand[0]
        for idx in cand:
            if normalize_cell(retrieval_df.iloc[idx].get("quote")) == quote:
                return idx
        return cand[0]

    # Fallback to quote text hash.
    cand = retrieval_by_quote_hash.get(qhash, [])
    if cand:
        return cand[0]
    return None


def build_reuse_map(reuse_paths: List[str]) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    out: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for p in reuse_paths:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"--reuse-results-csv not found: {p}")
        prev = pd.read_csv(path)
        required = {"quote_id", "prompt_type", "run", "llm_output", "quote", "rag_context"}
        missing = sorted(required - set(prev.columns))
        if missing:
            raise SystemExit(f"--reuse-results-csv missing columns {missing}: {p}")
        for _, row in prev.iterrows():
            key = (
                normalize_cell(row.get("quote_id")),
                normalize_cell(row.get("prompt_type")),
                int(row.get("run")),
            )
            out[key] = {
                "llm_output": normalize_cell(row.get("llm_output")),
                "quote": normalize_cell(row.get("quote")),
                "rag_context": normalize_cell(row.get("rag_context")),
            }
    return out


def extract_openai_output_text_from_body(body: dict) -> str:
    if not isinstance(body, dict):
        return ""

    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks = []
    for item in body.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if part.get("type") in ("output_text", "text"):
                text = part.get("text")
                if text:
                    chunks.append(text)

    return "".join(chunks).strip()


def wait_openai_batch(client: OpenAI, batch_id: str, poll_every_s: int):
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        rc = getattr(batch, "request_counts", None)
        if rc is not None:
            flush_print(
                f"[{datetime.now().strftime('%H:%M:%S')}] openai status={batch.status} "
                f"completed={getattr(rc, 'completed', None)} failed={getattr(rc, 'failed', None)} "
                f"total={getattr(rc, 'total', None)}"
            )
        else:
            flush_print(f"[{datetime.now().strftime('%H:%M:%S')}] openai status={batch.status}")

        if batch.status in terminal_statuses:
            return batch
        time.sleep(poll_every_s)


def parse_openai_batch_output_jsonl(path: Path) -> Dict[int, str]:
    indexed: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            custom_id = str(obj.get("custom_id", ""))
            if not custom_id.startswith("req_"):
                continue
            try:
                idx = int(custom_id.replace("req_", ""))
            except ValueError:
                continue
            response_obj = obj.get("response") if isinstance(obj.get("response"), dict) else {}
            body = response_obj.get("body") if isinstance(response_obj.get("body"), dict) else {}
            indexed[idx] = extract_openai_output_text_from_body(body)
    return indexed


def generate_with_openai_batch(
    prompts: List[str],
    artifacts_dir: Path,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    reasoning_effort: str,
    poll_every_s: int,
    max_requests_per_batch: int,
) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []

    total = len(prompts)
    n_blocks = (total + max_requests_per_batch - 1) // max_requests_per_batch if total else 0

    for block_idx, start in enumerate(range(0, total, max_requests_per_batch), start=1):
        end = min(start + max_requests_per_batch, total)
        block_prompts = prompts[start:end]

        shard_prefix = f"batch_{start:09d}_{end - 1:09d}"
        input_jsonl = artifacts_dir / f"{shard_prefix}_input.jsonl"
        output_jsonl = artifacts_dir / f"{shard_prefix}_output.jsonl"
        batch_info_json = artifacts_dir / f"{shard_prefix}_batch_info.json"

        with input_jsonl.open("w", encoding="utf-8") as f:
            for local_idx, prompt in enumerate(block_prompts):
                global_idx = start + local_idx
                rec = {
                    "custom_id": f"req_{global_idx}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        "input": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_output_tokens": max_output_tokens,
                        "reasoning": {"effort": reasoning_effort},
                        "text": {"verbosity": "low"},
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        flush_print(f"OpenAI block {block_idx}/{n_blocks}: items {start}..{end - 1}")
        with input_jsonl.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"runner": "rag_batch_openai_claude_gemini", "model": model},
        )
        flush_print(f"OpenAI batch_id={batch.id}")

        final_batch = wait_openai_batch(client, batch.id, poll_every_s=poll_every_s)
        batch_info = {
            "batch_id": final_batch.id,
            "status": getattr(final_batch, "status", None),
            "input_file_id": getattr(final_batch, "input_file_id", None),
            "output_file_id": getattr(final_batch, "output_file_id", None),
            "error_file_id": getattr(final_batch, "error_file_id", None),
            "updated_at": datetime.now().isoformat(),
        }
        batch_info_json.write_text(json.dumps(batch_info, indent=2), encoding="utf-8")

        if final_batch.status not in {"completed", "finalizing"}:
            raise RuntimeError(f"OpenAI batch failed: status={final_batch.status} batch_id={final_batch.id}")

        output_file_id = getattr(final_batch, "output_file_id", None)
        if not output_file_id:
            # brief wait for output file materialization
            for _ in range(18):
                time.sleep(10)
                refreshed = client.batches.retrieve(final_batch.id)
                output_file_id = getattr(refreshed, "output_file_id", None)
                if output_file_id:
                    break
        if not output_file_id:
            raise RuntimeError(f"OpenAI batch completed without output_file_id: {final_batch.id}")

        content = client.files.content(output_file_id)
        data = content.read()
        if isinstance(data, (bytes, bytearray)):
            output_jsonl.write_bytes(data)
        else:
            output_jsonl.write_text(str(data), encoding="utf-8")

        indexed = parse_openai_batch_output_jsonl(output_jsonl)
        missing = 0
        for idx in range(start, end):
            text = indexed.get(idx, "")
            if idx not in indexed:
                missing += 1
            outputs.append(text)
        if missing:
            flush_print(f"OpenAI block missing {missing} outputs (filled empty)")

    return outputs


def parse_claude_results(results_iterable) -> Dict[int, str]:
    indexed: Dict[int, str] = {}
    for r in results_iterable:
        custom_id = str(getattr(r, "custom_id", ""))
        if not custom_id.startswith("req_"):
            continue
        try:
            idx = int(custom_id.replace("req_", ""))
        except ValueError:
            continue

        result = getattr(r, "result", None)
        result_type = getattr(result, "type", "")
        if result_type != "succeeded":
            indexed[idx] = ""
            continue

        text = ""
        try:
            content = result.message.content
            if content:
                text = str(content[0].text).strip()
        except Exception:
            text = ""
        indexed[idx] = text

    return indexed


def generate_with_claude_batch(
    prompts: List[str],
    artifacts_dir: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget_tokens: int,
    poll_every_s: int,
    max_requests_per_batch: int,
) -> List[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []

    total = len(prompts)
    n_blocks = (total + max_requests_per_batch - 1) // max_requests_per_batch if total else 0

    for block_idx, start in enumerate(range(0, total, max_requests_per_batch), start=1):
        end = min(start + max_requests_per_batch, total)
        block_prompts = prompts[start:end]

        shard_prefix = f"batch_{start:09d}_{end - 1:09d}"
        batch_info_json = artifacts_dir / f"{shard_prefix}_batch_info.json"

        requests = []
        for local_idx, prompt in enumerate(block_prompts):
            global_idx = start + local_idx
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt}],
            }
            if thinking_budget_tokens > 0:
                params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
            requests.append(
                {
                    "custom_id": f"req_{global_idx}",
                    "params": params,
                }
            )

        flush_print(f"Claude block {block_idx}/{n_blocks}: items {start}..{end - 1}")
        batch = client.beta.messages.batches.create(requests=requests)
        batch_id = batch.id
        flush_print(f"Claude batch_id={batch_id}")

        while True:
            polled = client.beta.messages.batches.retrieve(batch_id)
            status = polled.processing_status
            flush_print(f"[{datetime.now().strftime('%H:%M:%S')}] claude status={status}")
            if status in {"ended", "errored", "canceled"}:
                break
            time.sleep(poll_every_s)

        batch_info = {
            "batch_id": batch_id,
            "status": polled.processing_status,
            "updated_at": datetime.now().isoformat(),
        }
        batch_info_json.write_text(json.dumps(batch_info, indent=2), encoding="utf-8")

        if polled.processing_status != "ended":
            raise RuntimeError(f"Claude batch failed: status={polled.processing_status} batch_id={batch_id}")

        results = client.beta.messages.batches.results(batch_id)
        indexed = parse_claude_results(results)

        missing = 0
        for idx in range(start, end):
            text = indexed.get(idx, "")
            if idx not in indexed:
                missing += 1
            outputs.append(text)
        if missing:
            flush_print(f"Claude block missing {missing} outputs (filled empty)")

    return outputs


def extract_text_from_gemini_response(response_obj) -> str:
    try:
        text = getattr(response_obj, "text", None)
        if text:
            return str(text).strip()
    except Exception:
        pass

    try:
        candidates = getattr(response_obj, "candidates", None) or []
        if candidates:
            parts = candidates[0].content.parts
            texts = []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
            return "".join(texts).strip()
    except Exception:
        pass
    return ""


class GeminiRateLimiter:
    def __init__(self, rpm: float):
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self.delay = 60.0 / rpm
        self.lock = asyncio.Lock()
        self.last_call = 0.0

    async def wait(self) -> None:
        async with self.lock:
            now = time.time()
            wait_time = self.delay - (now - self.last_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = time.time()


class GeminiCooldownController:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cooldown_until = 0.0

    async def wait_if_needed(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                remaining = self._cooldown_until - now
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 2.0))

    async def trigger(self, seconds: float) -> None:
        async with self._lock:
            now = time.monotonic()
            self._cooldown_until = max(self._cooldown_until, now + seconds)


def extract_status_code(exc: Exception) -> int:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    txt = str(exc)
    for c in (429, 500, 502, 503, 504):
        if str(c) in txt:
            return c
    return 0


async def generate_one_gemini_with_retries(
    client: genai.Client,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    cooldown: GeminiCooldownController,
    rate_limiter: GeminiRateLimiter,
) -> str:
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        await cooldown.wait_if_needed()
        await rate_limiter.wait()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_output_tokens,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                ),
                timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
            )
            return extract_text_from_gemini_response(response)
        except asyncio.TimeoutError:
            if attempt >= GEMINI_MAX_RETRIES:
                return ""
            await asyncio.sleep(min(60, 2**attempt))
        except Exception as exc:
            code = extract_status_code(exc)
            if attempt >= GEMINI_MAX_RETRIES or code not in RETRYABLE_STATUS_CODES:
                return ""
            if code == 429:
                cooldown_secs = min(45.0, 10.0 + 5.0 * attempt)
                await cooldown.trigger(cooldown_secs)
            sleep_base = min(GEMINI_MAX_BACKOFF_SECONDS, GEMINI_BASE_BACKOFF_SECONDS * (2**attempt))
            sleep_for = sleep_base + random.uniform(0.0, min(2.0, sleep_base * 0.2))
            await asyncio.sleep(sleep_for)
    return ""


async def generate_with_gemini_concurrent(
    prompts: List[str],
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_concurrency: int,
    rate_limit_rpm: float,
) -> List[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)
    max_concurrency = max(1, max_concurrency)
    rate_limiter = GeminiRateLimiter(rate_limit_rpm)
    cooldown = GeminiCooldownController()
    sem = asyncio.Semaphore(max_concurrency)
    outputs: List[str] = [""] * len(prompts)
    done = 0
    done_lock = asyncio.Lock()

    flush_print(
        f"Gemini concurrent mode: requests={len(prompts)} "
        f"max_concurrency={max_concurrency} rate_limit_rpm={rate_limit_rpm}"
    )

    async def run_one(i: int, prompt: str):
        nonlocal done
        async with sem:
            outputs[i] = await generate_one_gemini_with_retries(
                client=client,
                prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                cooldown=cooldown,
                rate_limiter=rate_limiter,
            )
        async with done_lock:
            done += 1
            if done % 200 == 0 or done == len(prompts):
                flush_print(f"Gemini progress: {done}/{len(prompts)}")

    await asyncio.gather(*(run_one(i, p) for i, p in enumerate(prompts)))
    return outputs


async def run_async(args):
    load_dotenv()

    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    if args.provider == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")
    if args.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (required for embeddings)")

    target_df = pd.read_csv(args.csv_path)
    flush_print(f"Loaded {len(target_df)} target quotes from {args.csv_path}")

    retrieval_corpus_csv = args.retrieval_corpus_csv.strip() if args.retrieval_corpus_csv else ""
    if retrieval_corpus_csv:
        retrieval_df = pd.read_csv(retrieval_corpus_csv)
        flush_print(f"Loaded {len(retrieval_df)} retrieval quotes from {retrieval_corpus_csv}")
    else:
        retrieval_df = target_df.copy()
        retrieval_corpus_csv = args.csv_path

    target_author_col, target_race_col, target_gender_col = infer_columns(target_df)
    retrieval_author_col, _, _ = infer_columns(retrieval_df)

    if "quote_id" not in target_df.columns:
        target_df["quote_id"] = range(len(target_df))
    if "quote_id" not in retrieval_df.columns:
        retrieval_df["quote_id"] = range(len(retrieval_df))

    if args.quote_range:
        start_idx, end_idx = args.quote_range
        target_df = target_df.iloc[start_idx - 1 : end_idx].reset_index(drop=True)
        flush_print(f"Using quote range {start_idx}..{end_idx} ({len(target_df)} rows)")
    elif args.random_subset_size:
        n = min(args.random_subset_size, len(target_df))
        target_df = target_df.sample(n=n, random_state=args.random_subset_seed).reset_index(drop=True)
        flush_print(f"Using random subset n={n} seed={args.random_subset_seed}")
    elif args.subset_size:
        target_df = target_df.sample(frac=1, random_state=42).reset_index(drop=True)
        target_df = target_df.head(args.subset_size)
        flush_print(f"Using subset_size={len(target_df)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = args.model
    provider = args.provider
    model_tag = sanitize_model_tag(model)
    dataset_tag = Path(args.csv_path).stem
    out_dir = Path(args.out_dir) / f"prompt_attribution_rag_{provider}_{model_tag}_{dataset_tag}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "batch_artifacts"

    embedding_cache_path = args.embedding_cache_path
    if not embedding_cache_path:
        safe_model = args.embedding_model.replace("/", "_")
        embedding_cache_path = f"results/rag_embedding_cache_{safe_model}.npz"

    embed_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    flush_print(f"Building embedding retriever ({args.embedding_model})")
    matrix = await build_embedding_matrix(
        df=retrieval_df,
        quote_col="quote",
        embedding_model=args.embedding_model,
        cache_path=embedding_cache_path,
        client=embed_client,
        batch_size=args.embedding_batch_size,
    )
    top_idx, top_scores = build_topk_neighbors(
        matrix,
        top_k=args.top_k,
        max_full_sims_n=args.max_full_sims_n,
        exclude_self=args.exclude_self_match,
    )

    if args.context_label_mode == "both":
        context_variants = [("labeled", True), ("unlabeled", False)]
    elif args.context_label_mode == "labeled":
        context_variants = [("labeled", True)]
    else:
        context_variants = [("unlabeled", False)]

    retrieval_by_quote_id, retrieval_by_quote_hash = build_retrieval_row_lookup(retrieval_df)

    all_prompts = []
    metadata = []
    missing_match_count = 0
    for _, row in target_df.iterrows():
        retrieval_idx = choose_retrieval_idx_for_target_row(
            row_like=row,
            retrieval_df=retrieval_df,
            retrieval_by_quote_id=retrieval_by_quote_id,
            retrieval_by_quote_hash=retrieval_by_quote_hash,
        )
        if retrieval_idx is None:
            missing_match_count += 1
            continue
        retrieved = retrieve_neighbors(
            query_idx=retrieval_idx,
            top_idx=top_idx,
            top_scores=top_scores,
            df=retrieval_df,
            author_col=retrieval_author_col,
        )
        for context_variant, include_author in context_variants:
            context = format_context(retrieved, include_author=include_author)
            for prompt_name, template in RAG_PROMPTS.items():
                for run_id in range(args.runs):
                    all_prompts.append(template.format(passage=row["quote"], context=context))
                    metadata.append(
                        {
                            "quote_id": row["quote_id"],
                            "prompt_type": f"{prompt_name}_{context_variant}",
                            "base_prompt_type": prompt_name,
                            "context_variant": context_variant,
                            "run": run_id,
                            "author": row[target_author_col],
                            "gender": row[target_gender_col] if target_gender_col is not None else "",
                            "race_ethnicity": row[target_race_col],
                            "quote": row["quote"],
                            "rag_top_k": args.top_k,
                            "rag_context": context,
                            "retrieval_corpus_csv": retrieval_corpus_csv,
                        }
                    )

    if missing_match_count > 0:
        flush_print(f"Warning: skipped {missing_match_count} target rows with no retrieval-corpus match")

    reuse_map = build_reuse_map(args.reuse_results_csv)
    pending_prompts: List[str] = []
    pending_meta: List[dict] = []
    all_outputs: List[str] = [""] * len(metadata)
    reused_count = 0
    for i, meta in enumerate(metadata):
        key = (
            normalize_cell(meta.get("quote_id")),
            normalize_cell(meta.get("prompt_type")),
            int(meta.get("run")),
        )
        prev = reuse_map.get(key)
        if prev is not None:
            if (
                normalize_cell(prev.get("quote")) == normalize_cell(meta.get("quote"))
                and normalize_cell(prev.get("rag_context")) == normalize_cell(meta.get("rag_context"))
            ):
                all_outputs[i] = normalize_cell(prev.get("llm_output"))
                reused_count += 1
                continue
        pending_prompts.append(all_prompts[i])
        pending_meta.append(meta)

    flush_print(
        f"Prepared {len(all_prompts)} outputs via {provider} batch "
        f"(reused={reused_count}, pending={len(pending_prompts)})"
    )

    if pending_prompts and provider == "openai":
        pending_outputs = generate_with_openai_batch(
            prompts=pending_prompts,
            artifacts_dir=artifacts_dir,
            model=model,
            temperature=args.openai_temperature,
            top_p=args.openai_top_p,
            max_output_tokens=args.openai_max_output_tokens,
            reasoning_effort=args.openai_reasoning_effort,
            poll_every_s=args.poll_every_s,
            max_requests_per_batch=args.max_requests_per_batch,
        )
        sampling_config = {
            "openai_temperature": args.openai_temperature,
            "openai_top_p": args.openai_top_p,
            "openai_max_output_tokens": args.openai_max_output_tokens,
            "openai_reasoning_effort": args.openai_reasoning_effort,
        }
    elif pending_prompts and provider == "claude":
        pending_outputs = generate_with_claude_batch(
            prompts=pending_prompts,
            artifacts_dir=artifacts_dir,
            model=model,
            temperature=args.claude_temperature,
            max_tokens=args.claude_max_tokens,
            thinking_budget_tokens=args.claude_thinking_budget_tokens,
            poll_every_s=args.poll_every_s,
            max_requests_per_batch=args.max_requests_per_batch,
        )
        sampling_config = {
            "claude_temperature": args.claude_temperature,
            "claude_max_tokens": args.claude_max_tokens,
            "claude_thinking_budget_tokens": args.claude_thinking_budget_tokens,
        }
    elif pending_prompts and provider == "gemini":
        pending_outputs = await generate_with_gemini_concurrent(
            prompts=pending_prompts,
            model=model,
            temperature=args.gemini_temperature,
            top_p=args.gemini_top_p,
            max_output_tokens=args.gemini_max_output_tokens,
            max_concurrency=args.gemini_max_concurrency,
            rate_limit_rpm=args.gemini_rate_limit_rpm,
        )
        sampling_config = {
            "gemini_temperature": args.gemini_temperature,
            "gemini_top_p": args.gemini_top_p,
            "gemini_max_output_tokens": args.gemini_max_output_tokens,
            "gemini_max_concurrency": args.gemini_max_concurrency,
            "gemini_rate_limit_rpm": args.gemini_rate_limit_rpm,
        }
    else:
        pending_outputs = []
        sampling_config = (
            {
                "openai_temperature": args.openai_temperature,
                "openai_top_p": args.openai_top_p,
                "openai_max_output_tokens": args.openai_max_output_tokens,
                "openai_reasoning_effort": args.openai_reasoning_effort,
            }
            if provider == "openai"
            else (
                {
                    "claude_temperature": args.claude_temperature,
                    "claude_max_tokens": args.claude_max_tokens,
                    "claude_thinking_budget_tokens": args.claude_thinking_budget_tokens,
                }
                if provider == "claude"
                else {
                    "gemini_temperature": args.gemini_temperature,
                    "gemini_top_p": args.gemini_top_p,
                    "gemini_max_output_tokens": args.gemini_max_output_tokens,
                    "gemini_max_concurrency": args.gemini_max_concurrency,
                    "gemini_rate_limit_rpm": args.gemini_rate_limit_rpm,
                }
            )
        )

    if len(pending_outputs) != len(pending_meta):
        raise RuntimeError(f"Pending output length mismatch: outputs={len(pending_outputs)} metadata={len(pending_meta)}")

    pending_iter = iter(pending_outputs)
    for i, out in enumerate(all_outputs):
        if out != "":
            continue
        all_outputs[i] = next(pending_iter, "")

    if len(all_outputs) != len(metadata):
        raise RuntimeError(f"Output length mismatch: outputs={len(all_outputs)} metadata={len(metadata)}")

    results_df = pd.DataFrame([{**m, "llm_output": o} for m, o in zip(metadata, all_outputs)])
    output_csv = out_dir / "raw_outputs.csv"
    results_df.to_csv(output_csv, index=False)

    config = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "csv_path": args.csv_path,
        "retrieval_corpus_csv": retrieval_corpus_csv,
        "runs": args.runs,
        "subset_size": args.subset_size,
        "random_subset_size": args.random_subset_size,
        "random_subset_seed": args.random_subset_seed,
        "quote_range": args.quote_range,
        "top_k": args.top_k,
        "embedding_model": args.embedding_model,
        "embedding_cache_path": embedding_cache_path,
        "embedding_batch_size": args.embedding_batch_size,
        "max_full_sims_n": args.max_full_sims_n,
        "context_label_mode": args.context_label_mode,
        "exclude_self_match": args.exclude_self_match,
        "poll_every_s": args.poll_every_s,
        "max_requests_per_batch": args.max_requests_per_batch,
        "total_target_quotes": len(target_df),
        "total_retrieval_quotes": len(retrieval_df),
        "missing_target_retrieval_matches": missing_match_count,
        "reused_outputs": reused_count,
        "pending_outputs_submitted": len(pending_prompts),
        "total_outputs": len(all_outputs),
        "reuse_results_csv": args.reuse_results_csv,
        **sampling_config,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    flush_print(f"Complete. Outputs saved: {output_csv}")
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run Together-style RAG prompts through OpenAI, Claude, or Gemini batch APIs.")
    parser.add_argument("--provider", choices=["openai", "claude", "gemini"], required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument(
        "--retrieval-corpus-csv",
        default="",
        help="Optional retrieval corpus CSV. If omitted, uses --csv-path as both target and retrieval corpus.",
    )
    parser.add_argument(
        "--reuse-results-csv",
        action="append",
        default=[],
        help="Optional prior raw_outputs.csv path(s) to reuse exact matches by quote_id/prompt_type/run + quote + rag_context.",
    )

    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--random-subset-size", type=int, default=0)
    parser.add_argument("--random-subset-seed", type=int, default=42)
    parser.add_argument("--quote-range", type=int, nargs=2, default=None, metavar=("START", "END"))

    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-cache-path", default="")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--max-full-sims-n", type=int, default=15000)
    parser.add_argument("--context-label-mode", choices=["labeled", "unlabeled", "both"], default="labeled")
    parser.add_argument("--exclude-self-match", action="store_true")

    parser.add_argument("--openai-temperature", type=float, default=0.7)
    parser.add_argument("--openai-top-p", type=float, default=0.95)
    parser.add_argument("--openai-max-output-tokens", type=int, default=500)
    parser.add_argument("--openai-reasoning-effort", choices=["none", "low", "medium", "high"], default="none")

    parser.add_argument("--claude-temperature", type=float, default=0.7)
    parser.add_argument("--claude-max-tokens", type=int, default=500)
    parser.add_argument("--claude-thinking-budget-tokens", type=int, default=0)

    parser.add_argument("--gemini-temperature", type=float, default=0.7)
    parser.add_argument("--gemini-top-p", type=float, default=0.95)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=500)
    parser.add_argument("--gemini-max-concurrency", type=int, default=DEFAULT_GEMINI_MAX_CONCURRENCY)
    parser.add_argument("--gemini-rate-limit-rpm", type=float, default=DEFAULT_GEMINI_RATE_LIMIT_RPM)

    parser.add_argument("--poll-every-s", type=int, default=DEFAULT_POLL_EVERY_S)
    parser.add_argument("--max-requests-per-batch", type=int, default=DEFAULT_MAX_REQUESTS_PER_BATCH)

    parser.add_argument(
        "--out-dir",
        default="results/rag_openai_claude_batch",
        help="Base output directory; script appends provider/model/dataset/timestamp subdir.",
    )

    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.max_requests_per_batch < 1:
        raise SystemExit("--max-requests-per-batch must be >= 1")
    if args.claude_thinking_budget_tokens < 0:
        raise SystemExit("--claude-thinking-budget-tokens must be >= 0")
    if args.gemini_max_concurrency < 1:
        raise SystemExit("--gemini-max-concurrency must be >= 1")
    if args.gemini_rate_limit_rpm <= 0:
        raise SystemExit("--gemini-rate-limit-rpm must be > 0")

    if not args.model:
        if args.provider == "openai":
            args.model = OPENAI_DEFAULT_MODEL
        elif args.provider == "claude":
            args.model = CLAUDE_DEFAULT_MODEL
        else:
            args.model = GEMINI_DEFAULT_MODEL

    args.subset_size = args.subset_size if args.subset_size > 0 else None
    args.random_subset_size = args.random_subset_size if args.random_subset_size > 0 else None

    return args


def main():
    args = parse_args()
    run_dir = asyncio.run(run_async(args))
    flush_print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
