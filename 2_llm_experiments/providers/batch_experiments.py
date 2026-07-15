#!/usr/bin/env python3
"""
Unified zero-shot batch attribution runner for OpenAI, Anthropic (Claude), and Google (Gemini).

The attribution prompts, system prompt, dataset loading, and manifest schema are shared; each
provider keeps its exact request body and batch-API submission/polling logic, so re-running any
provider reproduces the same model-facing payloads as before.

Usage:
    python batch_experiments.py --provider openai --csv-path DATA.csv --runs 3
    python batch_experiments.py --provider claude --csv-path DATA.csv --runs 3
    python batch_experiments.py --provider gemini --csv-path DATA.csv --runs 3 [--batch-size 1200]

Each provider writes to results/<provider>_batch/<dataset-stem>/raw_outputs.csv (override the
results root with the ATTRIBENCH_RESULTS env var via the repo's paths convention, or --run-label).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# Shared config
# ----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("ATTRIBENCH_ROOT", Path(__file__).resolve().parents[2])
)
load_dotenv(PROJECT_ROOT / ".env")

# Attribution prompts — identical across all providers.
PROMPTS = {
    "direct": (
        "Please identify the author of the following quote. "
        "Be concise and answer in maximum 1 sentence.\n\"{passage}\""
    ),
    "indirect": (
        "Briefly summarize the context in which the following quote was written. "
        "Be concise and answer in maximum 1 sentence.\n\"{passage}\""
    ),
    "indirect_overt": (
        "Briefly summarize the context in which the following quote was written, mentioning the author if relevant. "
        "Be concise and answer in maximum 1 sentence.\n\"{passage}\""
    ),
}

SYSTEM_PROMPT = (
    "You are a helpful assistant that provides concise, factual responses. "
    "Answer concisely. Do NOT repeat the quote. Answer in maximum 1 sentence."
)

# Generation parameters — shared where the provider supports them (see per-provider notes).
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 500

RESULTS_ROOT = Path(
    os.environ.get("ATTRIBENCH_RESULTS", PROJECT_ROOT / "results")
)
POLL_INTERVAL_SECONDS = 30


# ----------------------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------------------

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def normalize_cell(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(csv_path: str, smoke_rows: Optional[int] = None) -> pd.DataFrame:
    """Load an AttriBench CSV, validating the columns every provider needs.

    ``smoke_rows`` keeps the first N rows for a quick sanity run (head, not sampled).
    """
    df = pd.read_csv(csv_path)
    if smoke_rows is not None:
        df = df.head(smoke_rows).copy()

    author_col = "author" if "author" in df.columns else "author_clean" if "author_clean" in df.columns else None
    race_col = "race_ethnicity" if "race_ethnicity" in df.columns else "race" if "race" in df.columns else None

    if author_col is None:
        raise ValueError("Missing author column: need 'author' or 'author_clean'")
    if race_col is None:
        raise ValueError("Missing race column: need 'race_ethnicity' or 'race'")
    for col in ["quote", "gender"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "quote_id" not in df.columns:
        df["quote_id"] = range(len(df))

    df.attrs["author_col"] = author_col
    df.attrs["race_col"] = race_col
    return df


def base_manifest_row(row, prompt_name, run_id, author_col, race_col) -> Dict[str, Any]:
    """The manifest schema shared by every provider (the id column is added by each provider)."""
    return {
        "quote_id": row["quote_id"],
        "prompt_type": prompt_name,
        "run": run_id,
        "author": row[author_col],
        "gender": row["gender"],
        "race_ethnicity": row[race_col],
        "quote": row["quote"],
    }


def split_manifest_by_reusable_results(
    manifest_df: pd.DataFrame,
    reuse_results_csv: Optional[str],
    id_col: str,
    value_cols: List[str],
):
    """Split a manifest into (pending, reusable) by matching a prior raw_outputs.csv on ``id_col``."""
    empty_cols = list(manifest_df.columns) + [c for c in value_cols if c != id_col]
    if not reuse_results_csv:
        return manifest_df, pd.DataFrame(columns=empty_cols)

    reuse_path = Path(reuse_results_csv)
    if not reuse_path.exists():
        raise SystemExit(f"--reuse-results-csv not found: {reuse_results_csv}")

    prev = pd.read_csv(reuse_path)
    if id_col not in prev.columns:
        raise SystemExit(f"--reuse-results-csv missing '{id_col}' column: {reuse_results_csv}")

    present_cols = [c for c in value_cols if c in prev.columns]
    prev_small = prev[present_cols].drop_duplicates(subset=[id_col], keep="last")

    reusable = manifest_df.merge(prev_small, on=id_col, how="inner")
    pending = manifest_df.loc[~manifest_df[id_col].isin(set(reusable[id_col]))].copy()
    return pending, reusable


# ============================================================================
# Anthropic (Claude) — single Message Batches submission
# ============================================================================

CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_RESULTS_ROOT = RESULTS_ROOT / "claude_batch"


def claude_get_client():
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def claude_build_requests_and_manifest(df, prompts, runs):
    author_col = df.attrs["author_col"]
    race_col = df.attrs["race_col"]

    requests = []
    manifest_rows = []

    for _, row in df.iterrows():
        for prompt_name, template in prompts.items():
            for run in range(runs):
                custom_id = f"{row['quote_id']}_{prompt_name}_{run}"
                user_prompt = template.format(passage=row["quote"])

                requests.append({
                    "custom_id": custom_id,
                    "params": {
                        "model": CLAUDE_MODEL,
                        "max_tokens": MAX_OUTPUT_TOKENS,
                        "temperature": TEMPERATURE,
                        "messages": [
                            {
                                "role": "user",
                                "content": SYSTEM_PROMPT + "\n\n" + user_prompt,
                            }
                        ],
                    },
                })

                manifest_rows.append({"custom_id": custom_id, **base_manifest_row(row, prompt_name, run, author_col, race_col)})

    return requests, pd.DataFrame(manifest_rows)


def claude_submit_batch(client, requests):
    return client.beta.messages.batches.create(requests=requests)


def claude_poll_batch(client, batch_id):
    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        flush_print(f"[{datetime.now().strftime('%H:%M:%S')}] status={status}")
        if status in ["ended", "errored", "canceled"]:
            return batch
        time.sleep(POLL_INTERVAL_SECONDS)


def claude_fetch_results(client, batch_id):
    results = client.beta.messages.batches.results(batch_id)
    rows = []
    for r in results:
        result_type = getattr(r.result, "type", "")
        text = ""
        error_text = ""
        if result_type == "succeeded":
            try:
                text = r.result.message.content[0].text
            except Exception as exc:
                error_text = f"parse_error: {exc}"
        elif result_type == "errored":
            try:
                error_text = json.dumps(r.result.error.model_dump(), ensure_ascii=False)
            except Exception:
                error_text = str(getattr(r.result, "error", "unknown_error"))
        else:
            error_text = result_type
        rows.append({
            "custom_id": r.custom_id,
            "result_type": result_type,
            "llm_output": text,
            "error": error_text,
        })
    return pd.DataFrame(rows)


def claude_merge(manifest, outputs):
    df = manifest.merge(outputs, on="custom_id", how="left")
    df["llm_output"] = df["llm_output"].fillna("")
    return df


def run_claude(csv_path, runs, results_root, smoke_rows=None, reuse_results_csv=None):
    df = load_dataset(csv_path, smoke_rows=smoke_rows)
    requests, manifest = claude_build_requests_and_manifest(df, PROMPTS, runs)
    pending_manifest, reused_rows = split_manifest_by_reusable_results(
        manifest, reuse_results_csv, id_col="custom_id",
        value_cols=["custom_id", "result_type", "llm_output", "error"],
    )

    pending_ids = set(pending_manifest["custom_id"])
    pending_requests = [r for r in requests if r["custom_id"] in pending_ids]

    flush_print(f"Prepared {len(manifest)} total requests")
    flush_print(f"Reusing {len(reused_rows)} prior results")
    flush_print(f"Submitting {len(pending_requests)} new requests")

    if pending_requests:
        client = claude_get_client()
        batch = claude_submit_batch(client, pending_requests)
        batch_id = batch.id
        flush_print(f"Batch ID: {batch_id}")
        claude_poll_batch(client, batch_id)
        flush_print("Fetching results...")
        outputs = claude_fetch_results(client, batch_id)
        pending_final = claude_merge(pending_manifest, outputs)
    else:
        pending_final = pd.DataFrame(columns=list(manifest.columns) + ["result_type", "llm_output", "error"])

    final = pd.concat([reused_rows, pending_final], ignore_index=True, sort=False)
    final["llm_output"] = final["llm_output"].fillna("")
    if "quote_id" in final.columns and "prompt_type" in final.columns and "run" in final.columns:
        final = final.sort_values(by=["quote_id", "prompt_type", "run"], kind="mergesort").reset_index(drop=True)

    out_dir = results_root / Path(csv_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_outputs.csv"
    final.to_csv(out_path, index=False)
    flush_print(f"Saved -> {out_path}")


# ============================================================================
# Google (Gemini) — chunked file-batch submission with resource-exhausted retries
# ============================================================================

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "minimal")
GEMINI_RESULTS_ROOT = RESULTS_ROOT / "gemini_batch"


def gemini_get_client():
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def gemini_make_key(quote_id, prompt_name, run_idx):
    return f"{quote_id}_{prompt_name}_{run_idx}"


def gemini_build_requests_and_manifest(df, prompts, runs):
    from google.genai import types

    author_col = df.attrs["author_col"]
    race_col = df.attrs["race_col"]

    requests = []
    manifest_rows = []

    for _, row in df.iterrows():
        for prompt_name, template in prompts.items():
            for run_idx in range(runs):
                key = gemini_make_key(row["quote_id"], prompt_name, run_idx)
                user_prompt = template.format(passage=row["quote"])
                generation_config = {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                }
                # Only enable thinking_config on Gemini 3 models.
                if "gemini-3" in GEMINI_MODEL.lower():
                    try:
                        generation_config["thinking_config"] = types.ThinkingConfig(
                            thinking_level=types.ThinkingLevel.MINIMAL
                        )
                    except Exception:
                        # Older SDK fallback: ignore thinking_config
                        pass

                requests.append({
                    "key": key,
                    "request": {
                        "contents": [{
                            "role": "user",
                            "parts": [{"text": user_prompt}],
                        }],
                        "generation_config": generation_config,
                        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    },
                })

                manifest_rows.append({"key": key, **base_manifest_row(row, prompt_name, run_idx, author_col, race_col)})

    return requests, pd.DataFrame(manifest_rows)


def gemini_upload_batch_input(client, jsonl_path: Path):
    from google.genai import types

    return client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(display_name=jsonl_path.name, mime_type="jsonl"),
    )


def gemini_submit_batch(client, uploaded_file_name: str, display_name: str):
    return client.batches.create(
        model=GEMINI_MODEL,
        src=uploaded_file_name,
        config={"display_name": display_name},
    )


def gemini_is_resource_exhausted_error(exc: Exception) -> bool:
    msg = str(exc)
    return ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg)


def gemini_submit_batch_with_retries(
    client, uploaded_file_name: str, display_name: str,
    max_attempts: int = 5, retry_delays: Optional[List[int]] = None,
):
    delays = retry_delays or [30, 60, 120, 240, 480]
    attempt = 1
    while True:
        try:
            return gemini_submit_batch(client, uploaded_file_name, display_name=display_name)
        except Exception as exc:
            if not gemini_is_resource_exhausted_error(exc):
                raise
            if attempt >= max_attempts:
                flush_print(f"Batch submission failed after {attempt} attempts with RESOURCE_EXHAUSTED. Raising error.")
                raise
            delay = delays[min(attempt - 1, len(delays) - 1)]
            flush_print(
                f"Batch submission hit RESOURCE_EXHAUSTED (attempt {attempt}/{max_attempts}); retrying in {delay}s..."
            )
            time.sleep(delay)
            attempt += 1


def gemini_poll_batch(client, job_name: str):
    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        flush_print(f"[{datetime.now().strftime('%H:%M:%S')}] state={state}")
        if state in terminal:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)


def gemini_extract_text_from_response(response_obj) -> str:
    try:
        if hasattr(response_obj, "text") and response_obj.text:
            return response_obj.text.strip()
    except Exception:
        pass
    try:
        candidates = getattr(response_obj, "candidates", None) or []
        parts = candidates[0].content.parts
        texts = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
        return "".join(texts).strip()
    except Exception:
        return ""


def gemini_fetch_results(client, job):
    rows = []

    try:
        inline_responses = job.dest.inlined_responses
    except Exception:
        inline_responses = None

    if inline_responses:
        for item in inline_responses:
            key = getattr(item, "key", "")
            text = ""
            error = ""
            if getattr(item, "response", None):
                text = gemini_extract_text_from_response(item.response)
            else:
                error = str(getattr(item, "status", "unknown_error"))
            rows.append({"key": key, "llm_output": text, "error": error})
        return pd.DataFrame(rows)

    try:
        output_file_name = job.dest.file_name
    except Exception:
        output_file_name = None

    if not output_file_name:
        raise RuntimeError("Batch finished but no inline responses or output file was found.")

    out_file = client.files.download(file=output_file_name)
    raw = out_file.read() if hasattr(out_file, "read") else out_file
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = obj.get("key", "")
        response = obj.get("response")
        status = obj.get("status")
        text = ""
        error = ""
        if response:
            try:
                candidates = response.get("candidates", [])
                parts = candidates[0]["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts).strip()
            except Exception:
                text = ""
        elif status:
            error = json.dumps(status, ensure_ascii=False)
        rows.append({"key": key, "llm_output": text, "error": error})

    return pd.DataFrame(rows)


def gemini_merge_manifest_with_outputs(manifest_df, outputs_df):
    merged = manifest_df.merge(outputs_df, on="key", how="left")
    merged["llm_output"] = merged["llm_output"].fillna("")
    merged["error"] = merged["error"].fillna("")
    return merged


def gemini_chunk_records(records: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [records[i:i + batch_size] for i in range(0, len(records), batch_size)]


def run_gemini(csv_path, runs, results_root, smoke_rows=None, reuse_results_csv=None, batch_size=1200):
    dataset_stem = Path(csv_path).stem
    out_dir = results_root / dataset_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    raw_outputs_path = out_dir / "raw_outputs.csv"
    chunk_meta_path = out_dir / "batch_chunks_metadata.csv"

    df = load_dataset(csv_path, smoke_rows=smoke_rows)
    requests, manifest_df = gemini_build_requests_and_manifest(df, PROMPTS, runs)
    pending_manifest, reused_rows = split_manifest_by_reusable_results(
        manifest_df, reuse_results_csv, id_col="key", value_cols=["key", "llm_output", "error"],
    )

    pending_keys = set(pending_manifest["key"])
    pending_requests = [r for r in requests if r["key"] in pending_keys]

    manifest_df.to_csv(manifest_path, index=False)

    flush_print(f"Prepared {len(manifest_df)} total requests")
    flush_print(f"Reusing {len(reused_rows)} prior results")
    flush_print(f"Total pending requests: {len(pending_requests)}")

    if pending_requests:
        client = gemini_get_client()
        chunks = gemini_chunk_records(pending_requests, batch_size)
        total_chunks = len(chunks)
        flush_print(f"Batch size: {batch_size}")
        flush_print(f"Number of chunks: {total_chunks}")

        chunk_frames: List[pd.DataFrame] = []
        chunk_meta_rows: List[Dict[str, Any]] = []

        for chunk_idx, chunk_requests in enumerate(chunks, start=1):
            chunk_keys = [r["key"] for r in chunk_requests]
            chunk_manifest = pending_manifest[pending_manifest["key"].isin(set(chunk_keys))].copy()
            input_jsonl_path = out_dir / f"batch_input_part_{chunk_idx:04d}.jsonl"
            output_csv_path = out_dir / f"batch_output_part_{chunk_idx:04d}.csv"

            flush_print(f"Current chunk {chunk_idx} / {total_chunks}")
            flush_print(f"Requests in chunk: {len(chunk_requests)}")

            write_jsonl(chunk_requests, input_jsonl_path)
            flush_print(f"Uploading input file: {input_jsonl_path}")
            uploaded = gemini_upload_batch_input(client, input_jsonl_path)
            flush_print(f"Uploaded file: {uploaded.name}")

            display_name = (
                f"{dataset_stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
                f"part-{chunk_idx:04d}-of-{total_chunks:04d}"
            )
            job = gemini_submit_batch_with_retries(
                client, uploaded.name, display_name=display_name,
                max_attempts=5, retry_delays=[30, 60, 120, 240, 480],
            )
            flush_print(f"Job name: {job.name}")

            final_job = gemini_poll_batch(client, job.name)
            final_state = final_job.state.name
            flush_print(f"Final chunk state: {final_state}")

            chunk_meta = {
                "chunk_index": chunk_idx,
                "num_requests": len(chunk_requests),
                "job_name": job.name,
                "final_state": final_state,
                "input_jsonl_path": str(input_jsonl_path),
                "output_file_path": "",
            }

            if final_state != "JOB_STATE_SUCCEEDED":
                chunk_meta_rows.append(chunk_meta)
                pd.DataFrame(chunk_meta_rows).to_csv(chunk_meta_path, index=False)
                raise RuntimeError(f"Chunk {chunk_idx} ended with non-success state: {final_state}")

            outputs_df = gemini_fetch_results(client, final_job)
            outputs_df.to_csv(output_csv_path, index=False)
            chunk_meta["output_file_path"] = str(output_csv_path)
            chunk_meta_rows.append(chunk_meta)
            pd.DataFrame(chunk_meta_rows).to_csv(chunk_meta_path, index=False)

            chunk_frames.append(gemini_merge_manifest_with_outputs(chunk_manifest, outputs_df))

        pending_final = pd.concat(chunk_frames, ignore_index=True, sort=False) if chunk_frames else pd.DataFrame(
            columns=list(manifest_df.columns) + ["llm_output", "error"]
        )
    else:
        pending_final = pd.DataFrame(columns=list(manifest_df.columns) + ["llm_output", "error"])

    merged = pd.concat([reused_rows, pending_final], ignore_index=True, sort=False)
    merged["llm_output"] = merged["llm_output"].fillna("")
    merged["error"] = merged["error"].fillna("")
    if "quote_id" in merged.columns and "prompt_type" in merged.columns and "run" in merged.columns:
        merged = merged.sort_values(by=["quote_id", "prompt_type", "run"], kind="mergesort").reset_index(drop=True)
    merged.to_csv(raw_outputs_path, index=False)

    flush_print(f"Saved -> {raw_outputs_path}")


# ============================================================================
# OpenAI (GPT-5) — resumable /v1/responses batch with auto-split over 20k requests
# ============================================================================

OPENAI_MODEL = "gpt-5.1"
OPENAI_RESULTS_ROOT = RESULTS_ROOT / "gpt5_batch"
OPENAI_MAX_REQUESTS_PER_BATCH = 20000
OPENAI_MAX_FINALIZING_SECONDS = int(os.getenv("BATCH_MAX_FINALIZING_SECONDS", "7200"))


def openai_get_client():
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    return OpenAI()


def openai_extract_output_text_from_batch_response(body: dict) -> str:
    """Batch results for /v1/responses mirror Responses API payloads."""
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


def openai_build_requests_and_manifest(df, prompts, runs, model, run_start: int = 0):
    author_col = df.attrs["author_col"]
    race_col = df.attrs["race_col"]

    requests = []
    manifest_rows = []

    for _, row in df.iterrows():
        for prompt_name, template in prompts.items():
            for run_idx in range(runs):
                run_id = run_start + run_idx
                custom_id = f"qid={row['quote_id']}|prompt={prompt_name}|run={run_id}|author={row[author_col]}"
                user_prompt = template.format(passage=row["quote"])

                requests.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        "input": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "text": {"verbosity": "low"},
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "reasoning": {"effort": "none"},
                        "max_output_tokens": MAX_OUTPUT_TOKENS,
                    },
                })

                manifest_rows.append({"custom_id": custom_id, **base_manifest_row(row, prompt_name, run_id, author_col, race_col)})

    return requests, pd.DataFrame(manifest_rows)


def openai_upload_batch_input(client, jsonl_path: Path):
    with open(jsonl_path, "rb") as f:
        return client.files.create(file=f, purpose="batch")


def openai_create_batch_job(client, input_file_id: str, metadata: Optional[dict] = None):
    return client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata or {},
    )


def openai_wait_for_batch_completion(
    client, batch_id: str, batch_info_path: Path, batch_info: dict,
    poll_interval_seconds: int = 30, max_finalizing_seconds: int = OPENAI_MAX_FINALIZING_SECONDS,
):
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    flush_print(f"Polling every {poll_interval_seconds} seconds...")
    finalizing_started_at = None

    while True:
        batch = client.batches.retrieve(batch_id)
        now_iso = datetime.now().isoformat()
        batch_info["status"] = getattr(batch, "status", None)
        batch_info["updated_at"] = now_iso
        batch_info["output_file_id"] = getattr(batch, "output_file_id", None)
        batch_info["error_file_id"] = getattr(batch, "error_file_id", None)
        save_json(batch_info_path, batch_info)
        rc = getattr(batch, "request_counts", None)
        if rc is not None:
            total = getattr(rc, "total", None)
            completed = getattr(rc, "completed", None)
            failed = getattr(rc, "failed", None)
            flush_print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"status={batch.status} completed={completed} failed={failed} total={total}"
            )
        else:
            flush_print(f"[{datetime.now().strftime('%H:%M:%S')}] status={batch.status}")

        if batch.status in terminal_statuses:
            return batch

        if batch.status == "finalizing":
            if finalizing_started_at is None:
                finalizing_started_at = time.time()
            elapsed = time.time() - finalizing_started_at
            if elapsed >= max_finalizing_seconds:
                total = getattr(rc, "total", 0) if rc is not None else 0
                completed = getattr(rc, "completed", 0) if rc is not None else 0
                failed = getattr(rc, "failed", 0) if rc is not None else 0
                done = (completed or 0) + (failed or 0)
                if total and done >= total:
                    flush_print("Finalizing exceeded threshold with all requests done; proceeding to output-file wait.")
                    return batch
        else:
            finalizing_started_at = None

        time.sleep(poll_interval_seconds)


def openai_wait_for_output_file_id(client, batch_id: str, max_wait_seconds: int = 180, poll_interval_seconds: int = 10):
    started = time.time()
    last_batch = None
    while time.time() - started < max_wait_seconds:
        batch = client.batches.retrieve(batch_id)
        last_batch = batch
        if getattr(batch, "output_file_id", None):
            return batch
        time.sleep(poll_interval_seconds)
    return last_batch


def openai_download_file_content(client, file_id: str, out_path: Path):
    content = client.files.content(file_id)
    data = content.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
    with open(out_path, mode) as f:
        f.write(data)
    return out_path


def openai_parse_batch_output_jsonl(output_jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            response_obj = obj.get("response") if isinstance(obj.get("response"), dict) else {}
            error_obj = obj.get("error")
            status_code = response_obj.get("status_code")
            body = response_obj.get("body")
            if not isinstance(body, dict):
                body = {}
            try:
                llm_output = openai_extract_output_text_from_batch_response(body)
            except Exception as exc:
                llm_output = ""
                error_obj = {"parse_error": str(exc), "body_type": str(type(body))}
            error_text = ""
            if error_obj:
                try:
                    error_text = json.dumps(error_obj, ensure_ascii=False)
                except Exception:
                    error_text = str(error_obj)
            elif status_code:
                try:
                    if int(status_code) >= 400:
                        error_text = json.dumps(body, ensure_ascii=False)
                except Exception:
                    error_text = f"invalid_status_code:{status_code}"
            rows.append({
                "custom_id": custom_id,
                "llm_output": llm_output,
                "error": error_text,
                "http_status": status_code,
                "raw_response_body": json.dumps(body, ensure_ascii=False),
                "line_num": line_num,
            })
    return pd.DataFrame(rows)


def openai_merge_manifest_with_outputs(manifest_df, outputs_df):
    merged = manifest_df.merge(outputs_df, on="custom_id", how="left")
    if merged["llm_output"].isna().any():
        missing = int(merged["llm_output"].isna().sum())
        flush_print(f"Warning: {missing} rows had no batch result.")
    merged["llm_output"] = merged["llm_output"].fillna("")
    merged["error"] = merged["error"].fillna("")
    merged["http_status"] = merged["http_status"].fillna("")
    merged["raw_response_body"] = merged["raw_response_body"].fillna("")
    return merged


def openai_run_dataset_via_batch(
    csv_path, out_root: Path, model: str = OPENAI_MODEL,
    smoke_rows=None, runs: int = 1, prompts: dict = PROMPTS, run_start: int = 0,
):
    client = openai_get_client()

    dataset_stem = Path(csv_path).stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_prefix = f"prompt_attribution_{dataset_stem}_"
    existing_dirs = sorted(
        [p for p in out_root.glob(f"{out_prefix}*") if p.is_dir()],
        key=lambda p: p.name, reverse=True,
    )
    out_dir = None
    for candidate in existing_dirs:
        if (candidate / "raw_outputs.csv").exists():
            flush_print(f"Results already processed. Skipping. ({candidate})")
            return candidate
        if (
            (candidate / "batch_info.json").exists()
            or (candidate / "batch_output.jsonl").exists()
            or (candidate / "manifest.csv").exists()
            or (candidate / "batch_input.jsonl").exists()
        ):
            out_dir = candidate
            flush_print(f"Resuming existing output directory: {out_dir}")
            break

    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / f"{out_prefix}{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

    batch_input_path = out_dir / "batch_input.jsonl"
    manifest_path = out_dir / "manifest.csv"
    batch_info_path = out_dir / "batch_info.json"
    output_jsonl_path = out_dir / "batch_output.jsonl"
    raw_outputs_path = out_dir / "raw_outputs.csv"

    if raw_outputs_path.exists():
        flush_print("Results already processed. Skipping.")
        return out_dir

    flush_print(f"Loading dataset: {csv_path}")
    df = load_dataset(csv_path, smoke_rows=smoke_rows)
    flush_print(f"Loaded {len(df)} quotes")

    requests, manifest_df = openai_build_requests_and_manifest(
        df, prompts=prompts, runs=runs, model=model, run_start=run_start,
    )
    flush_print(f"Prepared {len(requests)} batch requests")

    if not batch_input_path.exists():
        write_jsonl(requests, batch_input_path)
        flush_print(f"Wrote batch input: {batch_input_path}")
    if not manifest_path.exists():
        manifest_df.to_csv(manifest_path, index=False)
        flush_print(f"Wrote manifest:    {manifest_path}")

    existing_batch_info = load_json(batch_info_path) or {}
    existing_batch_id = existing_batch_info.get("batch_id")

    final_batch = None
    input_file_id = existing_batch_info.get("input_file_id")

    if existing_batch_id:
        flush_print("Resuming existing batch job...")
        batch = client.batches.retrieve(existing_batch_id)
        existing_batch_info["status"] = getattr(batch, "status", None)
        existing_batch_info["updated_at"] = datetime.now().isoformat()
        existing_batch_info["output_file_id"] = getattr(batch, "output_file_id", None)
        existing_batch_info["error_file_id"] = getattr(batch, "error_file_id", None)
        save_json(batch_info_path, existing_batch_info)
        status = batch.status
        if status == "completed":
            final_batch = batch
        elif status in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"Existing batch is not resumable. Final status: {status}")
        else:
            final_batch = openai_wait_for_batch_completion(
                client, existing_batch_id, batch_info_path=batch_info_path,
                batch_info=existing_batch_info, poll_interval_seconds=POLL_INTERVAL_SECONDS,
            )
    else:
        flush_print("Submitting batch job...")
        uploaded = openai_upload_batch_input(client, batch_input_path)
        input_file_id = uploaded.id
        flush_print(f"Uploaded input file: {input_file_id}")

        batch = openai_create_batch_job(
            client, input_file_id=input_file_id,
            metadata={"dataset": dataset_stem, "model": model, "kind": "prompt_attribution"},
        )
        flush_print(f"Batch ID: {batch.id}")
        save_json(batch_info_path, {
            "batch_id": batch.id,
            "input_file_id": input_file_id,
            "status": "submitted",
            "dataset": dataset_stem,
            "model": model,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
        })

        final_batch = openai_wait_for_batch_completion(
            client, batch.id, batch_info_path=batch_info_path,
            batch_info=load_json(batch_info_path) or {}, poll_interval_seconds=POLL_INTERVAL_SECONDS,
        )

    flush_print(f"Final batch status: {final_batch.status}")

    batch_info = {
        "batch_id": final_batch.id,
        "status": final_batch.status,
        "input_file_id": getattr(final_batch, "input_file_id", None),
        "output_file_id": getattr(final_batch, "output_file_id", None),
        "error_file_id": getattr(final_batch, "error_file_id", None),
        "endpoint": getattr(final_batch, "endpoint", None),
        "completion_window": getattr(final_batch, "completion_window", None),
        "created_at": getattr(final_batch, "created_at", None),
        "completed_at": getattr(final_batch, "completed_at", None),
        "dataset": dataset_stem,
        "model": model,
        "updated_at": datetime.now().isoformat(),
    }
    save_json(batch_info_path, batch_info)

    if final_batch.status not in {"completed", "finalizing"}:
        raise RuntimeError(f"Batch did not complete successfully. Final status: {final_batch.status}")

    output_file_id = getattr(final_batch, "output_file_id", None) or (load_json(batch_info_path) or {}).get("output_file_id")
    if not output_file_id:
        flush_print("Batch is completed but output_file_id is not ready yet; waiting...")
        refreshed = openai_wait_for_output_file_id(client, final_batch.id)
        if refreshed is not None:
            final_batch = refreshed
            batch_info["output_file_id"] = getattr(final_batch, "output_file_id", None)
            batch_info["error_file_id"] = getattr(final_batch, "error_file_id", None)
            batch_info["updated_at"] = datetime.now().isoformat()
            save_json(batch_info_path, batch_info)
        output_file_id = getattr(final_batch, "output_file_id", None) or (load_json(batch_info_path) or {}).get("output_file_id")

    if not output_file_id:
        raise RuntimeError("Batch completed but no output_file_id was returned.")

    if output_jsonl_path.exists():
        flush_print(f"Found existing batch output. Skipping download: {output_jsonl_path}")
    else:
        flush_print("Downloading results...")
        openai_download_file_content(client, output_file_id, output_jsonl_path)
        flush_print(f"Downloaded batch output: {output_jsonl_path}")

    flush_print("Merging outputs...")
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
    outputs_df = openai_parse_batch_output_jsonl(output_jsonl_path)
    raw_outputs_df = openai_merge_manifest_with_outputs(manifest_df, outputs_df)

    flush_print("Writing raw_outputs.csv")
    raw_outputs_df.to_csv(raw_outputs_path, index=False)

    save_json(out_dir / "config.json", {
        "timestamp": datetime.now().isoformat(),
        "csv_path": csv_path,
        "dataset_stem": dataset_stem,
        "smoke_rows": smoke_rows,
        "runs": runs,
        "run_start": run_start,
        "model": model,
        "total_quotes": len(df),
        "total_outputs_requested": len(requests),
        "total_outputs_received": len(outputs_df),
        "total_errors": int(raw_outputs_df["error"].astype(str).str.strip().ne("").sum()),
        "batch_id": final_batch.id,
        "input_file_id": input_file_id,
        "output_file_id": output_file_id,
    })

    flush_print(f"Done. Results saved to: {raw_outputs_path}")
    return out_dir


def openai_run_dataset_via_batch_auto_split(
    csv_path, out_root: Path, model: str = OPENAI_MODEL,
    smoke_rows=None, runs: int = 1, prompts: dict = PROMPTS,
    max_requests_per_batch: int = OPENAI_MAX_REQUESTS_PER_BATCH,
):
    """Split large jobs into multiple batch submissions under max_requests_per_batch,
    across runs and (when needed) row shards, then merge into one raw_outputs.csv."""
    df_full = load_dataset(csv_path, smoke_rows=smoke_rows)
    total_rows = len(df_full)
    prompt_count = len(prompts)
    per_run_requests = total_rows * prompt_count

    if prompt_count <= 0:
        raise ValueError("No prompts configured.")
    if total_rows <= 0:
        raise ValueError("No requests to submit.")
    if max_requests_per_batch < prompt_count:
        raise RuntimeError(
            f"max_requests_per_batch={max_requests_per_batch} is too small for {prompt_count} prompts per row."
        )

    max_rows_per_batch_for_one_run = max_requests_per_batch // prompt_count
    rows_per_shard = min(total_rows, max_rows_per_batch_for_one_run)
    if rows_per_shard < 1:
        raise RuntimeError(
            f"Unable to fit one row in a batch under max_requests_per_batch={max_requests_per_batch}."
        )

    row_shard_specs = []
    row_start = 0
    while row_start < total_rows:
        row_count = min(rows_per_shard, total_rows - row_start)
        row_end = row_start + row_count - 1
        row_shard_specs.append((row_start, row_count, row_end))
        row_start += row_count

    if len(row_shard_specs) == 1:
        max_runs_per_batch = max_requests_per_batch // per_run_requests
    else:
        max_runs_per_batch = max_requests_per_batch // (rows_per_shard * prompt_count)
    max_runs_per_batch = max(1, max_runs_per_batch)

    run_chunk_specs = []
    start_run = 0
    while start_run < runs:
        chunk_runs = min(max_runs_per_batch, runs - start_run)
        end_run = start_run + chunk_runs - 1
        run_chunk_specs.append((start_run, chunk_runs, end_run))
        start_run += chunk_runs

    if len(row_shard_specs) == 1 and len(run_chunk_specs) == 1:
        return openai_run_dataset_via_batch(
            csv_path=csv_path, out_root=out_root, model=model,
            smoke_rows=smoke_rows, runs=runs, prompts=prompts, run_start=0,
        )

    dataset_stem = Path(csv_path).stem
    split_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_root = out_root / f"split_{dataset_stem}_{split_stamp}"
    chunk_root = split_root / "chunks"
    shard_root = split_root / "row_shards"
    chunk_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    flush_print(
        f"Auto-splitting submission: total_runs={runs}, total_rows={total_rows}, "
        f"per_run_requests={per_run_requests}, row_shards={len(row_shard_specs)}, "
        f"max_runs_per_batch={max_runs_per_batch}, run_chunks={len(run_chunk_specs)}"
    )

    shard_specs_with_paths = []
    if len(row_shard_specs) == 1:
        row_start, row_count, row_end = row_shard_specs[0]
        shard_specs_with_paths.append({
            "row_start": row_start, "row_count": row_count, "row_end": row_end,
            "csv_path": csv_path, "used_smoke": smoke_rows is not None,
        })
    else:
        for shard_idx, (row_start, row_count, row_end) in enumerate(row_shard_specs):
            shard_df = df_full.iloc[row_start:row_start + row_count].copy()
            shard_path = shard_root / f"rows_{row_start}_to_{row_end}.csv"
            shard_df.to_csv(shard_path, index=False)
            shard_specs_with_paths.append({
                "shard_idx": shard_idx, "row_start": row_start, "row_count": row_count,
                "row_end": row_end, "csv_path": str(shard_path), "used_smoke": False,
            })

    chunk_dirs = []
    chunk_frames = []
    for shard_spec in shard_specs_with_paths:
        row_start = shard_spec["row_start"]
        row_end = shard_spec["row_end"]
        shard_csv_path = shard_spec["csv_path"]
        shard_smoke_rows = smoke_rows if shard_spec.get("used_smoke") else None
        flush_print(f"Submitting row shard rows {row_start}..{row_end}")

        for start, chunk_runs, end in run_chunk_specs:
            this_out_root = chunk_root / f"rows_{row_start}_to_{row_end}" / f"runs_{start}_to_{end}"
            flush_print(f"Submitting run chunk {start}..{end} for rows {row_start}..{row_end}")
            chunk_dir = openai_run_dataset_via_batch(
                csv_path=shard_csv_path, out_root=this_out_root, model=model,
                smoke_rows=shard_smoke_rows, runs=chunk_runs, prompts=prompts, run_start=start,
            )
            chunk_dirs.append(str(chunk_dir))
            chunk_frames.append(pd.read_csv(Path(chunk_dir) / "raw_outputs.csv"))

    combined_df = pd.concat(chunk_frames, ignore_index=True)
    combined_df = combined_df.sort_values(
        by=["quote_id", "prompt_type", "run", "author"], kind="mergesort",
    ).reset_index(drop=True)

    combined_dir = split_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_raw_path = combined_dir / "raw_outputs.csv"
    combined_df.to_csv(combined_raw_path, index=False)

    save_json(combined_dir / "config.json", {
        "timestamp": datetime.now().isoformat(),
        "csv_path": csv_path,
        "dataset_stem": dataset_stem,
        "smoke_rows": smoke_rows,
        "runs": runs,
        "model": model,
        "total_rows": total_rows,
        "prompt_count": prompt_count,
        "max_requests_per_batch": max_requests_per_batch,
        "per_run_requests": per_run_requests,
        "rows_per_shard": rows_per_shard,
        "max_runs_per_batch": max_runs_per_batch,
        "row_shard_count": len(row_shard_specs),
        "run_chunk_count": len(run_chunk_specs),
        "chunk_count": len(chunk_dirs),
        "row_shards": shard_specs_with_paths,
        "chunk_dirs": chunk_dirs,
        "total_outputs": len(combined_df),
        "total_errors": int(combined_df["error"].fillna("").astype(str).str.strip().ne("").sum()),
    })

    flush_print(f"Combined outputs written: {combined_raw_path}")
    return combined_dir


def run_openai(csv_path, runs, results_root, smoke_rows=None):
    return openai_run_dataset_via_batch_auto_split(
        csv_path=csv_path, out_root=results_root, model=OPENAI_MODEL,
        smoke_rows=smoke_rows, runs=runs, prompts=PROMPTS,
    )


# ============================================================================
# Dispatcher
# ============================================================================

PROVIDER_RESULTS_ROOTS = {
    "openai": OPENAI_RESULTS_ROOT,
    "claude": CLAUDE_RESULTS_ROOT,
    "gemini": GEMINI_RESULTS_ROOT,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a zero-shot attribution batch for one provider and dataset."
    )
    parser.add_argument("--provider", required=True, choices=["openai", "claude", "gemini"],
                        help="Which provider's batch API to run.")
    parser.add_argument("--csv-path", required=True, help="Path to dataset CSV.")
    parser.add_argument("--runs", type=int, required=True, help="Number of runs per prompt.")
    parser.add_argument("--batch-size", type=int, default=1200,
                        help="Requests per Gemini batch job (gemini only).")
    parser.add_argument("--smoke-rows", type=int, default=0,
                        help="Optional dry-run row count (head). Use 0 to disable.")
    parser.add_argument("--run-label", default="",
                        help="Optional run label; outputs go under <results_root>/<run_label>/.")
    parser.add_argument("--reuse-results-csv", default="",
                        help="Optional prior raw_outputs.csv to reuse (claude/gemini only).")
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        raise SystemExit(f"Dataset not found: {csv_path}")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.smoke_rows < 0:
        raise SystemExit("--smoke-rows must be >= 0")

    results_root = PROVIDER_RESULTS_ROOTS[args.provider]
    run_root = results_root / args.run_label if args.run_label else results_root
    smoke_rows = args.smoke_rows if args.smoke_rows > 0 else None
    reuse_results_csv = args.reuse_results_csv.strip() or None

    flush_print(f"Provider: {args.provider}")
    flush_print(f"Running: {csv_path}")
    flush_print(f"Runs: {args.runs}")
    flush_print(f"Smoke rows: {smoke_rows}")
    flush_print(f"Output root: {run_root}")

    if args.provider == "claude":
        run_claude(csv_path, args.runs, run_root, smoke_rows=smoke_rows, reuse_results_csv=reuse_results_csv)
    elif args.provider == "gemini":
        flush_print(f"Batch size: {args.batch_size}")
        run_gemini(csv_path, args.runs, run_root, smoke_rows=smoke_rows,
                   reuse_results_csv=reuse_results_csv, batch_size=args.batch_size)
    elif args.provider == "openai":
        if reuse_results_csv:
            flush_print("Note: --reuse-results-csv is ignored for openai (it resumes via batch_info.json instead).")
        run_openai(csv_path, args.runs, run_root, smoke_rows=smoke_rows)

    flush_print("\nDone.")


if __name__ == "__main__":
    main()
