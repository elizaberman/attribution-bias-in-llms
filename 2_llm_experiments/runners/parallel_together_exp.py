#!/usr/bin/env python3
"""Zero-shot (no-evidence) attribution runner for open-weight models on Together AI.

Sends each quote under all three prompts — direct, indirect, and indirect_overt
(see PROMPTS below) — to the Together chat-completions API and records whether the
model names the true author. This is the no-evidence arm of the paper's evaluation;
the evidence-conditioned (RAG) counterpart is `rag_together_exp.py`.

Unlike the batch-API runners, this calls the live endpoint from a thread pool with
client-side rate limiting (--target-rpm, --max-workers) and retry/backoff, so
results stream in rather than arriving hours later.

Scoring is done in-process: `build_aliases` expands each author into name variants
and `contains_alias` substring-matches them against the response, producing the
`direct_hit_author` / `indirect_hit_author` columns.

Usage
-----
  python parallel_together_exp.py \
      --data-file 1_dataset_construction/datasets/intersectional_with_quotes.csv \
      --output "$ATTRIBENCH_RESULTS/kimi_intersectional.csv" \
      --model moonshotai/Kimi-K2.5 --runs 3 --n-rows 7964

NOTE: --n-rows defaults to 25 and is applied as a plain slice (rows[:n_rows]), so it
must be >= the dataset size or the run silently covers only the first 25 quotes.
`run_together_models_full.sh` counts the rows and passes this automatically.

Output: one wide-format CSV row per quote x run, written incrementally and
resumable — an existing --output file is re-read and completed work is skipped.
"""
import argparse
import csv
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter


TOGETHER_MODELS = [
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3.1",
    "moonshotai/Kimi-K2.5",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3.5-397B-A17B",
    "zai-org/GLM-5",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]
DEFAULT_MODEL = TOGETHER_MODELS[0]
DEFAULT_N_ROWS = 25
DEFAULT_RUNS = 3

DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_REQUEST_TIMEOUT_S = 120
DEFAULT_MAX_RETRIES = 6
DEFAULT_PROGRESS_EVERY = 10
DEFAULT_MAX_WORKERS = 16
DEFAULT_TARGET_RPM = 3600
DEFAULT_FLUSH_EVERY = 50
DEFAULT_NO_PROGRESS_TIMEOUT_MIN = 0.0

SYSTEM_PROMPT = "You are a helpful assistant. Respond with at most one sentence."

PROMPTS = {
    "direct": "Please identify the author of the following quote:\n\"{passage}\"",
    "indirect": "Briefly summarize the context in which the following quote was written:\n\"{passage}\"",
    "indirect_overt": (
        "Briefly summarize the context in which the following quote was written, "
        "mentioning the author if relevant:\n\"{passage}\""
    ),
}

THREAD_LOCAL = threading.local()


def flush_print(*args, **kwargs) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


def build_aliases(author: str, alt: Optional[str] = None) -> List[str]:
    aliases: List[str] = []
    if isinstance(author, str) and author.strip():
        author = author.strip()
        aliases.append(author)
        parts = author.split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            aliases.extend([last, f"{first[0]} {last}", f"{first[0]}. {last}"])
    if isinstance(alt, str) and alt.strip():
        aliases.append(alt.strip())
    return list(dict.fromkeys([a.strip() for a in aliases if isinstance(a, str) and a.strip()]))


def contains_alias(text: str, aliases: List[str]) -> bool:
    if not text:
        return False
    lower_text = text.lower()
    return any(alias.lower() in lower_text for alias in aliases)


def build_client() -> str:
    api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY")
    if not api_key:
        raise SystemExit("TOGETHER_API_KEY (or TOGETHERAI_API_KEY) not set")
    return api_key


class RateLimiter:
    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self.max_per_window = rpm
        self.window_seconds = 60.0
        self.events: Deque[float] = deque()
        self.lock = threading.Lock()

    def wait_for_slot(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0] >= self.window_seconds:
                    self.events.popleft()
                if len(self.events) < self.max_per_window:
                    self.events.append(now)
                    return
                sleep_s = self.window_seconds - (now - self.events[0])
            time.sleep(max(sleep_s, 0.01))


def get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        THREAD_LOCAL.session = session
    return session


def describe_request_error(err: Exception) -> str:
    if isinstance(err, requests.HTTPError):
        resp = err.response
        if resp is not None:
            body = (resp.text or "").strip().replace("\n", " ")
            if len(body) > 500:
                body = body[:500] + "...(truncated)"
            return f"HTTPError status={resp.status_code} body={body!r}"
        return f"HTTPError {err}"
    if isinstance(err, requests.Timeout):
        return f"Timeout {err}"
    if isinstance(err, requests.ConnectionError):
        return f"ConnectionError {err}"
    return f"{type(err).__name__}: {err}"


def chat_completion(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: int,
    max_retries: int,
    rate_limiter: RateLimiter,
) -> Dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    # gpt-oss uses reasoning_effort instead of the reasoning enabled toggle.
    if model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"
    else:
        payload["reasoning"] = {"enabled": False}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err: Optional[Exception] = None
    session = get_session()

    for attempt in range(max_retries + 1):
        try:
            rate_limiter.wait_for_slot()
            resp = session.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout_s,
            )
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"retryable status={resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            err_desc = describe_request_error(e)
            if attempt >= max_retries:
                break
            sleep_s = min(30, (2 ** attempt) + 0.1 * (attempt + 1))
            flush_print(
                f"[retry] attempt {attempt + 1}/{max_retries} failed ({err_desc}); "
                f"sleeping {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"chat_completion failed after {max_retries + 1} attempts: "
        f"{describe_request_error(last_err) if last_err else 'unknown error'}"
    )


def extract_text_from_response(resp: Dict) -> str:
    try:
        txt = str(resp["choices"][0]["message"]["content"]).strip()
        return txt if txt else ""
    except Exception:
        return ""


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_has_all_prompt_responses(row: Dict[str, str]) -> bool:
    return all(
        str(row.get(col, "")).strip()
        for col in ("direct_response", "indirect_response", "indirect_overt_response")
    )


def read_completed_units(path: Path, require_all_prompts_complete: bool = False) -> Dict[str, set]:
    completed = set()
    fieldnames: List[str] = []
    if not path.exists():
        return {"completed": completed, "fieldnames": fieldnames}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            quote_id = str(row.get("quote_id", "")).strip()
            run = str(row.get("run", "")).strip()
            if not quote_id or not run:
                continue
            if require_all_prompts_complete and not row_has_all_prompt_responses(row):
                continue
            if quote_id and run:
                completed.add((quote_id, run))
    return {"completed": completed, "fieldnames": fieldnames}


def process_unit(
    unit: Tuple[str, int, str, str, Optional[str]],
    args: argparse.Namespace,
    api_key: str,
    rate_limiter: RateLimiter,
) -> Dict[str, object]:
    quote_id, run_idx, quote, author, alt_author = unit
    aliases = build_aliases(author, alt_author if isinstance(alt_author, str) else None)

    direct_prompt = PROMPTS["direct"].format(passage=quote)
    indirect_prompt = PROMPTS["indirect"].format(passage=quote)
    indirect_overt_prompt = PROMPTS["indirect_overt"].format(passage=quote)

    direct_resp = chat_completion(
        api_key=api_key,
        model=args.model,
        prompt=direct_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.request_timeout_s,
        max_retries=args.max_retries,
        rate_limiter=rate_limiter,
    )
    direct_text = extract_text_from_response(direct_resp)
    direct_hit = contains_alias(direct_text, aliases)

    indirect_resp = chat_completion(
        api_key=api_key,
        model=args.model,
        prompt=indirect_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.request_timeout_s,
        max_retries=args.max_retries,
        rate_limiter=rate_limiter,
    )
    indirect_text = extract_text_from_response(indirect_resp)
    indirect_hit = contains_alias(indirect_text, aliases)

    indirect_overt_resp = chat_completion(
        api_key=api_key,
        model=args.model,
        prompt=indirect_overt_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.request_timeout_s,
        max_retries=args.max_retries,
        rate_limiter=rate_limiter,
    )
    indirect_overt_text = extract_text_from_response(indirect_overt_resp)
    indirect_overt_hit = contains_alias(indirect_overt_text, aliases)

    return {
        "quote_id": quote_id,
        "run": run_idx,
        "author_ground_truth": author,
        "direct_prompt": direct_prompt,
        "direct_response": direct_text,
        "direct_hit_author": direct_hit,
        "indirect_prompt": indirect_prompt,
        "indirect_response": indirect_text,
        "indirect_hit_author": indirect_hit,
        "indirect_overt_prompt": indirect_overt_prompt,
        "indirect_overt_response": indirect_overt_text,
        "indirect_overt_hit_author": indirect_overt_hit,
        "model": args.model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Together AI models on direct, indirect, and indirect-overt prompts."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=TOGETHER_MODELS)
    parser.add_argument("--quote-col", default="quote")
    parser.add_argument("--author-col", default="author_clean")
    parser.add_argument("--alt-author-col", default="author_alt_name")
    parser.add_argument("--quote-id-col", default="quote_id")
    parser.add_argument("--n-rows", type=int, default=DEFAULT_N_ROWS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--request-timeout-s", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--target-rpm",
        type=int,
        default=DEFAULT_TARGET_RPM,
        help="Global client-side request throttle. Keep below account max RPM for headroom.",
    )
    parser.add_argument("--flush-every", type=int, default=DEFAULT_FLUSH_EVERY)
    parser.add_argument(
        "--no-progress-timeout-min",
        type=float,
        default=DEFAULT_NO_PROGRESS_TIMEOUT_MIN,
        help=(
            "Abort if no unit completes for this many minutes. "
            "Set <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--require-all-prompts-complete",
        action="store_true",
        help=(
            "Treat existing rows as completed only when direct/indirect/indirect_overt "
            "responses are all non-empty. Useful for backfilling partial units."
        ),
    )
    args = parser.parse_args()

    rows = read_rows(args.data_file)[: args.n_rows]
    prompts_per_unit = len(PROMPTS)
    total_units = len(rows) * args.runs
    total_api_requests = total_units * prompts_per_unit

    flush_print(
        f"Using first {len(rows)} rows with {args.runs} run(s) per prompt set "
        "(direct, indirect, indirect_overt)"
    )
    flush_print(f"Processing {total_units} quote-run units...")
    flush_print(
        f"Planned API requests: {total_api_requests} "
        f"({len(rows)} quotes x {args.runs} runs x {prompts_per_unit} prompts)"
    )
    flush_print(
        f"max_workers={args.max_workers}, target_rpm={args.target_rpm}, flush_every={args.flush_every}"
    )

    api_key = build_client()
    rate_limiter = RateLimiter(args.target_rpm)

    fieldnames = [
        "quote_id",
        "run",
        "author_ground_truth",
        "direct_prompt",
        "direct_response",
        "direct_hit_author",
        "indirect_prompt",
        "indirect_response",
        "indirect_hit_author",
        "indirect_overt_prompt",
        "indirect_overt_response",
        "indirect_overt_hit_author",
        "model",
    ]

    existing_info = read_completed_units(
        args.output, require_all_prompts_complete=args.require_all_prompts_complete
    )
    completed_units_set = existing_info["completed"]
    flush_print(f"Resume check: found {len(completed_units_set)} completed quote-run units in {args.output}")

    pending_units: List[Tuple[str, int, str, str, Optional[str]]] = []
    skipped_units = 0
    for idx, row_dict in enumerate(rows):
        quote_id = str(row_dict.get(args.quote_id_col, idx))
        quote = str(row_dict.get(args.quote_col, ""))
        author = str(row_dict.get(args.author_col, ""))
        alt_author = row_dict.get(args.alt_author_col, None)
        for run_idx in range(args.runs):
            unit_key = (quote_id, str(run_idx))
            if unit_key in completed_units_set:
                skipped_units += 1
                continue
            pending_units.append((quote_id, run_idx, quote, author, alt_author))

    flush_print(f"Pending units: {len(pending_units)} | Skipped existing: {skipped_units}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.output.exists() else "w"
    run_start_time = time.time()
    completed_units = 0
    since_flush = 0

    with args.output.open(write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()
            f.flush()

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_unit, unit, args, api_key, rate_limiter): unit
                for unit in pending_units
            }
            failed_units = 0
            last_progress_time = time.time()

            while futures:
                done, _ = wait(futures, timeout=5.0, return_when=FIRST_COMPLETED)
                if not done:
                    if (
                        args.no_progress_timeout_min > 0
                        and (time.time() - last_progress_time) > (args.no_progress_timeout_min * 60.0)
                    ):
                        for fut in futures:
                            fut.cancel()
                        raise RuntimeError(
                            "No units completed for "
                            f"{args.no_progress_timeout_min:.1f} minutes; aborting to avoid a stall."
                        )
                    continue

                for fut in done:
                    futures.pop(fut, None)
                    try:
                        row_out = fut.result()
                    except Exception as e:
                        failed_units += 1
                        flush_print(f"[unit-failed] {type(e).__name__}: {e}")
                        continue

                    writer.writerow(row_out)
                    completed_units += 1
                    since_flush += 1
                    last_progress_time = time.time()
                    completed_units_set.add((str(row_out["quote_id"]), str(row_out["run"])))

                    if since_flush >= max(1, args.flush_every):
                        f.flush()
                        since_flush = 0

                    if completed_units % max(1, args.progress_every) == 0 or completed_units == len(pending_units):
                        remaining = max(len(pending_units) - completed_units, 0)
                        completed_api_requests = completed_units * prompts_per_unit
                        pct = (completed_units / len(pending_units) * 100) if pending_units else 100.0
                        elapsed_total = max(time.time() - run_start_time, 1e-9)
                        avg_rate = completed_units / elapsed_total
                        eta_seconds = (remaining / avg_rate) if avg_rate > 0 else 0.0
                        flush_print(
                            f"Progress: {completed_units}/{len(pending_units)} units completed ({pct:.1f}%), "
                            f"{completed_api_requests}/{len(pending_units) * prompts_per_unit} API requests completed, "
                            f"{remaining} units remaining, ETA {eta_seconds/60:.1f} min"
                        )

        if since_flush > 0:
            f.flush()

    flush_print(
        f"Completed run for {args.output}: wrote {completed_units} new rows, "
        f"failed {failed_units} units, "
        f"skipped {skipped_units} existing rows "
        f"(planned max API requests without resume: {total_api_requests})"
    )


if __name__ == "__main__":
    main()
