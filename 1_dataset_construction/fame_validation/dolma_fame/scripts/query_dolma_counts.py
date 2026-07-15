"""Query infini-gram for Dolma v1.7 counts of each author_clean string.

Reads:
  $ATTRIBENCH_RESULTS/dolma_fame/final_dataset_authors.csv

Writes (incrementally, resume-safe):
  $ATTRIBENCH_RESULTS/dolma_fame/dolma_counts.csv

Columns:
  author_clean, dolma_count, approx, n_tokens, latency_ms, error, queried_at

infini-gram API: https://infini-gram.io/api_doc
  POST https://api.infini-gram.io/  (Content-Type: application/json)
  body: {"index": "v4_dolma-v1_7_llama", "query_type": "count", "query": "<name>"}

Each request typically returns in tens of milliseconds. We use a small thread
pool (default 8 workers) and a per-request retry with exponential backoff to
ride out transient errors. Output is appended one row at a time and the script
skips authors already present in the CSV when re-run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.infini-gram.io/"
INDEX = "v4_dolma-v1_7_llama"

# --- AttriBench path resolution ---
_ATTRIBENCH_ROOT = os.environ.get("ATTRIBENCH_ROOT", str(Path(__file__).resolve().parents[4]))
_ATTRIBENCH_RESULTS = os.environ.get("ATTRIBENCH_RESULTS", os.path.join(_ATTRIBENCH_ROOT, "results"))
_DATASETS = Path(_ATTRIBENCH_ROOT) / "1_dataset_construction" / "datasets"
_DOLMA_OUT = Path(_ATTRIBENCH_RESULTS) / "dolma_fame"
# ----------------------------------
IN_PATH = _DOLMA_OUT / "final_dataset_authors.csv"
OUT_PATH = _DOLMA_OUT / "dolma_counts.csv"

FIELDS = [
    "author_clean", "dolma_count", "approx", "n_tokens",
    "latency_ms", "error", "queried_at",
]


def query_once(name: str, timeout: float = 30.0) -> dict:
    payload = {"index": INDEX, "query_type": "count", "query": name}
    r = requests.post(API_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def query_with_retry(name: str, max_retries: int = 6,
                     base_backoff: float = 2.0,
                     per_request_delay: float = 0.0) -> dict:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            data = query_once(name)
            # Server-side error returned in payload (not HTTP error).
            if "error" in data and "count" not in data:
                return {"error": str(data["error"])}
            if per_request_delay:
                time.sleep(per_request_delay)
            return data
        except requests.HTTPError as e:
            # 403 from WAF means we're rate-limited; back off hard.
            sc = e.response.status_code if e.response is not None else None
            last_exc = e
            sleep_for = (base_backoff ** attempt) * (4.0 if sc == 403 else 1.0)
            time.sleep(sleep_for)
        except Exception as e:  # noqa: BLE001 — retry on any transient
            last_exc = e
            time.sleep(base_backoff ** attempt)
    return {"error": f"max_retries_exhausted: {last_exc!r}"}


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df["author_clean"].astype(str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent requests (keep modest; shared research API)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only query the first N remaining authors (debug)")
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--output", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    authors_df = pd.read_csv(args.input)
    all_names = authors_df["author_clean"].astype(str).tolist()
    done = already_done(args.output)
    todo = [n for n in all_names if n not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Authors total: {len(all_names)}  already-done: {len(done)}  todo: {len(todo)}",
          file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output.exists()
    write_lock = threading.Lock()

    f = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()

    n_done = 0
    n_err = 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(query_with_retry, name): name for name in todo}
            for fut in as_completed(futures):
                name = futures[fut]
                data = fut.result()
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                row = {
                    "author_clean": name,
                    "dolma_count": data.get("count") if "count" in data else "",
                    "approx": data.get("approx") if "approx" in data else "",
                    "n_tokens": len(data.get("tokens", [])) if "tokens" in data else "",
                    "latency_ms": data.get("latency") if "latency" in data else "",
                    "error": data.get("error", ""),
                    "queried_at": ts,
                }
                if row["error"]:
                    n_err += 1
                with write_lock:
                    writer.writerow(row)
                    f.flush()
                n_done += 1
                if n_done % 100 == 0 or n_done == len(todo):
                    rate = n_done / max(1e-6, time.time() - t0)
                    print(f"  progress: {n_done}/{len(todo)}  errors: {n_err}  "
                          f"rate: {rate:.1f} req/s",
                          file=sys.stderr)
    finally:
        f.close()
    print(f"Done. wrote {n_done} rows (errors: {n_err}). Output: {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
