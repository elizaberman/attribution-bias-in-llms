#!/usr/bin/env python3
"""Measure quote-author co-occurrence on the open web (paper Appendix A.7.2, Table 5).

A robustness check on fame balancing. Author-level fame may not capture how often a
specific quote is publicly associated with its author, so for each (author, quote)
pair this queries the DataForSEO live SERP for pages containing BOTH the author name
and the full quote text, using the exact-match keyword `"Author Name" "full quote text"`.

If co-occurrence were systematically lower for nonwhite authors at equal fame, it
could explain the attribution gaps; the paper reports broadly similar rates across
subgroups, so it does not.

Writes $ATTRIBENCH_RESULTS/fame_validation/cooccurrence_hits.csv, with incremental
output and resume. NOTE: nothing in this repo aggregates that file into Table 5 —
the aggregation step is not distributed.

Adapted from ../../fame_scripts/get_fame.py.

Usage:
  # Pilot (20 quotes):
  python query_cooccurrence_google.py --pilot 20

  # Full intersectional dataset:
  python query_cooccurrence_google.py --dataset intersectional

  # Full multirace (only pairs not already in intersectional):
  python query_cooccurrence_google.py --dataset multirace

Requires DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD env vars.
"""

import os
import sys
import time
import csv
import base64
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

LIVE_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/regular"

LOCATION_CODE = 2840  # US
LANGUAGE_CODE = "en"
DEVICE = "desktop"
SLEEP_BETWEEN_REQUESTS = 3.2
TIMEOUT_SECS = 60
MAX_RETRIES = 4

SCRIPT_DIR = Path(__file__).resolve().parent

# --- AttriBench path resolution ---
_ATTRIBENCH_ROOT = os.environ.get("ATTRIBENCH_ROOT", str(SCRIPT_DIR.parents[2]))
_ATTRIBENCH_RESULTS = os.environ.get(
    "ATTRIBENCH_RESULTS", os.path.join(_ATTRIBENCH_ROOT, "results")
)
# ----------------------------------

_DATASETS = Path(_ATTRIBENCH_ROOT) / "1_dataset_construction" / "datasets"
INTERSECTIONAL_CSV = _DATASETS / "intersectional_with_quotes.csv"
MULTIRACE_CSV = _DATASETS / "multirace_with_quotes.csv"

OUT_CSV = Path(_ATTRIBENCH_RESULTS) / "fame_validation" / "cooccurrence_hits.csv"

FIELDS = ["quote_id", "author_clean", "quote", "race", "gender",
          "google_hits_author", "keyword", "cooccurrence_hits",
          "success", "error", "timestamp"]


def now_iso():
    return datetime.now().isoformat()


def basic_auth_header(login, password):
    token = base64.b64encode(f"{login}:{password}".encode()).decode("ascii")
    return f"Basic {token}"


def load_pairs(dataset):
    inter = pd.read_csv(INTERSECTIONAL_CSV)
    multi = pd.read_csv(MULTIRACE_CSV)

    if dataset == "intersectional":
        return inter
    elif dataset == "multirace":
        inter_keys = set(zip(inter["author_clean"], inter["quote"]))
        mask = ~multi.apply(lambda r: (r["author_clean"], r["quote"]) in inter_keys, axis=1)
        return multi[mask].copy()
    elif dataset == "both":
        combined = pd.concat([inter, multi]).drop_duplicates(subset=["author_clean", "quote"])
        return combined
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_done(path):
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = (row.get("quote_id") or "").strip()
            if qid:
                done.add(qid)
    return done


def build_keyword(author, quote):
    author_clean = author.strip().replace('"', '')
    quote_clean = quote.strip().replace('"', '')
    return f'"{author_clean}" "{quote_clean}"'


def extract_hits(resp_json):
    if resp_json.get("status_code") != 20000:
        return None, f"top status_code={resp_json.get('status_code')} msg={resp_json.get('status_message')}"
    tasks = resp_json.get("tasks") or []
    if not tasks:
        return None, "no tasks"
    t0 = tasks[0]
    if t0.get("status_code") != 20000:
        return None, f"task status_code={t0.get('status_code')} msg={t0.get('status_message')}"
    result = t0.get("result") or []
    if not result:
        return None, "no result"
    hits = result[0].get("se_results_count")
    if hits is None:
        return 0, None
    try:
        return int(hits), None
    except Exception:
        return None, f"se_results_count not numeric: {hits}"


def fetch_cooccurrence(session, author, quote):
    keyword = build_keyword(author, quote)
    payload = [{
        "keyword": keyword,
        "location_code": LOCATION_CODE,
        "language_code": LANGUAGE_CODE,
        "device": DEVICE,
    }]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(LIVE_URL, json=payload, timeout=TIMEOUT_SECS)
            j = r.json()
            hits, err = extract_hits(j)
            if err is None:
                return hits, keyword, None
            last_err = err
            if "rate" in err.lower() and "limit" in err.lower():
                time.sleep(60.0)
            else:
                time.sleep(2 ** (attempt - 1))
        except Exception as e:
            last_err = str(e)
            time.sleep(2 ** (attempt - 1))

    return None, keyword, last_err


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["intersectional", "multirace", "both"],
                        default="intersectional")
    parser.add_argument("--pilot", type=int, default=None,
                        help="Run only N quotes (stratified by subgroup) for testing")
    parser.add_argument("--output", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    login = os.getenv("DATAFORSEO_LOGIN")
    password = os.getenv("DATAFORSEO_PASSWORD")
    if not login or not password:
        sys.exit("Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables.")

    df = load_pairs(args.dataset)
    print(f"Dataset '{args.dataset}': {len(df)} pairs", file=sys.stderr)

    if args.pilot:
        pilot_rows = []
        for _, group in df.groupby(["race", "gender"]):
            sorted_g = group.sort_values("log10_hits", ascending=False)
            n_per = max(1, args.pilot // df.groupby(["race", "gender"]).ngroups)
            top = sorted_g.head(n_per // 2 + 1)
            bot = sorted_g.tail(n_per - len(top) if n_per > len(top) else n_per // 2)
            pilot_rows.append(pd.concat([top, bot]))
        df = pd.concat(pilot_rows).drop_duplicates(subset=["quote_id"])
        print(f"Pilot mode: {len(df)} pairs selected", file=sys.stderr)

    done = load_done(args.output)
    todo = df[~df["quote_id"].astype(str).isin(done)]
    print(f"Already done: {len(done)}, remaining: {len(todo)}", file=sys.stderr)

    if todo.empty:
        print("Nothing to do.", file=sys.stderr)
        return

    session = requests.Session()
    session.headers.update({
        "Authorization": basic_auth_header(login, password),
        "Content-Type": "application/json",
    })

    completed = 0
    errors = 0
    total = len(todo)

    print(f"\nStarting {total} queries at ~{SLEEP_BETWEEN_REQUESTS}s each "
          f"(est. {total * SLEEP_BETWEEN_REQUESTS / 3600:.1f}h)\n", file=sys.stderr)

    try:
        for _, row in todo.iterrows():
            hits, keyword, err = fetch_cooccurrence(
                session, row["author_clean"], row["quote"]
            )
            out_row = {
                "quote_id": row["quote_id"],
                "author_clean": row["author_clean"],
                "quote": row["quote"],
                "race": row["race"],
                "gender": row["gender"],
                "google_hits_author": row.get("google_hits", ""),
                "keyword": keyword,
                "cooccurrence_hits": hits if hits is not None else "",
                "success": err is None,
                "error": err or "",
                "timestamp": now_iso(),
            }
            append_row(args.output, out_row)
            completed += 1

            if err:
                errors += 1
                print(f"[{completed}/{total}] X {row['author_clean']}: {err}", flush=True)
            else:
                print(f"[{completed}/{total}] OK {row['author_clean']}: "
                      f"cooccur={hits:,} | fame={row.get('google_hits', '?')}",
                      flush=True)

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C). Progress saved; rerun to resume.", flush=True)

    print(f"\nDone. {completed} queries, {errors} errors. Output: {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
