#!/usr/bin/env python3
"""Score author fame via Google hit counts, and select a fame-matched sample.

Stage (c) of dataset construction. Fame is `se_results_count` from a DataForSEO live
SERP query for the exact-match keyword `"Author Name"`. Two modes, run in order:

  score   For every unique author in --input-csv, query the SERP endpoint and append
          the hit count to --out-csv. This is how the reference group's fame
          distribution is built.

  match   Define decile bins on log10(hits+1) from a reference distribution
          (--reference-hits, i.e. a `score` run's output), then query authors from
          --input-csv and keep a subset that fills the same per-decile quotas,
          stopping once every quota is met. This yields a pool whose fame
          distribution matches the reference, so that downstream attribution
          disparities cannot be confounded by author prominence.

The paper scores the nonwhite authors, then matches White authors to those deciles:

  python get_fame.py score
  python get_fame.py match

Both modes use the live endpoint rather than task_post/task_get, so no calls are
posted that aren't fetched; both throttle to ~18.75 requests/min (one per 3.2s); and
both are resumable — every scored author is written out as it completes, and reruns
skip authors already present in the output. `match` fills quotas from the cache
before making any new call, and caches *every* scored author whether or not it is
selected, since the call is already paid for.

REFERENCE-ONLY: calls a paid API and reads constraints files that are not distributed
with this repo. The released fame values ship in `datasets/*_with_quotes.csv`
(`google_hits` / `log10_hits`).

Requires DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD. Search and throttle settings are
env-overridable (LOCATION_CODE, LANGUAGE_CODE, DEVICE, SLEEP_BETWEEN_REQUESTS,
TIMEOUT_SECS, MAX_RETRIES).
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import time
from datetime import datetime
from math import log10
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ----------------------------
# CONFIG
# ----------------------------
AUTHOR_COL = "author"
HITS_COL = "google_hits"
SUCCESS_COL = "success"

CACHE_FIELDNAMES = ["author", "query", "google_hits", "success", "error", "timestamp"]

LOGIN = os.getenv("DATAFORSEO_LOGIN")
PASSWORD = os.getenv("DATAFORSEO_PASSWORD")

# LIVE endpoint
LIVE_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/regular"

# Search settings
LOCATION_CODE = int(os.getenv("LOCATION_CODE", "2840"))  # US
LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "en")
DEVICE = os.getenv("DEVICE", "desktop")

# Rate limiting: 20/min => >= 3.0s between requests. Use 3.2 for safety.
SLEEP_BETWEEN_REQUESTS = float(os.getenv("SLEEP_BETWEEN_REQUESTS", "3.2"))
TIMEOUT_SECS = int(os.getenv("TIMEOUT_SECS", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

EPS = 1e-9
N_DECILES = 10

# Defaults reproduce the paper's run: score nonwhite authors, then match White ones.
DEFAULT_SCORE_INPUT = "fame_control/constraints_ds/constrained_nonwhite_cap10_len10_30_min2.csv"
DEFAULT_SCORE_OUTPUT = "fame_control/outputs_live_hits/nonwhite_author_hits.csv"
DEFAULT_MATCH_INPUT = "fame_control/constraints_ds/constrained_white_cap10_len10_30_min2.csv"
DEFAULT_MATCH_REFERENCE = DEFAULT_SCORE_OUTPUT
DEFAULT_MATCH_OUTDIR = "fame_control/outputs_white_decile_match_live"


# ----------------------------
# DataForSEO client (shared by both modes)
# ----------------------------
def now_iso() -> str:
    return datetime.now().isoformat()


def basic_auth_header(login: str, password: str) -> str:
    token = base64.b64encode(f"{login}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Authorization": basic_auth_header(LOGIN, PASSWORD),
        "Content-Type": "application/json",
    })
    return session


def extract_hits(resp_json: dict):
    """Pull se_results_count out of resp_json["tasks"][0]["result"][0]."""
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
    if isinstance(hits, int):
        return hits, None
    try:
        return int(hits), None
    except Exception:
        return None, "se_results_count missing/non-numeric"


def fetch_hits(session: requests.Session, author: str) -> dict:
    """One throttled, retried live SERP lookup. Always returns a cache row."""
    query = f'"{author}"'
    payload = [{
        "keyword": query,
        "location_code": LOCATION_CODE,
        "language_code": LANGUAGE_CODE,
        "device": DEVICE,
        "tag": author,
    }]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(LIVE_URL, json=payload, timeout=TIMEOUT_SECS)
            j = r.json()
            hits, err = extract_hits(j)
            if err is None:
                return {
                    "author": author,
                    "query": query,
                    "google_hits": int(hits),
                    "success": True,
                    "error": "",
                    "timestamp": now_iso(),
                }
            last_err = err
            # rate limit handling: back off a full minute, else exponential
            if "rate" in err.lower() and "limit" in err.lower():
                time.sleep(60.0)
            else:
                time.sleep(2 ** (attempt - 1))
        except Exception as e:
            last_err = str(e)
            time.sleep(2 ** (attempt - 1))

    return {
        "author": author,
        "query": query,
        "google_hits": "",
        "success": False,
        "error": last_err or "unknown error",
        "timestamp": now_iso(),
    }


# ----------------------------
# IO helpers
# ----------------------------
def normalize_authors(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def safe_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    return False


def load_unique_authors(path: str) -> list[str]:
    df = pd.read_csv(path)
    if AUTHOR_COL not in df.columns:
        raise ValueError(f"Missing '{AUTHOR_COL}' column in {path}")
    authors = normalize_authors(df[AUTHOR_COL]).dropna()
    authors = authors[authors != ""].unique().tolist()
    return sorted(authors)


def load_scored_authors(path: Path) -> set[str]:
    """Authors already present in a cache/output CSV (for resume)."""
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            a = (row.get("author") or "").strip()
            if a:
                done.add(a)
    return done


def append_cache_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CACHE_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CACHE_FIELDNAMES})


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CACHE_FIELDNAMES)
    df = pd.read_csv(path)
    if "author" in df.columns:
        df["author"] = normalize_authors(df["author"])
    return df


def load_reference_hits(path: str) -> pd.DataFrame:
    """Unique, successful (author, google_hits) rows from a `score` run's output."""
    df = pd.read_csv(path)
    if AUTHOR_COL not in df.columns or HITS_COL not in df.columns:
        raise ValueError(f"Reference hits file must have columns '{AUTHOR_COL}', '{HITS_COL}'")

    if SUCCESS_COL in df.columns:
        df = df[df[SUCCESS_COL].apply(safe_bool)].copy()

    df[AUTHOR_COL] = normalize_authors(df[AUTHOR_COL])
    df = df.dropna(subset=[AUTHOR_COL, HITS_COL]).copy()
    df = df[df[AUTHOR_COL] != ""].copy()
    df = df.drop_duplicates(subset=[AUTHOR_COL], keep="first").copy()

    df[HITS_COL] = df[HITS_COL].astype(int)
    return df[[AUTHOR_COL, HITS_COL]]


# ----------------------------
# Decile matching (match mode)
# ----------------------------
def compute_edges_and_targets(reference_hits: np.ndarray):
    """Deciles on log10(hits+1). Returns edges (len 11, -inf/+inf) and per-decile targets."""
    scores = np.log10(reference_hits.astype(float) + 1.0)
    qs = np.linspace(0, 1, N_DECILES + 1)
    raw = np.quantile(scores, qs)

    edges = raw.copy()
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + EPS

    edges[0] = -np.inf
    edges[-1] = np.inf

    bins = np.searchsorted(edges, scores, side="right") - 1
    bins = np.clip(bins, 0, N_DECILES - 1)
    targets = np.bincount(bins, minlength=N_DECILES).astype(int)
    return edges, targets


def assign_decile(edges: np.ndarray, hits: int) -> int:
    score = log10(float(hits) + 1.0)
    d = int(np.searchsorted(edges, score, side="right") - 1)
    return max(0, min(N_DECILES - 1, d))


def write_matched_sample(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["author", "google_hits", "decile", "timestamp"])
    df.to_csv(path, index=False)


def write_targets(path: Path, edges: np.ndarray, targets: np.ndarray) -> None:
    pd.DataFrame({
        "decile": list(range(N_DECILES)),
        "target_n": targets,
        "edge_left_log10": [edges[i] for i in range(N_DECILES)],
        "edge_right_log10": [edges[i + 1] for i in range(N_DECILES)],
    }).to_csv(path, index=False)


def write_fill_status(path: Path, filled: np.ndarray, targets: np.ndarray) -> None:
    pd.DataFrame({
        "decile": list(range(N_DECILES)),
        "filled": filled.astype(int),
        "target": targets.astype(int),
        "remaining": np.maximum(0, targets - filled).astype(int),
    }).to_csv(path, index=False)


def print_fill(filled: np.ndarray, targets: np.ndarray) -> None:
    parts = [f"d{d}:{int(filled[d])}/{int(targets[d])}" for d in range(N_DECILES)]
    print("  " + "  ".join(parts), flush=True)


# ----------------------------
# Modes
# ----------------------------
def run_score(args) -> None:
    input_csv = args.input_csv
    out_csv = Path(args.out_csv)

    print(f"Input CSV: {Path(input_csv).resolve()}")
    authors = load_unique_authors(input_csv)
    print(f"Unique authors found: {len(authors):,}")

    done = load_scored_authors(out_csv)
    todo = [a for a in authors if a not in done]

    print("\nPlanned workload (printed BEFORE any API calls):")
    print(f"  Already completed: {len(done):,} (from {out_csv})")
    print(f"  Remaining authors: {len(todo):,}")
    print(f"  EXACT HTTP requests: {len(todo):,} (1 LIVE request per author)")
    print(f"  Throttle: {SLEEP_BETWEEN_REQUESTS:.2f}s between requests (~{60.0/SLEEP_BETWEEN_REQUESTS:.1f}/min)\n")

    if not todo:
        print("Nothing to do.")
        print(f"Output: {out_csv.resolve()}")
        return

    session = make_session()
    completed = 0
    total = len(todo)

    try:
        for author in todo:
            row = fetch_hits(session, author)
            append_cache_row(out_csv, row)
            completed += 1

            if row["success"]:
                print(f"[{completed}/{total}] ✓ {author}: {int(row['google_hits']):,}", flush=True)
            else:
                print(f"[{completed}/{total}] ✗ {author}: {row['error']}", flush=True)

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    except KeyboardInterrupt:
        print("\nStopped early (Ctrl+C). Progress is saved; rerun to resume.", flush=True)

    print("\nDONE / STOPPED.")
    print(f"Output CSV: {out_csv.resolve()}")


def run_match(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_csv = outdir / args.cache_name
    matched_csv = outdir / args.matched_name
    targets_csv = outdir / args.targets_name
    fill_status_csv = outdir / args.fill_status_name

    # 1) Reference distribution -> decile edges and targets
    reference_df = load_reference_hits(args.reference_hits)
    edges, targets = compute_edges_and_targets(reference_df[HITS_COL].values)
    write_targets(targets_csv, edges, targets)

    # 2) Candidate pool
    pool = load_unique_authors(args.input_csv)

    # 3) Already-scored candidates
    cache_df = read_cache(cache_csv)
    cache_done = (
        set(cache_df["author"].dropna().astype(str).str.strip().tolist())
        if "author" in cache_df.columns else set()
    )

    # 4) Fill quotas from the cache first, deterministically
    matched_rows: list[dict] = []
    filled = np.zeros(N_DECILES, dtype=int)
    selected_authors: set[str] = set()

    cache_success = cache_df.copy()
    if not cache_success.empty:
        if "success" in cache_success.columns:
            cache_success = cache_success[cache_success["success"].apply(safe_bool)].copy()
        cache_success = cache_success.dropna(subset=["author", "google_hits"]).copy()

        # deterministic order: earlier timestamp then author
        sort_cols = ["timestamp", "author"] if "timestamp" in cache_success.columns else ["author"]
        cache_success = cache_success.sort_values(sort_cols, kind="mergesort")

        for _, row in cache_success.iterrows():
            a = str(row["author"]).strip()
            if not a or a in selected_authors:
                continue
            hits = int(row["google_hits"])
            d = assign_decile(edges, hits)
            if filled[d] < targets[d]:
                filled[d] += 1
                selected_authors.add(a)
                matched_rows.append({
                    "author": a,
                    "google_hits": hits,
                    "decile": d,
                    "timestamp": row.get("timestamp", ""),
                })
            if int(np.maximum(0, targets - filled).sum()) == 0:
                break

    remaining_needed = int(np.maximum(0, targets - filled).sum())
    todo = [a for a in pool if a not in cache_done]

    # 5) Print the plan BEFORE any API call
    print("=" * 70)
    print("MATCH AUTHORS TO REFERENCE FAME DECILES (LIVE, THROTTLED)")
    print("=" * 70)
    print(f"Reference hits: {args.reference_hits}")
    print(f"Candidate pool: {args.input_csv}")
    print(f"OUTDIR:         {outdir.resolve()}")
    print()
    print("Reference decile targets:")
    for d in range(N_DECILES):
        print(f"  decile {d}: target {int(targets[d])}")
    print(f"Total reference authors used (targets sum): {int(targets.sum()):,}")
    print()
    print("Resume / cache status:")
    print(f"  Candidates in pool:                {len(pool):,}")
    print(f"  Candidates already scored (cache): {len(cache_done):,}  ({cache_csv})")
    print(f"  Candidates remaining to score:     {len(todo):,}")
    print()
    print("Quota fill from cache (before new calls):")
    print_fill(filled, targets)
    print(f"Remaining needed (authors) across deciles: {remaining_needed:,}")
    print()
    # Exact call counts are unknowable ahead of time: an author's decile is only known
    # once queried, so a lookup may or may not land in a decile that still needs filling.
    print("Planned workload (printed BEFORE any API calls):")
    print(f"  Throttle: {SLEEP_BETWEEN_REQUESTS:.2f}s between requests (~{60.0/SLEEP_BETWEEN_REQUESTS:.1f}/min)")
    print(f"  MIN possible new HTTP requests (best-case):  {remaining_needed:,}")
    print(f"  MAX possible new HTTP requests (worst-case): {len(todo):,}")
    if args.max_new_calls > 0:
        print(f"  HARD CAP this run (--max-new-calls): {args.max_new_calls:,}")
    print()

    write_matched_sample(matched_csv, matched_rows)
    write_fill_status(fill_status_csv, filled, targets)

    if remaining_needed <= 0:
        print("All quotas already satisfied from cached scores. No API calls needed.")
        print(f"Matched sample: {matched_csv.resolve()}")
        print(f"Fill status:    {fill_status_csv.resolve()}")
        return

    if not todo:
        print("No remaining candidates to score. Quotas not fully met.")
        print(f"Matched sample: {matched_csv.resolve()}")
        print(f"Fill status:    {fill_status_csv.resolve()}")
        return

    # 6) Live calls
    session = make_session()
    total_new = 0

    try:
        for author in todo:
            if remaining_needed <= 0:
                print("\nAll decile quotas satisfied — stopping early.", flush=True)
                break

            if args.max_new_calls and total_new >= args.max_new_calls:
                print("\nReached --max-new-calls cap — stopping.", flush=True)
                break

            row = fetch_hits(session, author)

            # ALWAYS cache: the call is already paid for, selected or not.
            append_cache_row(cache_csv, row)
            total_new += 1

            if row["success"]:
                hits = int(row["google_hits"])
                d = assign_decile(edges, hits)

                if filled[d] < targets[d]:
                    filled[d] += 1
                    remaining_needed = int(np.maximum(0, targets - filled).sum())
                    selected_authors.add(author)
                    matched_rows.append({
                        "author": author,
                        "google_hits": hits,
                        "decile": d,
                        "timestamp": row["timestamp"],
                    })
                    write_matched_sample(matched_csv, matched_rows)
                    print(f"[new {total_new}] ✓ SELECT  {author}: {hits:,}  decile={d}  ({filled[d]}/{targets[d]})", flush=True)
                else:
                    print(f"[new {total_new}] – scored (NOT selected; decile full) {author}: {hits:,}  decile={d}", flush=True)
            else:
                print(f"[new {total_new}] ✗ FAILED {author}: {row['error']}", flush=True)

            if total_new % 25 == 0:
                write_fill_status(fill_status_csv, filled, targets)
                print_fill(filled, targets)

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    except KeyboardInterrupt:
        print("\nStopped early (Ctrl+C). Progress saved. Rerun to resume.", flush=True)

    write_matched_sample(matched_csv, matched_rows)
    write_fill_status(fill_status_csv, filled, targets)

    print("\nDONE / STOPPED.")
    print(f"Targets:        {targets_csv.resolve()}")
    print(f"Cache:          {cache_csv.resolve()}  (ALL scored candidates)")
    print(f"Matched sample: {matched_csv.resolve()}  (selected subset)")
    print(f"Fill status:    {fill_status_csv.resolve()}")
    print("\nFinal fill:")
    print_fill(filled, targets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score author fame via DataForSEO Google hits, and select a fame-matched sample.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_score = sub.add_parser("score", help="Score every unique author in --input-csv.")
    p_score.add_argument("--input-csv", default=DEFAULT_SCORE_INPUT,
                         help="Author pool CSV (needs an 'author' column).")
    p_score.add_argument("--out-csv", default=DEFAULT_SCORE_OUTPUT,
                         help="Appended incrementally; rerunning resumes from it.")
    p_score.set_defaults(func=run_score)

    p_match = sub.add_parser(
        "match", help="Score candidates and select a subset matching the reference fame deciles.")
    p_match.add_argument("--input-csv", default=DEFAULT_MATCH_INPUT,
                         help="Candidate pool CSV (needs an 'author' column).")
    p_match.add_argument("--reference-hits", default=DEFAULT_MATCH_REFERENCE,
                         help="Output of a `score` run; defines the decile targets.")
    p_match.add_argument("--outdir", default=os.getenv("OUTDIR", DEFAULT_MATCH_OUTDIR))
    p_match.add_argument("--max-new-calls", type=int, default=int(os.getenv("MAX_NEW_WHITE_CALLS", "0")),
                         help="Hard cap on new lookups this run (0 = no cap).")
    p_match.add_argument("--cache-name", default="white_hits_cache.csv")
    p_match.add_argument("--matched-name", default="white_decile_matched_sample.csv")
    p_match.add_argument("--targets-name", default="nonwhite_decile_targets.csv")
    p_match.add_argument("--fill-status-name", default="white_decile_fill_status.csv")
    p_match.set_defaults(func=run_match)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not LOGIN or not PASSWORD:
        raise SystemExit("Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in your environment.")
    args.func(args)


if __name__ == "__main__":
    main()
