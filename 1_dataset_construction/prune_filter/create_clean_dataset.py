#!/usr/bin/env python3
"""
Create a cleaned, quote-length-filtered author dataset (auxiliary fame deciles kept)
without pandas/numpy.

Rules (per request):
- Clean author names (remove leading dashes/quotes/tildes, strip parentheses /
  nicknames / work titles; store removed bits as alternate_name).
- Quote word-count filter: keep quotes with 5-30 words (inclusive).
- Keep authors with even a single quote; cap to 10 quotes per author within
  each (race, gender) subgroup.
- Include authors that only have one quote.
- Preserve existing fame scores (google_hits) if already fetched; leave blank
  otherwise so they can be fetched later.

Outputs (in OUTDIR):
- clean_dataset.csv: filtered quotes with cleaned names and fame info.
- subgroup_counts.csv: authors/quotes per race×gender, plus how many have hits.
- decile_edges_log10.csv: decile edges on log10(google_hits) for authors
  with hits > 0.
- decile_subgroup_counts.csv: authors per decile by race×gender.
- missing_fame_authors.csv: unique authors lacking a fame score (helper list).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


# --- AttriBench path resolution ---
_ATTRIBENCH_ROOT = os.environ.get(
    "ATTRIBENCH_ROOT", str(Path(__file__).resolve().parents[2])
)
_ATTRIBENCH_RESULTS = os.environ.get(
    "ATTRIBENCH_RESULTS", os.path.join(_ATTRIBENCH_ROOT, "results")
)
# ----------------------------------

# Defaults (all overridable via CLI flags below). The raw JSTET corpus and the
# fame-hit caches are NOT shipped with the repo — supply your own copies. See
# this module's README; AttriBench's canonical entry point is the released CSVs.
DEFAULT_JSTET_CSV = Path(
    os.environ.get(
        "JSTET_CSV",
        os.path.join(_ATTRIBENCH_ROOT, "1_dataset_construction", "raw", "jstet_dataset.csv"),
    )
)
DEFAULT_NONWHITE_HITS = Path(_ATTRIBENCH_RESULTS) / "fame" / "nonwhite_author_hits.csv"
DEFAULT_WHITE_HITS = Path(_ATTRIBENCH_RESULTS) / "fame" / "white_hits_cache.csv"
# Written where the downstream fame-balancing stage (fame_scripts/)
# reads it. This dir is gitignored — generated data never enters the tracked tree.
DEFAULT_OUTDIR = Path(_ATTRIBENCH_ROOT) / "1_dataset_construction" / "outputs_clean_filtered"

RACE_COL = "race_ethnicity"
GENDER_COL = "gender"
AUTHOR_COL = "author"
QUOTE_COL = "quote"
QUOTE_ID_COL = "quote_id"

SUBGROUPS: List[Tuple[str, str]] = [
    ("white", "male"), ("white", "female"),
    ("black", "male"), ("black", "female"),
    ("asian", "male"), ("asian", "female"),
    ("latino", "male"), ("latino", "female"),
]

CAP_PER_AUTHOR = 10
MIN_WORDS = 5
MAX_WORDS = 30

WORD_REGEX = re.compile(r"\b\w+\b")
# Allow Latin scripts (basic + common extensions); drop anything outside for "English-ish" quotes.
_LATIN_RANGES = [
    (0x0020, 0x007E),   # basic ASCII
    (0x00A0, 0x00FF),   # Latin-1 supplement
    (0x0100, 0x017F),   # Latin Extended-A
    (0x0180, 0x024F),   # Latin Extended-B
    (0x1E00, 0x1EFF),   # Latin Extended Additional
]
_ARABIC_RANGES = [
    (0x0600, 0x06FF), (0x0750, 0x077F),
    (0x08A0, 0x08FF), (0xFB50, 0xFDFD), (0xFE70, 0xFEFF),
]


@dataclass
class FameHit:
    google_hits: int
    source: str  # e.g., white_cache / nonwhite_hits
    timestamp: str | None


def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_text(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def _in_ranges(codepoint: int, ranges) -> bool:
    return any(lo <= codepoint <= hi for lo, hi in ranges)


def is_latinish(text: str) -> bool:
    """
    Filter out quotes that are clearly non-English by rejecting text containing
    Arabic or non-Latin letters. Allows common Latin accents.
    """
    for ch in text:
        cp = ord(ch)
        if _in_ranges(cp, _ARABIC_RANGES):
            return False
        if ch.isalpha():
            if not _in_ranges(cp, _LATIN_RANGES):
                return False
    return True


def count_words(text: str) -> int:
    return len(WORD_REGEX.findall(text or ""))


def clean_author(raw: str) -> tuple[str, str]:
    """
    Remove leading punctuation, strip parentheses/nicknames/work titles, and
    keep removed bits in alternate_name.
    Also split out AKA aliases into the alternate name.
    """
    name = (raw or "").strip()
    # Remove surrounding quotes/backticks
    name = name.strip('"\''"`` ")
    # Remove leading dashes/tildes/en dashes/em dashes
    name = re.sub(r"^[\-\u2013\u2014~]+\s*", "", name)

    alt_parts: list[str] = []

    # AKA handling
    aka_split = re.split(r"\s+(?:aka|a\.k\.a\.)\s+", name, flags=re.IGNORECASE)
    if len(aka_split) > 1:
        name = aka_split[0].strip()
        alt_parts.extend([p.strip() for p in aka_split[1:] if p.strip()])

    # Parenthetical nicknames / descriptors -> alternate_name
    paren_parts = re.findall(r"\(([^)]*)\)", name)
    if paren_parts:
        alt_parts.extend(p.strip() for p in paren_parts if p.strip())
        name = re.sub(r"\s*\([^)]*\)", "", name).strip()

    # If there's a spaced dash/emdash separator, treat RHS as alt/descriptor
    if re.search(r"\s+[\-\u2013\u2014]\s+", name):
        lhs, rhs = re.split(r"\s+[\-\u2013\u2014]\s+", name, maxsplit=1)
        name = lhs.strip()
        if rhs.strip():
            alt_parts.append(rhs.strip())

    # Trailing all-lowercase phrase (e.g., appended work titles) -> alternate_name
    tokens = name.split()
    tail = []
    for tok in reversed(tokens):
        if tok == tok.lower() and re.search(r"[a-z]", tok) and not re.search(r"[A-Z]", tok):
            tail.append(tok)
        else:
            break
    if len(tail) >= 3:
        # Treat trailing lowercase phrase as non-name content (e.g., work title); drop it.
        keep_len = len(tokens) - len(tail)
        name = " ".join(tokens[:keep_len]).strip()

    name = re.sub(r"\s+", " ", name).strip(" ,;")

    alt = "; ".join(sorted(set(alt_parts))) if alt_parts else ""
    return name, alt


def load_fame_hits(path: Path, source: str) -> dict[str, FameHit]:
    hits: dict[str, FameHit] = {}
    if not path.exists():
        return hits
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            success = str(row.get("success", "")).strip().lower() == "true"
            if not success:
                continue
            try:
                gh_raw = row.get("google_hits", "")
                google_hits = int(float(gh_raw)) if gh_raw not in (None, "") else 0
            except ValueError:
                continue
            author_raw = (row.get("author") or "").strip()
            hits[author_raw] = FameHit(
                google_hits=google_hits,
                source=source,
                timestamp=row.get("timestamp"),
            )
    return hits


def compute_decile_edges(log_hits: List[float]) -> List[float]:
    if not log_hits:
        return []
    log_hits_sorted = sorted(log_hits)
    # statistics.quantiles returns 9 cut-points for n=10
    quantiles = statistics.quantiles(
        log_hits_sorted, n=10, method="inclusive"
    ) if len(log_hits_sorted) > 1 else [log_hits_sorted[0]] * 9
    edges = [log_hits_sorted[0]] + quantiles + [log_hits_sorted[-1]]
    # Ensure non-decreasing edges
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    return edges


def assign_decile(log_hit: float, edges: List[float]) -> int:
    # edges length 11; deciles 0-9
    if not edges:
        return -1
    for i in range(1, len(edges)):
        if log_hit <= edges[i] or i == len(edges) - 1:
            return i - 1
    return -1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clean and length-filter the raw JSTET author dataset."
    )
    ap.add_argument("--jstet-csv", type=Path, default=DEFAULT_JSTET_CSV,
                    help="Raw JSTET quotes CSV (bring your own; not shipped).")
    ap.add_argument("--nonwhite-hits", type=Path, default=DEFAULT_NONWHITE_HITS,
                    help="Fame-hit cache for non-white authors (optional).")
    ap.add_argument("--white-hits", type=Path, default=DEFAULT_WHITE_HITS,
                    help="Fame-hit cache for white authors (optional).")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help="Where to write the cleaned dataset + segmentation CSVs.")
    args = ap.parse_args()

    DATASET_CSV = args.jstet_csv
    NONWHITE_HITS = args.nonwhite_hits
    WHITE_HITS = args.white_hits
    OUTDIR = args.outdir

    ensure_outdir(OUTDIR)

    # Load fame hits caches
    fame_by_raw = {}
    fame_by_raw.update(load_fame_hits(WHITE_HITS, "white_cache"))
    fame_by_raw.update(load_fame_hits(NONWHITE_HITS, "nonwhite_hits"))

    required_cols = {RACE_COL, GENDER_COL, AUTHOR_COL, QUOTE_COL}
    rows: list[dict] = []

    # Stage 1: load + clean + length filter
    with DATASET_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            race = normalize_text(row.get(RACE_COL))
            gender = normalize_text(row.get(GENDER_COL))
            if (race, gender) not in SUBGROUPS:
                continue
            author_raw = (row.get(AUTHOR_COL) or "").strip()
            # Drop clearly non-author entries
            if author_raw.lower() == "claudia in baby proof":
                continue
            quote = (row.get(QUOTE_COL) or "").strip()
            if not author_raw or not quote:
                continue
            if not is_latinish(quote):
                continue
            quote_words = count_words(quote)
            if not (MIN_WORDS <= quote_words <= MAX_WORDS):
                continue
            author_clean, alt_name = clean_author(author_raw)
            if not author_clean:
                continue
            rows.append({
                "_orig_idx": idx,
                "quote_id": row.get(QUOTE_ID_COL, ""),
                "author_raw": author_raw,
                "author_clean": author_clean,
                "author_alt_name": alt_name,
                "quote": quote,
                "quote_words": quote_words,
                "category": row.get("category", ""),
                "gender": gender,
                "gender_source": row.get("gender_source", ""),
                "race": race,
                "race_source": row.get("race_source", ""),
                "chatgpt_gender": row.get("chatgpt_gender", ""),
                "perplexity_gender": row.get("perplexity_gender", ""),
                "chatgpt_race": row.get("chatgpt_race", ""),
                "perplexity_race": row.get("perplexity_race", ""),
            })

    # Stage 2: cap quotes per (race, gender, author_clean)
    rows.sort(key=lambda r: (r["race"], r["gender"], r["author_clean"], r["_orig_idx"]))
    kept: list[dict] = []
    cap_counter: dict[tuple[str, str, str], int] = defaultdict(int)
    raw_names_by_author: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    alt_parts_by_author: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for r in rows:
        key = (r["race"], r["gender"], r["author_clean"])
        if cap_counter[key] >= CAP_PER_AUTHOR:
            continue
        cap_counter[key] += 1
        raw_names_by_author[key].add(r["author_raw"])
        if r["author_alt_name"]:
            alt_parts_by_author[key].add(r["author_alt_name"])
        kept.append(r)

    # Stage 3: merge alt names and fame hits per author
    fame_by_author: dict[tuple[str, str, str], FameHit] = {}
    for key, raw_names in raw_names_by_author.items():
        for raw_name in raw_names:
            hit = fame_by_raw.get(raw_name)
            if hit:
                fame_by_author[key] = hit
                break

    edges: List[float] = []
    log_hits: List[float] = [
        math.log10(hit.google_hits)
        for hit in fame_by_author.values()
        if hit.google_hits > 0
    ]
    edges = compute_decile_edges(log_hits)

    for r in kept:
        key = (r["race"], r["gender"], r["author_clean"])
        alt_joined = "; ".join(sorted(alt_parts_by_author.get(key, [])))
        r["author_alt_name"] = alt_joined

        hit = fame_by_author.get(key)
        if hit:
            r["google_hits"] = hit.google_hits
            r["hits_source"] = hit.source
            r["hits_timestamp"] = hit.timestamp or ""
            r["log10_hits"] = math.log10(hit.google_hits) if hit.google_hits > 0 else ""
            r["decile"] = assign_decile(r["log10_hits"], edges) if r["log10_hits"] != "" else ""
        else:
            r["google_hits"] = ""
            r["hits_source"] = ""
            r["hits_timestamp"] = ""
            r["log10_hits"] = ""
            r["decile"] = ""

    # Stage 4: write outputs
    dataset_path = OUTDIR / "clean_dataset.csv"
    ensure_outdir(dataset_path.parent)
    fieldnames = [
        "quote_id", "author_clean", "author_alt_name", "author_raw", "quote", "quote_words",
        "race", "gender", "category", "gender_source", "race_source",
        "chatgpt_gender", "perplexity_gender", "chatgpt_race", "perplexity_race",
        "google_hits", "log10_hits", "decile", "hits_source", "hits_timestamp",
    ]
    with dataset_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in kept:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Subgroup counts
    subgroup_counts_path = OUTDIR / "subgroup_counts.csv"
    with subgroup_counts_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames_counts = [
            "race", "gender", "n_authors", "n_quotes", "n_authors_with_hits",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames_counts)
        writer.writeheader()
        for race, gender in SUBGROUPS:
            subset = [r for r in kept if r["race"] == race and r["gender"] == gender]
            authors = {r["author_clean"] for r in subset}
            authors_with_hits = {
                r["author_clean"]
                for r in subset
                if r.get("google_hits") not in ("", None)
            }
            writer.writerow({
                "race": race,
                "gender": gender,
                "n_authors": len(authors),
                "n_quotes": len(subset),
                "n_authors_with_hits": len(authors_with_hits),
            })

    # Decile edges
    edges_path = OUTDIR / "decile_edges_log10.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["decile", "edge_log10"])
        writer.writeheader()
        if edges:
            for i in range(10):
                writer.writerow({"decile": i, "edge_log10": edges[i]})
            writer.writerow({"decile": "max", "edge_log10": edges[-1]})

    # Decile subgroup counts (authors per decile)
    decile_counts_path = OUTDIR / "decile_subgroup_counts.csv"
    with decile_counts_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["decile", "race", "gender", "n_authors"],
        )
        writer.writeheader()
        for d in range(10):
            for race, gender in SUBGROUPS:
                subset = {
                    r["author_clean"]
                    for r in kept
                    if r["race"] == race and r["gender"] == gender and r.get("decile") == d
                }
                writer.writerow({
                    "decile": d,
                    "race": race,
                    "gender": gender,
                    "n_authors": len(subset),
                })

    # Missing fame helper list
    missing_path = OUTDIR / "missing_fame_authors.csv"
    with missing_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["author_clean", "author_alt_name", "race", "gender", "example_quote"],
        )
        writer.writeheader()
        seen_missing = set()
        for r in kept:
            if r.get("google_hits") in ("", None):
                key = (r["author_clean"], r["race"], r["gender"])
                if key in seen_missing:
                    continue
                seen_missing.add(key)
                writer.writerow({
                    "author_clean": r["author_clean"],
                    "author_alt_name": r["author_alt_name"],
                    "race": r["race"],
                    "gender": r["gender"],
                    "example_quote": r["quote"],
                })

    # Console summary
    total_authors = len({(r["race"], r["gender"], r["author_clean"]) for r in kept})
    total_quotes = len(kept)
    print(f"Wrote dataset: {dataset_path}")
    print(f"Wrote subgroup counts: {subgroup_counts_path}")
    print(f"Wrote decile edges: {edges_path}")
    print(f"Wrote decile subgroup counts: {decile_counts_path}")
    print(f"Wrote missing fame list: {missing_path}")
    print(f"Totals -> authors: {total_authors:,} | quotes: {total_quotes:,}")


if __name__ == "__main__":
    main()
