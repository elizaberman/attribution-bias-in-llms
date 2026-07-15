#!/usr/bin/env python3
"""Build the author-level author universe for fame balancing.

``universe_authors.csv`` is the input to ``mk_new_datasets.py`` (the fame-balancing
stage). It is the quote-level cleaned dataset aggregated to one row per author.

The input is the **quote-capped** cleaned quote-level table produced by
``prune_filter/create_clean_dataset.py`` (which applies the <=10 quotes/author
cap). That table is already fully merged: it carries the validated demographic labels
(``race``, ``gender`` — the consensus of the Wikidata / gpt-4o-mini / Perplexity labels,
with the per-source columns retained) and the fame proxy (``google_hits`` -> ``log10_hits``).
This script collapses it to author level, taking each author's (constant) race / gender /
fame and counting their quotes.

Output columns (one row per author): author_clean, race, gender, log10_hits, count

Provenance verified against the paper's own run: the shipped ``universe_authors.csv`` is
byte-identical to the pipeline's universe, and aggregating the capped cleaned dataset (the
original run's ``constrained_dataset.csv``) reproduces it exactly for 99.4% of authors with
100% author-set coverage (residual = a few fame re-queries / quote-count drift between
snapshots, plus a downstream author-selection step). Feed this script the **capped** cleaned
dataset; an uncapped intermediate will not reproduce the shipped counts.

Reference-only: the released AttriBench CSVs in ``1_dataset_construction/datasets/`` are
the canonical benchmark; this reproduces the balancer's input from the cleaned dataset.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

_ATTRIBENCH_ROOT = os.environ.get(
    "ATTRIBENCH_ROOT",
    str(Path(__file__).resolve().parent.parent),  # 1_dataset_construction/
)
DEFAULT_CLEAN_DATASET = (
    Path(_ATTRIBENCH_ROOT) / "outputs_clean_filtered" / "clean_dataset.csv"
)

UNIVERSE_COLUMNS = ["author_clean", "race", "gender", "log10_hits", "count"]


def build_universe(clean_dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(clean_dataset_path)

    required = {"author_clean", "race", "gender", "log10_hits"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{clean_dataset_path} is missing required columns: {sorted(missing)}")

    # Keep only quote rows that carry a validated label and a fame score; authors
    # without fame cannot be fame-matched and are excluded from the universe.
    df = df.dropna(subset=["author_clean", "race", "gender", "log10_hits"])
    df = df[df["author_clean"].astype(str).str.strip() != ""]

    # One row per author: race/gender/log10_hits are constant per author; count = #quotes
    # (already capped at 10/author by the prune stage).
    universe = (
        df.groupby("author_clean", sort=True)
          .agg(
              race=("race", "first"),
              gender=("gender", "first"),
              log10_hits=("log10_hits", "first"),
              count=("author_clean", "size"),
          )
          .reset_index()
    )

    # Sanity check: a normalized author should map to a single (race, gender). Warn if not.
    conflicts = df.groupby("author_clean")[["race", "gender"]].nunique()
    ambiguous = conflicts[(conflicts["race"] > 1) | (conflicts["gender"] > 1)]
    if len(ambiguous):
        print(f"Warning: {len(ambiguous)} author(s) had >1 race/gender across quotes; "
              f"kept the first. Examples: {list(ambiguous.index[:5])}")

    return universe[UNIVERSE_COLUMNS]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate the cleaned quote-level dataset into the author-level fame-balancing universe."
    )
    parser.add_argument("--clean-dataset", default=str(DEFAULT_CLEAN_DATASET),
                        help="Quote-level clean_dataset.csv from prune_filter/.")
    parser.add_argument("--out", default="universe_authors.csv",
                        help="Output author-level CSV (input to mk_new_datasets.py).")
    args = parser.parse_args()

    universe = build_universe(args.clean_dataset)
    universe.to_csv(args.out, index=False)
    print(f"Wrote {len(universe)} authors -> {args.out}")
    print(universe["race"].value_counts().to_dict())


if __name__ == "__main__":
    main()
