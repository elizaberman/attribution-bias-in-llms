"""Build the union list of unique authors across the two final datasets.

Reads:
  1_dataset_construction/datasets/multirace_with_quotes.csv
  1_dataset_construction/datasets/intersectional_with_quotes.csv

Writes:
  $ATTRIBENCH_RESULTS/dolma_fame/final_dataset_authors.csv

One row per unique author_clean. Columns:
  author_clean, author_alt_name, log10_hits_google, google_hits,
  in_multirace (0/1), in_intersectional (0/1), n_quotes_multirace,
  n_quotes_intersectional, race, gender.

If an author appears in both datasets with different log10_hits (shouldn't
happen, but defensively): take the max.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# --- AttriBench path resolution ---
_ATTRIBENCH_ROOT = os.environ.get("ATTRIBENCH_ROOT", str(Path(__file__).resolve().parents[4]))
_ATTRIBENCH_RESULTS = os.environ.get("ATTRIBENCH_RESULTS", os.path.join(_ATTRIBENCH_ROOT, "results"))
_DATASETS = Path(_ATTRIBENCH_ROOT) / "1_dataset_construction" / "datasets"
_DOLMA_OUT = Path(_ATTRIBENCH_RESULTS) / "dolma_fame"
# ----------------------------------
MR_PATH = _DATASETS / "multirace_with_quotes.csv"
IX_PATH = _DATASETS / "intersectional_with_quotes.csv"
OUT_PATH = _DOLMA_OUT / "final_dataset_authors.csv"


def _load(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["author_clean"].notna()].copy()
    df["__dataset__"] = tag
    return df


def main() -> None:
    mr = _load(MR_PATH, "multirace")
    ix = _load(IX_PATH, "intersectional")
    combined = pd.concat([mr, ix], ignore_index=True)

    # Per-author quote counts per dataset.
    counts = (
        combined.groupby(["author_clean", "__dataset__"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"multirace": "n_quotes_multirace",
                         "intersectional": "n_quotes_intersectional"})
    )
    for col in ("n_quotes_multirace", "n_quotes_intersectional"):
        if col not in counts:
            counts[col] = 0
    counts["in_multirace"] = (counts["n_quotes_multirace"] > 0).astype(int)
    counts["in_intersectional"] = (counts["n_quotes_intersectional"] > 0).astype(int)

    # One row per author: take first non-null values for metadata, max log10_hits.
    agg = (
        combined.groupby("author_clean")
        .agg(
            author_alt_name=("author_alt_name", lambda s: next((x for x in s if pd.notna(x) and str(x).strip()), pd.NA)),
            google_hits=("google_hits", "max"),
            log10_hits_google=("log10_hits", "max"),
            race=("race", lambda s: next((x for x in s if pd.notna(x)), pd.NA)),
            gender=("gender", lambda s: next((x for x in s if pd.notna(x)), pd.NA)),
        )
    )

    out = agg.join(counts).reset_index()
    out = out[[
        "author_clean", "author_alt_name", "race", "gender",
        "google_hits", "log10_hits_google",
        "in_multirace", "in_intersectional",
        "n_quotes_multirace", "n_quotes_intersectional",
    ]]
    out = out.sort_values("author_clean").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(out)} unique authors to {OUT_PATH}")
    print(f"  in_multirace only:      {((out.in_multirace==1) & (out.in_intersectional==0)).sum()}")
    print(f"  in_intersectional only: {((out.in_multirace==0) & (out.in_intersectional==1)).sum()}")
    print(f"  in both:                {((out.in_multirace==1) & (out.in_intersectional==1)).sum()}")
    print(f"  missing log10_hits:     {out.log10_hits_google.isna().sum()}")


if __name__ == "__main__":
    main()
