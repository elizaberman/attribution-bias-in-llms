"""
Generate appendix-ready subgroup variance tables from precomputed metrics.

Reads subgroup_metrics.csv files (quote-level) from the results/quote_metrics
folder and writes compact variance tables per dataset folder.
"""

from __future__ import annotations

# --- AttriBench path resolution ---
import os as _os
from pathlib import Path as _Path
_ATTRIBENCH_ROOT = _os.environ.get(
    "ATTRIBENCH_ROOT", str(_Path(__file__).resolve().parents[2])
)
_ATTRIBENCH_RESULTS = _os.environ.get(
    "ATTRIBENCH_RESULTS", _os.path.join(_ATTRIBENCH_ROOT, "results")
)
# -------------------------------------------------------------

import argparse
from pathlib import Path
from typing import List

import pandas as pd

BASE = Path(f"{_ATTRIBENCH_RESULTS}/quote_metrics")


def build_table(input_csv: Path, out_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    keep = [
        "model",
        "prompt_type",
        "race_ethnicity",
        "gender",
        "n_quotes",
        "variance_accuracy",
        "variance_suppression",
        "std_accuracy",
        "std_suppression",
        "se_accuracy",
        "se_suppression",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build subgroup variance tables.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["intersectional", "multirace", "intersectional_rag", "multirace_rag"],
        help="Dataset folders under results/quote_metrics",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        input_csv = BASE / ds / "subgroup_metrics.csv"
        if not input_csv.exists():
            print(f"[warning] missing {input_csv}")
            continue
        out_csv = BASE / ds / "appendix_variance_table.csv"
        build_table(input_csv, out_csv)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
