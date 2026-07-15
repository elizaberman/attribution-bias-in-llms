"""
Add statistical significance annotations to subgroup metrics.

This script reads a pre-aggregated `subgroup_metrics.csv` (quote-level aggregation already
performed) and computes 95% confidence intervals for a selected metric (accuracy or
suppression). For each (model, prompt_type) group, it identifies the reference subgroup(s)
based on the best mean and marks which other subgroups are not significantly different
using a z-test on the difference between subgroup means:

    z = abs(reference_mean - subgroup_mean) / sqrt(reference_se^2 + subgroup_se^2)

Assuming independence between subgroup estimates.

Inputs:
- subgroup_metrics.csv with columns including:
  model, prompt_type, subgroup columns (race_ethnicity[, gender]), n_quotes,
  mean_accuracy, variance_accuracy, std_accuracy, se_accuracy,
  mean_suppression, variance_suppression, std_suppression, se_suppression

Outputs:
1) subgroup_metrics_with_significance.csv
2) heatmap_bolding_table.csv

DATASET_MODE controls subgrouping:
- "multirace" -> subgroup columns = [race_ethnicity]
- "intersectional" -> subgroup columns = [race_ethnicity, gender]

METRIC selects the metric used for CIs and comparisons:
- "accuracy" or "suppression"
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
from typing import List, Tuple

import numpy as np
import pandas as pd

# -------------------------
# User-settable parameters
# -------------------------
DATASET_MODE = "multirace"  # or "intersectional"
METRIC = "accuracy"  # or "suppression"

DEFAULT_INPUT = Path(
    f"{_ATTRIBENCH_RESULTS}/quote_metrics/subgroup_metrics.csv"
)
DEFAULT_OUTDIR = Path(
    f"{_ATTRIBENCH_RESULTS}/quote_metrics"
)
DEFAULT_RAG_INPUT = Path(
    f"{_ATTRIBENCH_RESULTS}/quote_metrics/rag/subgroup_metrics.csv"
)


def validate_columns(df: pd.DataFrame, dataset_mode: str, metric: str) -> List[str]:
    required_base = ["model", "prompt_type", "n_quotes"]
    if dataset_mode == "multirace":
        subgroup_cols = ["race_ethnicity"]
    elif dataset_mode == "intersectional":
        subgroup_cols = ["race_ethnicity", "gender"]
    else:
        raise ValueError(f"Unsupported DATASET_MODE: {dataset_mode}")

    metric_cols = get_metric_columns(metric)
    required = required_base + subgroup_cols + metric_cols

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return subgroup_cols


def get_metric_columns(metric: str) -> List[str]:
    metric = metric.lower().strip()
    if metric not in {"accuracy", "suppression"}:
        raise ValueError(f"Unsupported METRIC: {metric}")
    return [
        f"mean_{metric}",
        f"variance_{metric}",
        f"std_{metric}",
        f"se_{metric}",
    ]


def compute_confidence_intervals(
    df: pd.DataFrame, metric: str
) -> pd.DataFrame:
    df = df.copy()
    mean_col = f"mean_{metric}"
    var_col = f"variance_{metric}"
    std_col = f"std_{metric}"
    se_col = f"se_{metric}"

    # Compute std/se from variance where needed
    df["metric_mean"] = df[mean_col]
    df["metric_variance"] = df[var_col]

    # std from variance if missing
    df["metric_std"] = df[std_col]
    needs_std = df["metric_std"].isna() & df["metric_variance"].notna()
    df.loc[needs_std, "metric_std"] = np.sqrt(df.loc[needs_std, "metric_variance"])

    # se from std and n_quotes if missing
    df["metric_se"] = df[se_col]
    needs_se = df["metric_se"].isna() & df["metric_std"].notna()
    n = df.loc[needs_se, "n_quotes"].astype(float)
    df.loc[needs_se, "metric_se"] = df.loc[needs_se, "metric_std"] / np.sqrt(n.replace(0, np.nan))

    df["ci_lower"] = df["metric_mean"] - 1.96 * df["metric_se"]
    df["ci_upper"] = df["metric_mean"] + 1.96 * df["metric_se"]
    return df


def _build_subgroup_label(row: pd.Series, dataset_mode: str) -> str:
    if dataset_mode == "multirace":
        return str(row.get("race_ethnicity", ""))

    # intersectional
    gender = str(row.get("gender", ""))
    race = str(row.get("race_ethnicity", ""))

    gender_map = {
        "male": "M",
        "m": "M",
        "female": "F",
        "f": "F",
        "man": "M",
        "woman": "F",
    }
    gkey = gender.strip().lower()
    gshort = gender_map.get(gkey)
    if gshort:
        return f"{gshort}/{race.title()}" if race else f"{gshort}/"
    return f"{gender}/{race}" if race else str(gender)


def annotate_significance_vs_group_reference(
    df: pd.DataFrame, subgroup_cols: List[str], metric: str
) -> pd.DataFrame:
    df = df.copy()
    group_cols = ["model", "prompt_type"]

    metric = metric.lower().strip()
    if metric == "accuracy":
        ref_agg = "max"
    elif metric == "suppression":
        ref_agg = "min"
    else:
        raise ValueError(f"Unsupported METRIC: {metric}")

    df["reference_mean_in_group"] = df.groupby(group_cols)["metric_mean"].transform(ref_agg)
    df["is_reference_mean"] = df["metric_mean"] == df["reference_mean_in_group"]

    # For each group, compare each row to all reference rows and keep the smallest z-score.
    def _mark_group(group: pd.DataFrame) -> pd.DataFrame:
        ref_rows = group[group["is_reference_mean"]]
        if ref_rows.empty:
            group["reference_se_in_group"] = np.nan
            group["difference_from_reference"] = np.nan
            group["se_difference"] = np.nan
            group["z_difference"] = np.nan
            group["not_significantly_different_from_reference"] = False
            group["asterisk_not_significant"] = False
            group["asterisk_on_reference"] = False
            return group

        ref_mean = ref_rows["metric_mean"].iloc[0]

        def _best_vs_reference(row: pd.Series) -> Tuple[float, float, float, float, bool]:
            if row["is_reference_mean"]:
                ref_se = row["metric_se"]
                diff = 0.0
                se_diff = np.sqrt(ref_se ** 2 + row["metric_se"] ** 2) if pd.notna(ref_se) and pd.notna(row["metric_se"]) else np.nan
                if pd.notna(se_diff) and se_diff != 0:
                    z = abs(diff) / se_diff
                else:
                    z = np.nan
                return ref_se, diff, se_diff, z, True

            best_ref_se = np.nan
            best_diff = np.nan
            best_se_diff = np.nan
            best_z = np.nan
            for _, ref in ref_rows.iterrows():
                ref_se = ref["metric_se"]
                row_se = row["metric_se"]
                if pd.notna(ref_se) and pd.notna(row_se):
                    se_diff = np.sqrt(ref_se ** 2 + row_se ** 2)
                else:
                    se_diff = np.nan

                if pd.notna(se_diff) and se_diff != 0:
                    z = abs(ref_mean - row["metric_mean"]) / se_diff
                else:
                    z = np.nan

                if pd.isna(best_z) and not pd.isna(z):
                    best_ref_se, best_diff, best_se_diff, best_z = ref_se, ref_mean - row["metric_mean"], se_diff, z
                elif not pd.isna(z) and not pd.isna(best_z) and z < best_z:
                    best_ref_se, best_diff, best_se_diff, best_z = ref_se, ref_mean - row["metric_mean"], se_diff, z

            if pd.isna(best_z):
                best_diff = ref_mean - row["metric_mean"]
            return best_ref_se, best_diff, best_se_diff, best_z, False

        results = group.apply(_best_vs_reference, axis=1, result_type="expand")
        results.columns = [
            "reference_se_in_group",
            "difference_from_reference",
            "se_difference",
            "z_difference",
            "_is_reference_row",
        ]
        group = pd.concat([group, results], axis=1)
        group["not_significantly_different_from_reference"] = (
            group["_is_reference_row"]
            | (group["z_difference"] <= 1.96)
        )
        group.loc[group["z_difference"].isna() & ~group["_is_reference_row"], "not_significantly_different_from_reference"] = False
        group["asterisk_not_significant"] = group["not_significantly_different_from_reference"]
        group["asterisk_on_reference"] = False
        if len(ref_rows) == 1:
            non_ref = ~group["is_reference_mean"]
            if non_ref.any():
                all_significant = group.loc[non_ref, "z_difference"].gt(1.96).fillna(False).all()
                group.loc[group["is_reference_mean"], "asterisk_on_reference"] = bool(all_significant)
        group = group.drop(columns=["_is_reference_row"])
        return group

    pieces = []
    for keys, _grp in df.groupby(group_cols):
        marked = _mark_group(_grp)
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                if col not in marked.columns:
                    marked[col] = val
        else:
            if group_cols[0] not in marked.columns:
                marked[group_cols[0]] = keys
        pieces.append(marked)
    df = pd.concat(pieces, ignore_index=True)
    df["subgroup_label"] = df.apply(lambda r: _build_subgroup_label(r, DATASET_MODE), axis=1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate subgroup metrics with significance vs group reference."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to subgroup_metrics.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default=DATASET_MODE,
        choices=["multirace", "intersectional"],
        help="Dataset mode",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=METRIC,
        choices=["accuracy", "suppression"],
        help="Metric selector",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run both datasets and both metrics using the default directory layout",
    )
    parser.add_argument(
        "--run_all_rag",
        action="store_true",
        help="Run both datasets and both metrics for RAG and write to intersectional_rag/multirace_rag",
    )
    parser.add_argument(
        "--rag_intersectional_input",
        type=Path,
        default=None,
        help="Path to RAG intersectional subgroup_metrics.csv",
    )
    parser.add_argument(
        "--rag_multirace_input",
        type=Path,
        default=None,
        help="Path to RAG multirace subgroup_metrics.csv",
    )
    args = parser.parse_args()

    if args.run_all:
        base = Path(
            f"{_ATTRIBENCH_RESULTS}/quote_metrics"
        )
        runs = [
            ("intersectional", "accuracy"),
            ("intersectional", "suppression"),
            ("multirace", "accuracy"),
            ("multirace", "suppression"),
        ]
        for dataset_mode, metric in runs:
            input_csv = base / dataset_mode / "subgroup_metrics.csv"
            outdir = base / dataset_mode
            _run_once(input_csv, outdir, dataset_mode, metric)
        return

    if args.run_all_rag:
        base = Path(
            f"{_ATTRIBENCH_RESULTS}/quote_metrics"
        )
        int_input = args.rag_intersectional_input
        mul_input = args.rag_multirace_input
        if int_input is None or mul_input is None:
            raise SystemExit(
                "run_all_rag requires --rag_intersectional_input and --rag_multirace_input"
            )
        runs = [
            ("intersectional", "accuracy", int_input, base / "intersectional_rag"),
            ("intersectional", "suppression", int_input, base / "intersectional_rag"),
            ("multirace", "accuracy", mul_input, base / "multirace_rag"),
            ("multirace", "suppression", mul_input, base / "multirace_rag"),
        ]
        for dataset_mode, metric, input_csv, outdir in runs:
            _run_once(input_csv, outdir, dataset_mode, metric)
        return

    _run_once(args.input_csv, args.outdir, args.dataset_mode, args.metric)


def _run_once(input_csv: Path, outdir: Path, dataset_mode: str, metric: str) -> None:
    df = pd.read_csv(input_csv)
    subgroup_cols = validate_columns(df, dataset_mode, metric)

    annotated = compute_confidence_intervals(df, metric)
    annotated = annotate_significance_vs_group_reference(annotated, subgroup_cols, metric)
    if annotated.index.name is not None or annotated.index.nlevels > 1:
        annotated = annotated.reset_index()

    outdir.mkdir(parents=True, exist_ok=True)

    annotated_out = outdir / f"subgroup_metrics_with_significance_{metric}.csv"
    annotated.to_csv(annotated_out, index=False)

    heatmap_cols = ["model", "prompt_type"] + subgroup_cols + [
        "metric_mean",
        "ci_lower",
        "ci_upper",
        "is_reference_mean",
        "asterisk_on_reference",
    ]
    heatmap = annotated[heatmap_cols].copy()
    heatmap_out = outdir / f"heatmap_bolding_table_{metric}.csv"
    heatmap.to_csv(heatmap_out, index=False)

    print(f"\nDataset={dataset_mode} Metric={metric}")
    print("Preview: subgroup_metrics_with_significance.csv")
    print(annotated.head(10).to_string(index=False))

    print("\nPer-group bolding summary:")
    for (model, prompt), group in annotated.groupby(["model", "prompt_type"]):
        print(f"\n{model} | {prompt}")
        for _, row in group.iterrows():
            subgroup = " / ".join(str(row[c]) for c in subgroup_cols)
            mean = row["metric_mean"]
            lo = row["ci_lower"]
            hi = row["ci_upper"]
            bold = row["is_reference_mean"]
            asterisk = row["asterisk_on_reference"]
            print(
                f"- {subgroup}: mean={mean:.4f} CI=[{lo:.4f}, {hi:.4f}] "
                f"bold={bold} asterisk={asterisk}"
            )


if __name__ == "__main__":
    main()
