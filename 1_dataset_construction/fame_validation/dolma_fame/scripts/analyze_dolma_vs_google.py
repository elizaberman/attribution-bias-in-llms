"""Compare Dolma-based fame (infini-gram counts) against the Google-hits
metric used in the paper, separately for the multirace and intersectional
final datasets.

For each of the two datasets, writes (under $ATTRIBENCH_RESULTS/dolma_fame/<dataset>/):
  correlation_summary.csv       Pearson & Spearman, overall + per subgroup
  subgroup_balance.csv          Mean / std / median of log10 fame per subgroup,
                                 for both Google and Dolma
  pairwise_imbalance.csv        Pairwise KS + mean diff between subgroups,
                                 separately for each metric
  decile_shifts.csv             Per-author decile under Google vs Dolma + shift
  decile_shift_summary.csv      Single-row summary of shift magnitudes
  decile_crosstab.csv           Counts of (Google decile, Dolma decile) pairs
  report.md                     Human-readable summary
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- AttriBench path resolution ---
_ATTRIBENCH_ROOT = os.environ.get("ATTRIBENCH_ROOT", str(Path(__file__).resolve().parents[4]))
_ATTRIBENCH_RESULTS = os.environ.get("ATTRIBENCH_RESULTS", os.path.join(_ATTRIBENCH_ROOT, "results"))
_DATASETS = Path(_ATTRIBENCH_ROOT) / "1_dataset_construction" / "datasets"
_DOLMA_OUT = Path(_ATTRIBENCH_RESULTS) / "dolma_fame"
# ----------------------------------
OUT_ROOT = _DOLMA_OUT

AUTHORS_CSV = OUT_ROOT / "final_dataset_authors.csv"
DOLMA_CSV = OUT_ROOT / "dolma_counts.csv"

MR_PATH = _DATASETS / "multirace_with_quotes.csv"
IX_PATH = _DATASETS / "intersectional_with_quotes.csv"

MULTIRACE_ORDER = ["white", "latino", "asian", "black"]
INTERSECTIONAL_ORDER = [("white", "male"), ("white", "female"),
                       ("black", "male"), ("black", "female")]


def log10p1(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(x, dtype=float) + 1.0)


def _safe_corr(x: pd.Series, y: pd.Series) -> dict:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan}
    pr = stats.pearsonr(x[mask], y[mask])
    sr = stats.spearmanr(x[mask], y[mask])
    return {"n": int(mask.sum()), "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def _load_dolma_map() -> pd.DataFrame:
    counts = pd.read_csv(DOLMA_CSV)
    counts["log10_dolma"] = log10p1(counts["dolma_count"])
    return counts[["author_clean", "dolma_count", "log10_dolma"]]


def load_dataset(dataset: str) -> tuple[pd.DataFrame, list]:
    if dataset == "multirace":
        df = pd.read_csv(MR_PATH)
        subgroup_cols = ["race"]
        order = [(r,) for r in MULTIRACE_ORDER]
    elif dataset == "intersectional":
        df = pd.read_csv(IX_PATH)
        subgroup_cols = ["race", "gender"]
        order = INTERSECTIONAL_ORDER
    else:
        raise ValueError(dataset)
    dolma = _load_dolma_map()
    df = df.merge(dolma, on="author_clean", how="left")
    df = df.rename(columns={"log10_hits": "log10_google"})
    df["subgroup"] = (df[subgroup_cols].astype(str).agg("_".join, axis=1)
                      if len(subgroup_cols) > 1 else df["race"])
    return df, order


def correlations(df: pd.DataFrame, dataset: str, order: list) -> pd.DataFrame:
    rows: list[dict] = []
    rows.append({"scope": "quote-level (all)",
                 **_safe_corr(df["log10_google"], df["log10_dolma"])})
    # Author-level (dedup within dataset).
    au = df.drop_duplicates("author_clean")
    rows.append({"scope": "author-level (all)",
                 **_safe_corr(au["log10_google"], au["log10_dolma"])})
    if dataset == "multirace":
        for r in MULTIRACE_ORDER:
            sub = df[df["race"] == r]
            rows.append({"scope": f"quote-level/race={r}",
                         **_safe_corr(sub["log10_google"], sub["log10_dolma"])})
    else:
        for race, gender in INTERSECTIONAL_ORDER:
            sub = df[(df["race"] == race) & (df["gender"] == gender)]
            rows.append({"scope": f"quote-level/race={race},gender={gender}",
                         **_safe_corr(sub["log10_google"], sub["log10_dolma"])})
    return pd.DataFrame(rows)


def subgroup_balance(df: pd.DataFrame, dataset: str, order: list) -> pd.DataFrame:
    cols = ["race"] if dataset == "multirace" else ["race", "gender"]
    rows: list[dict] = []
    for keys in order:
        keys = (keys,) if not isinstance(keys, tuple) else keys
        mask = np.ones(len(df), dtype=bool)
        for col, val in zip(cols, keys):
            mask &= df[col] == val
        sub = df[mask]
        row = {col: val for col, val in zip(cols, keys)}
        row.update({
            "n_quotes": len(sub),
            "mean_log10_google": float(sub["log10_google"].mean()),
            "std_log10_google": float(sub["log10_google"].std()),
            "median_log10_google": float(sub["log10_google"].median()),
            "mean_log10_dolma": float(sub["log10_dolma"].mean()),
            "std_log10_dolma": float(sub["log10_dolma"].std()),
            "median_log10_dolma": float(sub["log10_dolma"].median()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_imbalance(df: pd.DataFrame, dataset: str, order: list) -> pd.DataFrame:
    cols = ["race"] if dataset == "multirace" else ["race", "gender"]
    rows: list[dict] = []
    keys = [(k,) if not isinstance(k, tuple) else k for k in order]
    for metric in ("log10_google", "log10_dolma"):
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                m1 = np.ones(len(df), dtype=bool)
                m2 = np.ones(len(df), dtype=bool)
                for col, v in zip(cols, k1):
                    m1 &= df[col] == v
                for col, v in zip(cols, k2):
                    m2 &= df[col] == v
                s1 = df.loc[m1, metric].dropna()
                s2 = df.loc[m2, metric].dropna()
                ks = stats.ks_2samp(s1, s2)
                rows.append({
                    "metric": metric,
                    "group_a": "_".join(k1),
                    "group_b": "_".join(k2),
                    "n_a": len(s1), "n_b": len(s2),
                    "mean_diff": float(s1.mean() - s2.mean()),
                    "ks_stat": float(ks.statistic),
                    "ks_p": float(ks.pvalue),
                })
    return pd.DataFrame(rows)


def decile_shifts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Decile based on the per-author fame, not per-quote (so each author is in
    one cell). Quote-level dataset is then joined to author-level deciles."""
    au = df.drop_duplicates("author_clean").copy()
    au = au[au["log10_google"].notna() & au["log10_dolma"].notna()]
    au["decile_google"] = pd.qcut(au["log10_google"], 10, labels=False,
                                  duplicates="drop") + 1
    au["decile_dolma"] = pd.qcut(au["log10_dolma"], 10, labels=False,
                                 duplicates="drop") + 1
    au["decile_shift"] = au["decile_dolma"] - au["decile_google"]
    crosstab = pd.crosstab(au["decile_google"], au["decile_dolma"])
    summary = {
        "n_authors": int(len(au)),
        "same_decile_pct": float((au["decile_shift"] == 0).mean() * 100),
        "within_1_decile_pct": float((au["decile_shift"].abs() <= 1).mean() * 100),
        "within_2_decile_pct": float((au["decile_shift"].abs() <= 2).mean() * 100),
        "mean_abs_shift": float(au["decile_shift"].abs().mean()),
        "max_abs_shift": int(au["decile_shift"].abs().max()),
    }
    keep = au[["author_clean", "log10_google", "log10_dolma",
               "decile_google", "decile_dolma", "decile_shift"]]
    return keep, crosstab, summary


FIGURE_DESCRIPTIONS = {
    "scatter_dolma_vs_google": (
        "**`figures/scatter_dolma_vs_google.{png,pdf}` — per-author scatter, "
        "log10(Google hits) vs log10(Dolma count + 1).** Each point is one "
        "unique author in this dataset, colored by subgroup. The dashed line "
        "is y=x. This figure shows the raw agreement between the two fame "
        "metrics: if Google hits were a noisy but unbiased proxy for a "
        "training-data-frequency notion of fame, points "
        "would scatter tightly around y=x, with no systematic offset between "
        "subgroups. Subgroup-specific clusters that sit above or below the "
        "diagonal indicate that a subgroup is over- or under-represented in "
        "Dolma relative to its Google hits."
    ),
    "subgroup_regression_fits": (
        "**`figures/subgroup_regression_fits.{png,pdf}` — same scatter, with "
        "one OLS regression line fit per subgroup.** Each line is computed "
        "on its subgroup's authors only; the legend reports the per-subgroup "
        "slope and within-subgroup Pearson r. This figure isolates whether "
        "the *form* of the Google→Dolma mapping is the same across "
        "subgroups: if all lines have similar slopes and intercepts, then "
        "Google hits and Dolma counts encode fame the same way for every "
        "subgroup and the choice of metric is largely a units question. "
        "Lines that are parallel but vertically offset would mean Google "
        "and Dolma agree on rank within a subgroup but disagree about the "
        "subgroup's absolute fame level; lines with different slopes mean "
        "the metrics disagree more for famous than obscure authors in some "
        "subgroups (or vice versa)."
    ),
    "subgroup_distribution": (
        "**`figures/subgroup_distribution.{png,pdf}` — violin plot of "
        "log10 fame per subgroup, side-by-side for Google (light grey) and "
        "Dolma (dark grey).** The paper's matched-fame design is constructed "
        "so that Google hit distributions across subgroups are nearly "
        "identical (light-grey violins should overlap heavily). This figure "
        "tests whether that balance survives switching to Dolma counts: if "
        "the dark-grey violins also overlap with similar medians, then the "
        "paper's per-subgroup comparisons are robust to which fame metric "
        "is used. Divergence between the dark-grey violins (different "
        "medians or shapes across subgroups) signals residual fame "
        "imbalance under Dolma that the Google-based matching did not "
        "remove."
    ),
    "decile_crosstab": (
        "**`figures/decile_crosstab.{png,pdf}` — 10×10 heatmap of how "
        "authors move when re-deciled by Dolma.** Rows are the author's "
        "fame decile under Google hits; columns are the decile under Dolma "
        "counts (both within this dataset's author set). Cell values are "
        "raw author counts; the colormap is row-normalized so that strongly "
        "diagonal cells indicate that authors in a given Google decile tend "
        "to land in the same Dolma decile. A tight diagonal would mean the "
        "two metrics are essentially interchangeable for decile-level "
        "analyses; off-diagonal mass means the choice of metric materially "
        "changes which authors are 'high-fame' vs 'low-fame'."
    ),
}


def render_report(dataset: str, corr: pd.DataFrame, balance: pd.DataFrame,
                  pairwise: pd.DataFrame, summary: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Dolma vs Google fame — {dataset} dataset\n")

    lines.append("## Correlations (log10 scale)\n")
    lines.append("How well does Dolma counting agree with Google hits on each "
                 "author's fame, on a log scale? Pearson measures linear "
                 "agreement on the log scale; Spearman measures rank agreement.\n")
    lines.append(corr.to_markdown(index=False, floatfmt=".3f"))

    lines.append("\n## Subgroup balance\n")
    lines.append("Mean / std / median log10 fame per subgroup. The paper's design "
                 "is balanced on Google hits; this table shows whether the same "
                 "subgroups remain balanced under Dolma counts.\n")
    lines.append(balance.to_markdown(index=False, floatfmt=".3f"))

    lines.append("\n## Pairwise subgroup imbalance (KS statistic + mean diff)\n")
    lines.append("For every pair of subgroups, two-sample KS statistic and "
                 "difference in mean log10 fame, computed separately under "
                 "Google hits and Dolma counts. Smaller values = better balance.\n")
    lines.append(pairwise.to_markdown(index=False, floatfmt=".3f"))

    lines.append("\n## Decile shifts (Google → Dolma)\n")
    lines.append("Authors are re-deciled by their Dolma count and compared to "
                 "their Google-hits decile. The columns below summarize how "
                 "stable the decile assignment is across metrics.\n")
    lines.append(pd.DataFrame([summary]).to_markdown(index=False, floatfmt=".2f"))

    lines.append("\n## Figures\n")
    lines.append("Three paper-styled figures accompany this report. Each is "
                 "saved as both PNG (for previewing) and PDF (for inclusion in "
                 "the manuscript), under `figures/`.\n")
    for key in ("scatter_dolma_vs_google", "subgroup_regression_fits",
                "subgroup_distribution", "decile_crosstab"):
        lines.append("- " + FIGURE_DESCRIPTIONS[key])

    out_path.write_text("\n".join(lines))


def run_for_dataset(dataset: str) -> None:
    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    df, order = load_dataset(dataset)

    corr = correlations(df, dataset, order)
    corr.to_csv(out_dir / "correlation_summary.csv", index=False)

    balance = subgroup_balance(df, dataset, order)
    balance.to_csv(out_dir / "subgroup_balance.csv", index=False)

    pairwise = pairwise_imbalance(df, dataset, order)
    pairwise.to_csv(out_dir / "pairwise_imbalance.csv", index=False)

    deciles, crosstab, summary = decile_shifts(df)
    deciles.to_csv(out_dir / "decile_shifts.csv", index=False)
    crosstab.to_csv(out_dir / "decile_crosstab.csv")
    pd.DataFrame([summary]).to_csv(out_dir / "decile_shift_summary.csv",
                                   index=False)

    render_report(dataset, corr, balance, pairwise, summary,
                  out_dir / "report.md")
    print(f"[{dataset}] n={len(df)} quotes, {df['author_clean'].nunique()} authors")
    print(f"  -> wrote {out_dir}/")


def main() -> None:
    for d in ("multirace", "intersectional"):
        run_for_dataset(d)


if __name__ == "__main__":
    main()
