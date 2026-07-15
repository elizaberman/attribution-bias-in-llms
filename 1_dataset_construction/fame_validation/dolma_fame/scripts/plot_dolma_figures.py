"""Paper-styled figures comparing Dolma counts to Google hits, per dataset.

Three figures per dataset (multirace, intersectional), written to
  $ATTRIBENCH_RESULTS/dolma_fame/<dataset>/figures/:

  1) scatter_dolma_vs_google.{png,pdf}
     Per-author scatter of log10(Google hits) vs log10(Dolma+1), colored by
     subgroup. Correlation reported in the title; y=x reference line.

  2) subgroup_distribution.{png,pdf}
     Side-by-side violins of log10(Google hits) vs log10(Dolma+1) per
     subgroup. Visualizes whether the paper's matched-fame design still
     holds under Dolma counts.

  3) decile_crosstab.{png,pdf}
     Heatmap of (Google decile, Dolma decile) cell counts. Strong diagonal =
     deciles agree; off-diagonal mass = re-ordering under Dolma.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

mpl.use("Agg")

# --- AttriBench path resolution ---
_ATTRIBENCH_ROOT = os.environ.get("ATTRIBENCH_ROOT", str(Path(__file__).resolve().parents[4]))
_ATTRIBENCH_RESULTS = os.environ.get("ATTRIBENCH_RESULTS", os.path.join(_ATTRIBENCH_ROOT, "results"))
_DATASETS = Path(_ATTRIBENCH_ROOT) / "1_dataset_construction" / "datasets"
_DOLMA_OUT = Path(_ATTRIBENCH_RESULTS) / "dolma_fame"
# ----------------------------------
OUT_ROOT = _DOLMA_OUT

MULTIRACE_ORDER = ["white", "latino", "asian", "black"]
INTERSECTIONAL_ORDER = ["white_male", "white_female", "black_male", "black_female"]

# Paper-friendly palette; greys for the bulk distribution, accent per subgroup.
SUBGROUP_COLOR = {
    "white": "#4E79A7",
    "latino": "#F28E2B",
    "asian": "#76B7B2",
    "black": "#E15759",
    "white_male": "#4E79A7",
    "white_female": "#A0CBE8",
    "black_male": "#E15759",
    "black_female": "#FF9D9A",
}

SUBGROUP_LABEL = {
    "white": "White",
    "latino": "Latino",
    "asian": "Asian",
    "black": "Black",
    "white_male": "White male",
    "white_female": "White female",
    "black_male": "Black male",
    "black_female": "Black female",
}


def setup_paper_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 6.2,
        "axes.titlesize": 6.4,
        "axes.labelsize": 6.1,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.linewidth": 0.3,
        "xtick.major.width": 0.3,
        "ytick.major.width": 0.3,
        "xtick.major.size": 1.4,
        "ytick.major.size": 1.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(path.with_suffix(ext))
    plt.close(fig)


def _subgroup_order(dataset: str) -> list[str]:
    return MULTIRACE_ORDER if dataset == "multirace" else INTERSECTIONAL_ORDER


def _load_quote_df(dataset: str) -> pd.DataFrame:
    if dataset == "multirace":
        src = _DATASETS / "multirace_with_quotes.csv"
        df = pd.read_csv(src)
        df["subgroup"] = df["race"]
    else:
        src = _DATASETS / "intersectional_with_quotes.csv"
        df = pd.read_csv(src)
        df["subgroup"] = df["race"].astype(str) + "_" + df["gender"].astype(str)
    counts = pd.read_csv(OUT_ROOT / "dolma_counts.csv")
    counts["log10_dolma"] = np.log10(counts["dolma_count"].astype(float) + 1.0)
    df = df.merge(counts[["author_clean", "dolma_count", "log10_dolma"]],
                  on="author_clean", how="left")
    df = df.rename(columns={"log10_hits": "log10_google"})
    return df


def fig_scatter(dataset: str, df: pd.DataFrame, fig_dir: Path) -> None:
    au = df.drop_duplicates("author_clean").dropna(subset=["log10_google",
                                                          "log10_dolma"])
    pr = stats.pearsonr(au["log10_google"], au["log10_dolma"])
    sr = stats.spearmanr(au["log10_google"], au["log10_dolma"])

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    for sg in _subgroup_order(dataset):
        sub = au[au["subgroup"] == sg]
        if sub.empty:
            continue
        ax.scatter(sub["log10_google"], sub["log10_dolma"],
                   s=4, alpha=0.45, edgecolor="none",
                   color=SUBGROUP_COLOR.get(sg, "#7F7F7F"),
                   label=f"{SUBGROUP_LABEL.get(sg, sg)} (n={len(sub)})")

    lo = min(au["log10_google"].min(), au["log10_dolma"].min())
    hi = max(au["log10_google"].max(), au["log10_dolma"].max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.4,
            linestyle="--", alpha=0.6)

    ax.set_xlabel(r"$\log_{10}$ Google hits")
    ax.set_ylabel(r"$\log_{10}$(Dolma count + 1)")
    ax.set_title(f"{dataset.title()}: Dolma vs Google fame  "
                 f"(n = {len(au):,} authors;  "
                 f"Pearson r = {pr.statistic:.2f},  "
                 f"Spearman $\\rho$ = {sr.statistic:.2f})")

    ax.tick_params(length=1.4, width=0.3)
    ax.grid(True, linewidth=0.2, alpha=0.45)
    leg = ax.legend(frameon=False, loc="lower right", handlelength=0.9,
                    handletextpad=0.4, borderaxespad=0.2)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
        handle.set_sizes([10])
    fig.tight_layout(pad=0.3)
    _save(fig, fig_dir / "scatter_dolma_vs_google")


def fig_scatter_with_subgroup_fits(dataset: str, df: pd.DataFrame,
                                   fig_dir: Path) -> None:
    """Scatter with per-subgroup OLS fits overlaid. Shows whether the
    Google->Dolma relationship has a common slope/intercept across subgroups."""
    au = df.drop_duplicates("author_clean").dropna(subset=["log10_google",
                                                          "log10_dolma"])
    pr = stats.pearsonr(au["log10_google"], au["log10_dolma"])

    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    # Reference y=x line.
    lo = float(min(au["log10_google"].min(), au["log10_dolma"].min()))
    hi = float(max(au["log10_google"].max(), au["log10_dolma"].max()))
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.4, linestyle="--",
            alpha=0.55, zorder=1)

    handles = []
    labels = []
    for sg in _subgroup_order(dataset):
        sub = au[au["subgroup"] == sg]
        if sub.empty:
            continue
        color = SUBGROUP_COLOR.get(sg, "#7F7F7F")
        ax.scatter(sub["log10_google"], sub["log10_dolma"],
                   s=2.8, alpha=0.25, edgecolor="none", color=color, zorder=2)
        # Per-subgroup OLS fit on log scale.
        slope, intercept, r, _, _ = stats.linregress(sub["log10_google"],
                                                    sub["log10_dolma"])
        x = np.linspace(sub["log10_google"].min(), sub["log10_google"].max(), 50)
        line, = ax.plot(x, slope * x + intercept, color=color, lw=1.1,
                        alpha=0.95, zorder=3)
        handles.append(line)
        labels.append(f"{SUBGROUP_LABEL.get(sg, sg)}  "
                      f"(slope={slope:.2f}, $r$={r:.2f}, n={len(sub)})")

    ax.set_xlabel(r"$\log_{10}$ Google hits")
    ax.set_ylabel(r"$\log_{10}$(Dolma count + 1)")
    ax.set_title(f"{dataset.title()}: per-subgroup Dolma vs Google fits  "
                 f"(overall Pearson r = {pr.statistic:.2f})")
    ax.tick_params(length=1.4, width=0.3)
    ax.grid(True, linewidth=0.2, alpha=0.45)
    ax.legend(handles, labels, frameon=False, loc="lower right",
              handlelength=1.4, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    _save(fig, fig_dir / "subgroup_regression_fits")


def fig_subgroup_violin(dataset: str, df: pd.DataFrame, fig_dir: Path) -> None:
    """Side-by-side violins per subgroup, for Google vs Dolma."""
    subgroups = _subgroup_order(dataset)
    n = len(subgroups)

    google_data = [df.loc[df["subgroup"] == sg, "log10_google"].dropna().values
                   for sg in subgroups]
    dolma_data = [df.loc[df["subgroup"] == sg, "log10_dolma"].dropna().values
                  for sg in subgroups]

    fig, ax = plt.subplots(figsize=(0.85 * n + 1.4, 2.6))
    width = 0.36
    positions_g = np.arange(n) - width / 2 - 0.02
    positions_d = np.arange(n) + width / 2 + 0.02

    def _violin(data: list, positions: np.ndarray, facecolor: str) -> None:
        parts = ax.violinplot(data, positions=positions, widths=width,
                              showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(facecolor)
            body.set_edgecolor("black")
            body.set_alpha(0.75)
            body.set_linewidth(0.4)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(0.6)

    _violin(google_data, positions_g, "#B0B0B0")  # Google: light grey
    _violin(dolma_data, positions_d, "#6E6E6E")   # Dolma: dark grey

    # Subgroup means as small marker on top of each violin.
    for i, sg in enumerate(subgroups):
        mg = np.mean(google_data[i]) if len(google_data[i]) else np.nan
        md = np.mean(dolma_data[i]) if len(dolma_data[i]) else np.nan
        ax.plot(positions_g[i], mg, marker="o", ms=2.5,
                color=SUBGROUP_COLOR.get(sg, "black"), zorder=5)
        ax.plot(positions_d[i], md, marker="o", ms=2.5,
                color=SUBGROUP_COLOR.get(sg, "black"), zorder=5)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([SUBGROUP_LABEL.get(sg, sg) for sg in subgroups],
                       rotation=0)
    ax.set_ylabel(r"$\log_{10}$ fame")
    ax.set_title(f"{dataset.title()}: fame distributions per subgroup  "
                 f"— Google (light) vs Dolma (dark)")
    ax.tick_params(length=1.4, width=0.3)
    ax.grid(axis="y", linewidth=0.2, alpha=0.45)

    # Custom legend (two greys).
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#B0B0B0", edgecolor="black", linewidth=0.4,
              label="Google hits"),
        Patch(facecolor="#6E6E6E", edgecolor="black", linewidth=0.4,
              label="Dolma count"),
    ]
    ax.legend(handles=legend_handles, frameon=False,
              loc="upper right", handlelength=0.9, handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    _save(fig, fig_dir / "subgroup_distribution")


def fig_decile_crosstab(dataset: str, dataset_out_dir: Path,
                        fig_dir: Path) -> None:
    crosstab = pd.read_csv(dataset_out_dir / "decile_crosstab.csv",
                           index_col=0)
    crosstab.index = crosstab.index.astype(int)
    crosstab.columns = crosstab.columns.astype(int)
    crosstab = crosstab.reindex(index=range(1, 11),
                                columns=range(1, 11), fill_value=0)
    mat = crosstab.values.astype(float)
    # Row-normalize for readability.
    row_norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    im = ax.imshow(row_norm, cmap="Greys", aspect="equal",
                   vmin=0, vmax=row_norm.max())
    for i in range(10):
        for j in range(10):
            v = mat[i, j]
            if v == 0:
                continue
            color = "white" if row_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    fontsize=4.6, color=color)
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(1, 11))
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(1, 11))
    ax.set_xlabel("Dolma decile")
    ax.set_ylabel("Google decile")
    ax.set_title(f"{dataset.title()}: Google decile $\\rightarrow$ Dolma decile  "
                 f"(row-normalized)")
    ax.tick_params(length=1.4, width=0.3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(length=1.0, width=0.3, labelsize=5.0)
    cbar.outline.set_linewidth(0.3)
    fig.tight_layout(pad=0.3)
    _save(fig, fig_dir / "decile_crosstab")


def run_for_dataset(dataset: str) -> None:
    dataset_out_dir = OUT_ROOT / dataset
    fig_dir = dataset_out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = _load_quote_df(dataset)
    fig_scatter(dataset, df, fig_dir)
    fig_scatter_with_subgroup_fits(dataset, df, fig_dir)
    fig_subgroup_violin(dataset, df, fig_dir)
    fig_decile_crosstab(dataset, dataset_out_dir, fig_dir)
    print(f"[{dataset}] wrote figures to {fig_dir}/")


def main() -> None:
    setup_paper_style()
    for d in ("multirace", "intersectional"):
        run_for_dataset(d)


if __name__ == "__main__":
    main()
