"""
Compute quote-level accuracy and suppression statistics from no-knowledge results.

This script:
- Parses `noknowledge_results_summary.md` to discover model -> raw results CSV mappings.
- Loads each raw results CSV and computes row-level accuracy and suppression indicators
  using the project's existing author-matching logic (adapted from
  `3_analysis/summaries/analyze_outputs.py`).
- Collapses 3 runs per (model, quote_id, prompt_type) into quote-level means.
- Computes overall and subgroup mean/variance statistics across quotes.

Expected inputs:
- A markdown summary file listing models and raw results CSV paths.
- Raw results CSVs with columns:
  `quote_id, prompt_type, run, author, gender, race_ethnicity, llm_output`.

Outputs written (CSV):
1. `quote_level_metrics.csv`
2. `overall_metrics.csv`
3. `subgroup_metrics.csv`
4. `model_file_map.csv`

`DATASET_MODE` controls subgrouping:
- "multirace" -> subgroup by `race_ethnicity`
- "intersectional" -> subgroup by (`race_ethnicity`, `gender`)
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
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# -------------------------
# User-settable parameters
# -------------------------
DATASET_MODE = "multirace"  # or "intersectional"

DEFAULT_SUMMARY_MD = Path(
    f"{_ATTRIBENCH_RESULTS}/author_presence/noknowledge_results_summary.md"
)
DEFAULT_RAG_SUMMARY_MD = Path(
    f"{_ATTRIBENCH_RESULTS}/author_presence/rag_results_summary.md"
)
DEFAULT_OUTDIR = Path(f"{_ATTRIBENCH_RESULTS}/quote_metrics")
MATCHING_DATASETS_DIR = Path(f"{_ATTRIBENCH_ROOT}/1_dataset_construction/datasets")
CACHE_VERSION = "v4"

# -------------------------
# Author matching logic (imported from ../summaries/analyze_outputs.py)
# -------------------------
_SUMMARIES = Path(__file__).resolve().parents[1] / "summaries"
sys.path.insert(0, str(_SUMMARIES))

try:
    from analyze_outputs import (  # type: ignore
        AuthorMatcher,
        canonicalize_author,
        extract_last_name,
        is_non_person_author,
        mentions_any_name,
        normalize_for_match,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import author-matching logic from 3_analysis/summaries/analyze_outputs.py"
    ) from exc


def safe_normalize(text) -> str:
    if not isinstance(text, str):
        return ""
    return normalize_for_match(text)


# -------------------------
# Parsing and processing
# -------------------------

@dataclass(frozen=True)
class ModelFile:
    dataset: str
    model: str
    source_csv: Path
    summary_md: Path
    is_local: bool


def _extract_csv_path(text: str) -> Optional[Path]:
    if not text:
        return None
    m = re.search(r"`([^`]*\.csv)`", text)
    if m:
        return Path(m.group(1))
    m = re.search(r"(/[^\s`]*\.csv)", text)
    if m:
        return Path(m.group(1))
    return None


def _row_mentions_local(text: str) -> bool:
    if not text:
        return False
    return re.search(r"\blocal\b", text, flags=re.IGNORECASE) is not None


def parse_summary_md(summary_md: Path) -> List[ModelFile]:
    if not summary_md.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_md}")
    lines = summary_md.read_text().splitlines()

    rows: List[ModelFile] = []
    header: Optional[List[str]] = None
    for line in lines:
        if line.strip().startswith("|") and "Source CSV" in line and "Model" in line:
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            continue
        if header and line.strip().startswith("|---"):
            continue
        if header and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != len(header):
                continue
            row = dict(zip(header, cells))
            src_cell = row.get("Source CSV") or ""
            src_path = _extract_csv_path(src_cell) or _extract_csv_path(line)
            if not src_path:
                continue
            dataset = (row.get("Dataset") or "unknown").strip()
            model = (row.get("Model") or "unknown").strip()
            row_text = " | ".join(cells)
            is_local = _row_mentions_local(row_text)
            rows.append(
                ModelFile(
                    dataset=dataset,
                    model=model,
                    source_csv=src_path,
                    summary_md=summary_md,
                    is_local=is_local,
                )
            )

    # Deduplicate while preserving order
    seen = set()
    unique_rows: List[ModelFile] = []
    for r in rows:
        key = (r.dataset, r.model, str(r.source_csv))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(r)
    return unique_rows


_DATASET_META_CACHE: dict[str, pd.DataFrame] = {}


def infer_dataset_key(dataset: str, source_csv: Path) -> Optional[str]:
    dataset = (dataset or "").strip().lower()
    if dataset in {"intersectional", "multirace"}:
        return dataset
    lower_path = str(source_csv).lower()
    if "intersectional" in lower_path:
        return "intersectional"
    if "multirace" in lower_path:
        return "multirace"
    return None


def load_dataset_metadata(dataset_key: str) -> pd.DataFrame:
    if dataset_key in _DATASET_META_CACHE:
        return _DATASET_META_CACHE[dataset_key]
    path = MATCHING_DATASETS_DIR / f"{dataset_key}_with_quotes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset metadata: {path}")
    meta = pd.read_csv(path)
    # Some files have an unnamed index column
    if "" in meta.columns:
        meta = meta.drop(columns=[""])
    meta = meta.rename(columns={"race": "race_ethnicity"})
    _DATASET_META_CACHE[dataset_key] = meta
    return meta


def normalize_input_df(df: pd.DataFrame, dataset: str, source_csv: Path) -> pd.DataFrame:
    df = df.copy()

    # Convert wide format (direct/indirect/indirect_overt columns) to long format.
    if "prompt_type" not in df.columns:
        response_map = {}
        for p in ["direct", "indirect", "indirect_overt"]:
            col = f"{p}_response"
            if col in df.columns:
                response_map[p] = col
        if response_map:
            base_cols = [c for c in ["quote_id", "run", "model"] if c in df.columns]
            author_col = "author" if "author" in df.columns else (
                "author_ground_truth" if "author_ground_truth" in df.columns else None
            )
            if author_col:
                base_cols.append(author_col)
            long_parts = []
            for prompt, resp_col in response_map.items():
                cols = base_cols + [resp_col]
                part = df[cols].copy()
                rename_map = {resp_col: "llm_output"}
                if author_col:
                    rename_map[author_col] = "author"
                part = part.rename(columns=rename_map)
                part["prompt_type"] = prompt
                long_parts.append(part)
            df = pd.concat(long_parts, ignore_index=True)

    if "author" not in df.columns and "author_ground_truth" in df.columns:
        df["author"] = df["author_ground_truth"]

    # Bring in demographic metadata if missing.
    if "gender" not in df.columns or "race_ethnicity" not in df.columns:
        dataset_key = infer_dataset_key(dataset, source_csv)
        if dataset_key:
            meta = load_dataset_metadata(dataset_key)
            keep_cols = ["quote_id", "author_clean", "gender", "race_ethnicity"]
            meta = meta[keep_cols]
            df = df.merge(meta, on="quote_id", how="left")
            if "author" in df.columns:
                df["author"] = df["author"].fillna(df["author_clean"])
            else:
                df["author"] = df["author_clean"]
            df = df.drop(columns=["author_clean"], errors="ignore")

    return df


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", text.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


def build_cache_key(source_csv: Path, dataset: str, model: str) -> str:
    stat = source_csv.stat()
    payload = {
        "path": str(source_csv),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "dataset": dataset,
        "model": model,
        "cache_version": CACHE_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def load_cached_rows(cache_path: Path) -> Optional[pd.DataFrame]:
    if not cache_path.exists():
        return None
    return pd.read_csv(cache_path, low_memory=False)


def save_cached_rows(df: pd.DataFrame, cache_path: Path, meta_path: Path, meta: dict) -> None:
    df.to_csv(cache_path, index=False)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def per_quote_cache_paths(base_dir: Path, dataset: str, model: str, key: str) -> tuple[Path, Path]:
    # If base_dir is already dataset-specific, don't nest again.
    if base_dir.name in {"intersectional", "multirace", "intersectional_rag", "multirace_rag"}:
        dataset_dir = base_dir / "cached_per_quote"
    else:
        dataset_dir = base_dir / dataset / "cached_per_quote"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_slug = slugify(f"{dataset}__{model}")
    cache_path = dataset_dir / f"{cache_slug}__{key}.csv"
    meta_path = dataset_dir / f"{cache_slug}__{key}.json"
    return cache_path, meta_path


def compute_row_indicators(
    df: pd.DataFrame, model_name: str, dataset: str, source_csv: Path, rag_mode: bool
) -> pd.DataFrame:
    df = normalize_input_df(df, dataset, source_csv)

    # Ensure required columns exist
    for col in ["quote_id", "prompt_type", "run", "author", "gender", "race_ethnicity", "llm_output"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "model" not in df.columns:
        df["model"] = model_name
    else:
        df["model"] = df["model"].fillna(model_name)
        df.loc[df["model"].astype(str).str.strip() == "", "model"] = model_name

    if "dataset" not in df.columns:
        df["dataset"] = dataset

    # Normalize demographics
    for demo_col in ["gender", "race_ethnicity"]:
        df[demo_col] = (
            df[demo_col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
        )

    df["author_canonical"] = df["author"].apply(canonicalize_author)
    df["author_norm"] = df["author_canonical"].apply(safe_normalize)
    df["author_last_name"] = df["author_canonical"].apply(extract_last_name)
    df["norm_output"] = df["llm_output"].apply(safe_normalize)

    all_authors = [
        a for a in pd.unique(df["author_canonical"])
        if isinstance(a, str) and not is_non_person_author(a)
    ]
    matcher = AuthorMatcher(all_authors)

    norm_outputs = df["norm_output"].tolist()
    norm_authors = df["author_norm"].tolist()
    last_names = df["author_last_name"].tolist()
    raw_outputs = df["llm_output"].tolist()

    identified = []
    mentioned_any = []

    for norm_output, norm_author, last_name, raw_output in zip(
        norm_outputs, norm_authors, last_names, raw_outputs
    ):
        identified_correct = matcher.check_author_match(norm_output, norm_author, last_name)
        mentioned_author = matcher.extract_mentioned_author(norm_output)
        mentioned_any_author = bool(mentioned_author) or mentions_any_name(raw_output)
        identified.append(1 if identified_correct else 0)
        mentioned_any.append(1 if mentioned_any_author else 0)

    df["accuracy_row"] = identified
    if rag_mode:
        # RAG suppression = wrong OR no author (i.e., not correct).
        df["suppression_row"] = [1 - v for v in identified]
    else:
        # No-knowledge suppression = no author mentioned.
        df["suppression_row"] = [1 - v for v in mentioned_any]

    return df


def aggregate_quote_level(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "model", "quote_id", "prompt_type"]
    agg = df.groupby(group_cols, dropna=False).agg(
        author=("author", "first"),
        gender=("gender", "first"),
        race_ethnicity=("race_ethnicity", "first"),
        accuracy_quote=("accuracy_row", "mean"),
        suppression_quote=("suppression_row", "mean"),
    )
    return agg.reset_index()


def compute_summary_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)

    acc = grouped["accuracy_quote"].agg(["count", "mean", "var", "std"]).reset_index()
    acc = acc.rename(
        columns={
            "count": "n_quotes",
            "mean": "mean_accuracy",
            "var": "variance_accuracy",
            "std": "std_accuracy",
        }
    )
    acc["se_accuracy"] = acc["std_accuracy"] / np.sqrt(acc["n_quotes"].replace(0, np.nan))

    sup = grouped["suppression_quote"].agg(["count", "mean", "var", "std"]).reset_index()
    sup = sup.rename(
        columns={
            "count": "n_quotes",
            "mean": "mean_suppression",
            "var": "variance_suppression",
            "std": "std_suppression",
        }
    )
    sup["se_suppression"] = sup["std_suppression"] / np.sqrt(sup["n_quotes"].replace(0, np.nan))

    merged = acc.merge(sup, on=group_cols + ["n_quotes"], how="outer")
    return merged


def build_comparison_tables(overall: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = overall.pivot_table(
        index=["dataset", "model"],
        columns="prompt_type",
        values="mean_accuracy",
    ).reset_index()
    sup = overall.pivot_table(
        index=["dataset", "model"],
        columns="prompt_type",
        values="mean_suppression",
    ).reset_index()
    return acc, sup


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute quote-level accuracy and suppression metrics from NOKNOWLEDGE results."
    )
    parser.add_argument(
        "--summary_md",
        type=Path,
        default=DEFAULT_SUMMARY_MD,
        help="Path to noknowledge_results_summary.md",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default=DATASET_MODE,
        choices=["multirace", "intersectional"],
        help="Subgrouping mode",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Run the RAG pipeline (uses rag_results_summary.md and writes to *_rag folders by default)",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Optional cache directory for per-model processed rows",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="Disable cache and recompute from raw CSVs",
    )
    args = parser.parse_args()

    summary_md = args.summary_md
    if args.rag and args.summary_md == DEFAULT_SUMMARY_MD:
        summary_md = DEFAULT_RAG_SUMMARY_MD
    if not summary_md.exists() and summary_md.name == "no_knowledge_results_summary.md":
        alt = summary_md.with_name("noknowledge_results_summary.md")
        if alt.exists():
            summary_md = alt

    model_files = parse_summary_md(summary_md)
    if not model_files:
        print(f"No model entries found in summary: {summary_md}", file=sys.stderr)
        sys.exit(1)

    if args.rag:
        print("RAG mode enabled.")

    outdir = args.outdir
    if outdir == DEFAULT_OUTDIR:
        suffix = f"{args.dataset_mode}_rag" if args.rag else args.dataset_mode
        outdir = DEFAULT_OUTDIR / suffix
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (outdir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Discovered model -> file mapping:")
    for mf in model_files:
        print(f"- {mf.dataset} | {mf.model} -> {mf.source_csv}")

    model_map_df = pd.DataFrame(
        [
            {
                "dataset": mf.dataset,
                "model": mf.model,
                "source_csv": str(mf.source_csv),
                "summary_md": str(mf.summary_md),
                "is_local": mf.is_local,
            }
            for mf in model_files
        ]
    )
    # Always write the full discovered map for traceability.
    model_map_df.to_csv(outdir / "model_file_map_all.csv", index=False)

    usable_models = [mf for mf in model_files if not mf.is_local]
    skipped_local = [mf for mf in model_files if mf.is_local]
    if skipped_local:
        print("Skipping local models designated in the markdown:")
        for mf in skipped_local:
            print(f"- {mf.dataset} | {mf.model} -> {mf.source_csv}")

    # Apply dataset filter
    dataset_filter = args.dataset_mode
    if dataset_filter in {"intersectional", "multirace"}:
        usable_models = [mf for mf in usable_models if mf.dataset == dataset_filter]

    if not usable_models:
        print("No non-local models found after filtering; exiting.", file=sys.stderr)
        sys.exit(1)

    # Write the filtered map used for this run.
    filtered_map_df = pd.DataFrame(
        [
            {
                "dataset": mf.dataset,
                "model": mf.model,
                "source_csv": str(mf.source_csv),
                "summary_md": str(mf.summary_md),
                "is_local": mf.is_local,
            }
            for mf in usable_models
        ]
    )
    filtered_map_df.to_csv(outdir / "model_file_map.csv", index=False)

    all_rows: List[pd.DataFrame] = []
    total_models = len(usable_models)
    for idx, mf in enumerate(usable_models, start=1):
        prefix = f"[{idx}/{total_models}]"
        if not mf.source_csv.exists():
            print(f"{prefix} WARNING: missing raw results file for {mf.model}: {mf.source_csv}", file=sys.stderr)
            continue
        cache_key = build_cache_key(mf.source_csv, mf.dataset, mf.model)
        cache_slug = slugify(f"{mf.dataset}__{mf.model}")
        row_cache_path = cache_dir / f"{cache_slug}__{cache_key}.csv"
        row_meta_path = cache_dir / f"{cache_slug}__{cache_key}.json"
        quote_cache_path, quote_meta_path = per_quote_cache_paths(outdir, mf.dataset, mf.model, cache_key)

        if not args.disable_cache:
            cached_quote = load_cached_rows(quote_cache_path)
            if cached_quote is not None:
                print(f"{prefix} cache hit (quote-level): {mf.dataset} | {mf.model}")
                all_rows.append(cached_quote)
                continue

        try:
            print(f"{prefix} reading: {mf.dataset} | {mf.model}")
            df = pd.read_csv(mf.source_csv, low_memory=False)
        except Exception as exc:
            print(f"{prefix} WARNING: failed to read {mf.source_csv}: {exc}", file=sys.stderr)
            try:
                df = pd.read_csv(mf.source_csv, engine="python", on_bad_lines="skip")
                print(f"{prefix} WARNING: re-read with python engine + on_bad_lines=skip: {mf.source_csv}", file=sys.stderr)
            except Exception as exc2:
                print(f"{prefix} WARNING: still failed to read {mf.source_csv}: {exc2}", file=sys.stderr)
                continue

        # Drop exact duplicate rows to be safe
        subset_cols = [c for c in ["quote_id", "prompt_type", "run", "author", "llm_output"] if c in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols)

        try:
            df = compute_row_indicators(df, mf.model, mf.dataset, mf.source_csv, args.rag)
        except Exception as exc:
            print(f"{prefix} WARNING: failed to compute indicators for {mf.source_csv}: {exc}", file=sys.stderr)
            continue

        # Persist row-level cache (optional legacy)
        meta = {
            "dataset": mf.dataset,
            "model": mf.model,
            "source_csv": str(mf.source_csv),
            "cache_version": CACHE_VERSION,
            "cache_level": "row",
        }
        try:
            save_cached_rows(df, row_cache_path, row_meta_path, meta)
        except Exception as exc:
            print(f"{prefix} WARNING: failed to write row cache for {mf.model}: {exc}", file=sys.stderr)

        # Build and persist quote-level cache for fast reruns
        quote_df = aggregate_quote_level(df)
        quote_meta = {
            "dataset": mf.dataset,
            "model": mf.model,
            "source_csv": str(mf.source_csv),
            "cache_version": CACHE_VERSION,
            "cache_level": "quote",
        }
        try:
            save_cached_rows(quote_df, quote_cache_path, quote_meta_path, quote_meta)
        except Exception as exc:
            print(f"{prefix} WARNING: failed to write quote cache for {mf.model}: {exc}", file=sys.stderr)

        all_rows.append(quote_df)

    if not all_rows:
        print("No valid data processed; exiting.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(all_rows, ignore_index=True)

    # If inputs are already quote-level, keep as-is.
    if {"accuracy_quote", "suppression_quote"}.issubset(combined.columns):
        quote_level = combined
    else:
        quote_level = aggregate_quote_level(combined)

    # Safety: enforce dataset filtering on outputs.
    if "dataset" in quote_level.columns:
        before = len(quote_level)
        quote_level = quote_level[quote_level["dataset"] == args.dataset_mode].copy()
        dropped = before - len(quote_level)
        if dropped > 0:
            print(f"WARNING: dropped {dropped} rows not matching dataset_mode={args.dataset_mode}", file=sys.stderr)
    quote_level.to_csv(outdir / "quote_level_metrics.csv", index=False)

    overall = compute_summary_stats(quote_level, ["dataset", "model", "prompt_type"])
    overall.to_csv(outdir / "overall_metrics.csv", index=False)

    subgroup_cols = ["race_ethnicity"] if args.dataset_mode == "multirace" else ["race_ethnicity", "gender"]
    subgroup = compute_summary_stats(quote_level, ["dataset", "model", "prompt_type"] + subgroup_cols)
    subgroup.to_csv(outdir / "subgroup_metrics.csv", index=False)

    acc_table, sup_table = build_comparison_tables(overall)
    acc_table.to_csv(outdir / "mean_accuracy_by_model_prompt.csv", index=False)
    sup_table.to_csv(outdir / "mean_suppression_by_model_prompt.csv", index=False)

    print("\nPreview: model_file_map.csv")
    print(model_map_df.head(10).to_string(index=False))
    print("\nPreview: quote_level_metrics.csv")
    print(quote_level.head(10).to_string(index=False))
    print("\nPreview: overall_metrics.csv")
    print(overall.head(10).to_string(index=False))
    print("\nPreview: subgroup_metrics.csv")
    print(subgroup.head(10).to_string(index=False))
    print("\nPreview: mean_accuracy_by_model_prompt.csv")
    print(acc_table.head(10).to_string(index=False))
    print("\nPreview: mean_suppression_by_model_prompt.csv")
    print(sup_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
