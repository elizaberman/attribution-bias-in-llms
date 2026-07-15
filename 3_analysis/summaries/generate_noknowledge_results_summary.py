#!/usr/bin/env python3
"""Aggregate the no-evidence runs into `noknowledge_results_summary.md`.

Step 1 of the analysis pipeline, for the no-evidence (zero-shot) setting. Parses the
`analysis/summary.txt` report that `analyze_outputs.py` writes for each run and
combines them into one markdown summary: overall accuracy by prompt type, subgroup
breakdowns, and a completion-status table. The RAG twin is
`generate_rag_results_summary.py`.

Reads : $ATTRIBENCH_RESULTS/<run_dir>/.../analysis/summary.txt  (per RUN_SPECS below)
        1_dataset_construction/datasets/*.csv                    (quote counts)
Writes: $ATTRIBENCH_RESULTS/author_presence/noknowledge_results_summary.md

The downstream consumer (`stats/compute_noknowledge_quote_metrics.py`) reads that
markdown from the same location.

IMPORTANT: RUN_SPECS below is a hardcoded manifest of the run directories from the
paper's own execution, including cluster job IDs. It is not portable — edit it to
point at your own run directories. Entries whose `status` is not "done" are reported
as pending and contribute no rows.
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

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(f"{_ATTRIBENCH_ROOT}")
RESULTS = Path(f"{_ATTRIBENCH_RESULTS}")
OUT_MD = RESULTS / "author_presence" / "noknowledge_results_summary.md"

DATASET_PATHS = {
    "intersectional": ROOT / "1_dataset_construction" / "datasets" / "intersectional_with_quotes.csv",
    "multirace": ROOT / "1_dataset_construction" / "datasets" / "multirace_with_quotes.csv",
}

PROMPT_MAP = {
    "DIRECT": "direct",
    "INDIRECT": "indirect",
    "INDIRECT_OVERT": "indirect_overt",
}
PROMPT_ORDER = {"direct": 0, "indirect": 1, "indirect_overt": 2}
INTERSECTIONAL_SUBGROUP_ORDER = {"male_white": 0, "female_white": 1, "male_black": 2, "female_black": 3}
MULTIRACE_SUBGROUP_ORDER = {"white": 0, "latino": 1, "asian": 2, "black": 3}


@dataclass
class RunSpec:
    dataset: str
    model: str
    status: str
    source_csv: Path
    summary_path: Path


RUN_SPECS: List[RunSpec] = [
    RunSpec(
        "intersectional",
        "moonshotai/Kimi-K2.5",
        "running",
        RESULTS / "together_models_full_parallel_reasoning_off_t07_p095/moonshotai_Kimi-K2_5_intersectional_full_runs3.csv",
        RESULTS / "together_models_full_parallel_reasoning_off_t07_p095/analyze_outputs_prepared/moonshotai_Kimi-K2_5_intersectional_full_runs3/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "deepseek-ai/DeepSeek-V3.1",
        "done",
        RESULTS / "deepseek_ai_DeepSeek_V3_1_batch_full/deepseek_ai_DeepSeek_V3_1_experiment_intersectional_full.csv",
        RESULTS / "deepseek_ai_DeepSeek_V3_1_batch_full/analyze_outputs_prepared/deepseek_ai_DeepSeek_V3_1_experiment_intersectional_full/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "done",
        RESULTS / "Qwen_Qwen3_Next_80B_A3B_Instruct_batch_full/Qwen_Qwen3_Next_80B_A3B_Instruct_experiment_intersectional_full.csv",
        RESULTS / "Qwen_Qwen3_Next_80B_A3B_Instruct_batch_full/analyze_outputs_prepared/Qwen_Qwen3_Next_80B_A3B_Instruct_experiment_intersectional_full/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "Qwen/Qwen3.5-397B-A17B",
        "done",
        RESULTS / "Qwen_Qwen3_5_397B_A17B_batch_full/Qwen_Qwen3_5_397B_A17B_experiment_intersectional_full.csv",
        RESULTS / "Qwen_Qwen3_5_397B_A17B_batch_full/analyze_outputs_prepared/Qwen_Qwen3_5_397B_A17B_experiment_intersectional_full/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "zai-org/GLM-5",
        "done",
        RESULTS / "zai_org_GLM_5_batch_full_intersectional/zai_org_GLM_5_experiment_intersectional_full.csv",
        RESULTS / "zai_org_GLM_5_batch_full_intersectional/analyze_outputs_prepared/zai_org_GLM_5_experiment_intersectional_full/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "openai/gpt-oss-120b",
        "done",
        RESULTS / "openai_gpt_oss_120b_batch_full_intersectional/openai_gpt_oss_120b_experiment_intersectional_full.csv",
        RESULTS / "openai_gpt_oss_120b_batch_full_intersectional/analyze_outputs_prepared/openai_gpt_oss_120b_experiment_intersectional_full/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "done",
        RESULTS / "meta_llama_Llama_4_Maverick_17B_128E_Instruct_FP8_batch_full_intersect/meta_llama_Llama_4_Maverick_17B_128E_Instruct_FP8_experiment_intersectional_full.csv",
        RESULTS / "meta_llama_Llama_4_Maverick_17B_128E_Instruct_FP8_batch_full_intersect/analyze_outputs_prepared/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "done",
        RESULTS / "mistralai_Mixtral_8x7B_Instruct_v0_1_batch_full_intersect/mistralai_Mixtral_8x7B_Instruct_v0_1_experiment_intersectional_full.csv",
        RESULTS / "mistralai_Mixtral_8x7B_Instruct_v0_1_batch_full_intersect/analyze_outputs_prepared/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "openai/gpt-5.1",
        "done",
        RESULTS / "gpt5_batch/by_dataset/intersectional_with_quotes/job_4381786_20260316_185748_30006/split_intersectional_with_quotes_20260316_185755/combined/raw_outputs.csv",
        RESULTS / "gpt5_batch/by_dataset/intersectional_with_quotes/job_4381786_20260316_185748_30006/split_intersectional_with_quotes_20260316_185755/combined/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "anthropic/claude-4.6-sonnet",
        "done",
        RESULTS / "claude_batch/job_4769882_20260322_180107_27321/intersectional_with_quotes/raw_outputs.csv",
        RESULTS / "claude_batch/job_4769882_20260322_180107_27321/intersectional_with_quotes/analysis/summary.txt",
    ),
    RunSpec(
        "intersectional",
        "google/gemini-2.5-flash-lite",
        "done",
        RESULTS / "gemini_concurrent/intersectional_analysis/raw_outputs.csv",
        RESULTS / "gemini_concurrent/intersectional_analysis/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "moonshotai/Kimi-K2.5",
        "done",
        RESULTS / "together_models_full_parallel_reasoning_off_t07_p095/moonshotai_Kimi-K2_5_multirace_full_runs3.csv",
        RESULTS / "together_models_full_parallel_reasoning_off_t07_p095/analyze_outputs_prepared/moonshotai_Kimi-K2_5_multirace_full_runs3/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "deepseek-ai/DeepSeek-V3.1",
        "done",
        RESULTS / "deepseek_ai_DeepSeek_V3_1_batch_full/deepseek_ai_DeepSeek_V3_1_experiment_multirace_full.csv",
        RESULTS / "deepseek_ai_DeepSeek_V3_1_batch_full/analyze_outputs_prepared/deepseek_ai_DeepSeek_V3_1_experiment_multirace_full/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "done",
        RESULTS / "Qwen_Qwen3_Next_80B_A3B_Instruct_batch_full/Qwen_Qwen3_Next_80B_A3B_Instruct_experiment_multirace_full.csv",
        RESULTS / "Qwen_Qwen3_Next_80B_A3B_Instruct_batch_full/analyze_outputs_prepared/Qwen_Qwen3_Next_80B_A3B_Instruct_experiment_multirace_full/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "Qwen/Qwen3.5-397B-A17B",
        "done",
        RESULTS / "Qwen_Qwen3_5_397B_A17B_batch_full/Qwen_Qwen3_5_397B_A17B_experiment_multirace_full.csv",
        RESULTS / "Qwen_Qwen3_5_397B_A17B_batch_full/analyze_outputs_prepared/Qwen_Qwen3_5_397B_A17B_experiment_multirace_full/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "zai-org/GLM-5",
        "done",
        RESULTS / "zai_org_GLM_5_batch_full_multirace/zai_org_GLM_5_experiment_multirace_full.csv",
        RESULTS / "zai_org_GLM_5_batch_full_multirace/analyze_outputs_prepared/zai_org_GLM_5_experiment_multirace_full/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "openai/gpt-oss-120b",
        "done",
        RESULTS / "openai_gpt_oss_120b_batch_full_multirace/openai_gpt_oss_120b_experiment_multirace_full.csv",
        RESULTS / "openai_gpt_oss_120b_batch_full_multirace/analyze_outputs_prepared/openai_gpt_oss_120b_experiment_multirace_full/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "openai/gpt-5.1",
        "done",
        RESULTS / "gpt5_batch/by_dataset/multirace_with_quotes/job_4366762_20260316_143754_13850/split_multirace_with_quotes_20260316_143801/combined/raw_outputs.csv",
        RESULTS / "gpt5_batch/by_dataset/multirace_with_quotes/job_4366762_20260316_143754_13850/split_multirace_with_quotes_20260316_143801/combined/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "anthropic/claude-4.6-sonnet",
        "done",
        RESULTS / "claude_batch/job_4770629_20260322_182451_16910/multirace_with_quotes/raw_outputs.csv",
        RESULTS / "claude_batch/job_4770629_20260322_182451_16910/multirace_with_quotes/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "google/gemini-2.5-flash-lite",
        "done",
        RESULTS / "gemini_concurrent/multirace_with_quotes_full_raw_outputs_clean21_resume1.csv",
        RESULTS / "gemini_concurrent/multirace_analysis/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "done",
        RESULTS / "meta_llama_Llama_4_Maverick_17B_128E_Instruct_FP8_batch_full_multirace/meta_llama_Llama_4_Maverick_17B_128E_Instruct_FP8_experiment_multirace_full.csv",
        RESULTS / "meta_llama_Llama_4_Maverick_17B_128E_Instruct_FP8_batch_full_multirace/analyze_outputs_prepared/analysis/summary.txt",
    ),
    RunSpec(
        "multirace",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "done",
        RESULTS / "mistralai_Mixtral_8x7B_Instruct_v0_1_batch_full_multirace/mistralai_Mixtral_8x7B_Instruct_v0_1_experiment_multirace_full.csv",
        RESULTS / "mistralai_Mixtral_8x7B_Instruct_v0_1_batch_full_multirace/analyze_outputs_prepared/analysis/summary.txt",
    ),
]


def count_dataset_quotes(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def norm_intersectional_subgroup(label: str) -> str:
    s = label.strip().lower().replace("-", "_")
    parts = s.split("_")
    if len(parts) == 2 and parts[0] in {"male", "female", "unknown"}:
        return f"{parts[0]}_{parts[1]}"
    if len(parts) == 2 and parts[1] in {"male", "female", "unknown"}:
        return f"{parts[1]}_{parts[0]}"
    return s


def parse_summary(summary_path: Path, dataset: str) -> Tuple[List[Dict], List[Dict]]:
    lines = summary_path.read_text(errors="ignore").splitlines()
    section = None
    cur_prompt = None
    in_demo_by_race = False

    overall_rows: List[Dict] = []
    inter_correct: Dict[Tuple[str, str], float] = {}
    inter_no: Dict[Tuple[str, str], float] = {}
    race_correct: Dict[Tuple[str, str], float] = {}
    race_no: Dict[Tuple[str, str], float] = {}

    for line in lines:
        stripped = line.strip()
        if "OVERALL ACCURACY BY PROMPT TYPE (THREE-WAY BREAKDOWN)" in line:
            section = "overall"
            cur_prompt = None
            continue
        if "CONDITIONAL (GIVEN DIRECT CORRECT)" in line:
            # Do not mix conditional rates into overall metrics.
            section = "conditional"
            cur_prompt = None
            continue
        if "INTERSECTIONAL ANALYSIS (Gender × Race)" in line:
            section = "inter"
            cur_prompt = None
            continue
        if "NO AUTHOR MENTIONED BY SUBGROUP (RACE x GENDER)" in line:
            section = "inter_no"
            cur_prompt = None
            continue
        if "ACCURACY BY DEMOGRAPHIC GROUP" in line:
            section = "demo"
            cur_prompt = None
            in_demo_by_race = False
            continue
        if "NO AUTHOR MENTIONED BY RACE" in line:
            section = "race_no"
            cur_prompt = None
            continue
        if stripped.startswith("="):
            continue

        m_prompt = re.match(r"^(DIRECT|INDIRECT|INDIRECT_OVERT):$", stripped)
        if m_prompt:
            cur_prompt = PROMPT_MAP[m_prompt.group(1)]
            continue

        m_demo = re.match(r"^(DIRECT|INDIRECT|INDIRECT_OVERT) - By (Gender|Race):$", stripped)
        if section == "demo" and m_demo:
            cur_prompt = PROMPT_MAP[m_demo.group(1)]
            in_demo_by_race = m_demo.group(2) == "Race"
            continue

        if section == "overall" and cur_prompt:
            m1 = re.search(r"Correct Author Identified:\s+([0-9.]+)%", line)
            m2 = re.search(r"Wrong Author Mentioned:\s+([0-9.]+)%", line)
            m3 = re.search(r"No Author Mentioned:\s+([0-9.]+)%", line)
            if m1:
                overall_rows.append({"prompt": cur_prompt, "metric": "correct_pct", "value": float(m1.group(1))})
            if m2:
                overall_rows.append({"prompt": cur_prompt, "metric": "wrong_pct", "value": float(m2.group(1))})
            if m3:
                overall_rows.append({"prompt": cur_prompt, "metric": "no_author_pct", "value": float(m3.group(1))})

        if section == "inter" and cur_prompt:
            m = re.match(r"^\s{2}([A-Za-z_]+)\s+([0-9.]+)% \(n=\d+\)", line)
            if m:
                sg = norm_intersectional_subgroup(m.group(1))
                inter_correct[(cur_prompt, sg)] = float(m.group(2))

        if section == "inter_no" and cur_prompt:
            m = re.match(r"^\s{2}([A-Za-z_]+)\s+\d+/\d+ \(([0-9.]+)%\)", line)
            if m:
                sg = norm_intersectional_subgroup(m.group(1))
                inter_no[(cur_prompt, sg)] = float(m.group(2))

        if section == "demo" and cur_prompt and in_demo_by_race:
            m = re.match(r"^\s{2}([A-Za-z_]+):\s+([0-9.]+)% \(n=\d+\)", line)
            if m:
                sg = m.group(1).strip().lower().replace("-", "_")
                race_correct[(cur_prompt, sg)] = float(m.group(2))

        if section == "race_no" and cur_prompt:
            m = re.match(r"^\s{2}([A-Za-z_]+)\s+\d+/\d+ \(([0-9.]+)%\)", line)
            if m:
                sg = m.group(1).strip().lower().replace("-", "_")
                race_no[(cur_prompt, sg)] = float(m.group(2))

    overall_map: Dict[str, Dict[str, float]] = {}
    for row in overall_rows:
        overall_map.setdefault(row["prompt"], {})[row["metric"]] = row["value"]
    overall = [
        {"prompt": p, "correct_pct": v.get("correct_pct"), "wrong_pct": v.get("wrong_pct"), "no_author_pct": v.get("no_author_pct")}
        for p, v in overall_map.items()
    ]

    subgroup: List[Dict] = []
    if dataset == "intersectional" and (inter_correct or inter_no):
        keys = sorted(set(inter_correct.keys()) | set(inter_no.keys()))
        for prompt, sg in keys:
            c = inter_correct.get((prompt, sg))
            n = inter_no.get((prompt, sg))
            w = round(100.0 - c - n, 3) if c is not None and n is not None else None
            subgroup.append({"prompt": prompt, "subgroup": sg, "correct_pct": c, "wrong_pct": w, "no_author_pct": n})
    elif dataset == "multirace" and (race_correct or race_no):
        keys = sorted(set(race_correct.keys()) | set(race_no.keys()))
        for prompt, sg in keys:
            c = race_correct.get((prompt, sg))
            n = race_no.get((prompt, sg))
            w = round(100.0 - c - n, 3) if c is not None and n is not None else None
            subgroup.append({"prompt": prompt, "subgroup": sg, "correct_pct": c, "wrong_pct": w, "no_author_pct": n})
    return overall, subgroup


def count_complete_units(source_csv: Path) -> int:
    best: Dict[Tuple[str, str], int] = {}
    long_seen: Dict[Tuple[str, str], set] = {}
    raw = source_csv.read_bytes().replace(b"\x00", b"")
    text = io.StringIO(raw.decode("utf-8", errors="replace"))
    reader = csv.DictReader(text)
    fieldnames = set(reader.fieldnames or [])
    is_long_format = {"prompt_type", "llm_output"}.issubset(fieldnames)

    for row in reader:
        qid = str(row.get("quote_id", "")).strip()
        run = str(row.get("run", "")).strip() or "1"
        if not qid:
            continue
        key = (qid, run)
        if is_long_format:
            prompt = str(row.get("prompt_type", "")).strip().lower()
            if prompt in {"direct", "indirect", "indirect_overt"}:
                long_seen.setdefault(key, set()).add(prompt)
        else:
            score = 0
            if str(row.get("direct_response", "")).strip():
                score += 1
            if str(row.get("indirect_response", "")).strip():
                score += 1
            if str(row.get("indirect_overt_response", "")).strip():
                score += 1
            best[key] = max(best.get(key, 0), score)

    if is_long_format:
        return sum(1 for seen in long_seen.values() if len(seen) == 3)
    return sum(1 for s in best.values() if s == 3)


def fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return ""
    if abs(v) < 0.05:
        v = 0.0
    return f"{v:.1f}%"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    dataset_sizes = {k: count_dataset_quotes(v) for k, v in DATASET_PATHS.items()}

    rows_status = []
    rows_overall_inter = []
    rows_subgroup_inter = []
    rows_overall_multi = []
    rows_subgroup_multi = []
    notes = []

    for spec in RUN_SPECS:
        expected = dataset_sizes[spec.dataset] * 3
        complete = count_complete_units(spec.source_csv) if spec.source_csv.exists() else 0
        pending = max(expected - complete, 0)

        status = spec.status
        if spec.summary_path.exists():
            status = "done"
        elif status == "done":
            status = "running"

        rows_status.append(
            {
                "dataset": spec.dataset,
                "model": spec.model,
                "status": status,
                "pending": pending,
                "complete": complete,
                "expected": expected,
                "source_csv": str(spec.source_csv) if spec.source_csv.exists() else "",
                "summary": str(spec.summary_path) if spec.summary_path.exists() else "",
            }
        )

        if status != "done" or not spec.summary_path.exists():
            notes.append(f"`{spec.model}` for **{spec.dataset}** is marked running; no prompt-rate table row is included for it.")
            continue

        overall, subgroup = parse_summary(spec.summary_path, spec.dataset)
        for o in overall:
            row = {"model": spec.model, **o}
            if spec.dataset == "intersectional":
                rows_overall_inter.append(row)
            else:
                rows_overall_multi.append(row)
        for s in subgroup:
            row = {"model": spec.model, **s}
            if spec.dataset == "intersectional":
                rows_subgroup_inter.append(row)
            else:
                rows_subgroup_multi.append(row)

    rows_overall_inter.sort(key=lambda r: (r["model"], PROMPT_ORDER.get(r["prompt"], 99)))
    rows_overall_multi.sort(key=lambda r: (r["model"], PROMPT_ORDER.get(r["prompt"], 99)))
    rows_subgroup_inter.sort(key=lambda r: (r["model"], PROMPT_ORDER.get(r["prompt"], 99), INTERSECTIONAL_SUBGROUP_ORDER.get(r["subgroup"], 99), r["subgroup"]))
    rows_subgroup_multi.sort(key=lambda r: (r["model"], PROMPT_ORDER.get(r["prompt"], 99), MULTIRACE_SUBGROUP_ORDER.get(r["subgroup"], 99), r["subgroup"]))

    lines: List[str] = []
    lines.append("# NOKNOWLEDGE Results Summary")
    lines.append("")
    lines.append("Source: `summary.txt` from each model/dataset prepared analysis directory.")
    lines.append("")
    lines.append("## Missing Rows / Completion Status")
    lines.append("")
    lines.append(
        md_table(
            ["Dataset", "Model", "Status", "Pending Units", "Complete/Expected", "Source CSV", "Summary"],
            [
                [
                    r["dataset"],
                    r["model"],
                    r["status"],
                    str(r["pending"]),
                    f'{r["complete"]}/{r["expected"]}',
                    f'`{r["source_csv"]}`' if r["source_csv"] else "",
                    f'`{r["summary"]}`' if r["summary"] else "",
                ]
                for r in rows_status
            ],
        )
    )
    lines.append("")
    lines.append("## Intersectional Overall (Per Prompt, Per Model)")
    lines.append("")
    lines.append(
        md_table(
            ["Model", "Prompt", "% Correct", "% Wrong", "% No Author"],
            [[r["model"], r["prompt"], fmt_pct(r["correct_pct"]), fmt_pct(r["wrong_pct"]), fmt_pct(r["no_author_pct"])] for r in rows_overall_inter],
        )
    )
    lines.append("")
    lines.append("## Intersectional Subgroup Split (Per Prompt, Per Model)")
    lines.append("")
    lines.append(
        md_table(
            ["Model", "Prompt", "Subgroup", "% Correct", "% Wrong", "% No Author"],
            [[r["model"], r["prompt"], r["subgroup"], fmt_pct(r["correct_pct"]), fmt_pct(r["wrong_pct"]), fmt_pct(r["no_author_pct"])] for r in rows_subgroup_inter],
        )
    )
    lines.append("")
    lines.append("## Multirace Overall (Per Prompt, Per Model)")
    lines.append("")
    lines.append(
        md_table(
            ["Model", "Prompt", "% Correct", "% Wrong", "% No Author"],
            [[r["model"], r["prompt"], fmt_pct(r["correct_pct"]), fmt_pct(r["wrong_pct"]), fmt_pct(r["no_author_pct"])] for r in rows_overall_multi],
        )
    )
    lines.append("")
    lines.append("## Multirace Subgroup Split (Per Prompt, Per Model)")
    lines.append("")
    lines.append(
        md_table(
            ["Model", "Prompt", "Subgroup", "% Correct", "% Wrong", "% No Author"],
            [[r["model"], r["prompt"], r["subgroup"], fmt_pct(r["correct_pct"]), fmt_pct(r["wrong_pct"]), fmt_pct(r["no_author_pct"])] for r in rows_subgroup_multi],
        )
    )
    lines.append("")
    lines.append("## Notes")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("- Subgroup `% Wrong` is computed as `100 - %Correct - %NoAuthor` using subgroup rates parsed from summary sections.")
    lines.append("- For multirace subgroup split, `%Correct` comes from `ACCURACY BY DEMOGRAPHIC GROUP` (`DIRECT/INDIRECT/INDIRECT_OVERT - By Race`) and `%NoAuthor` comes from `NO AUTHOR MENTIONED BY RACE` in each summary.")
    lines.append("- For intersectional subgroup split, `%Correct` comes from `INTERSECTIONAL ANALYSIS (Gender × Race)` and `%NoAuthor` comes from `NO AUTHOR MENTIONED BY SUBGROUP (RACE x GENDER)` in each summary.")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
import io
