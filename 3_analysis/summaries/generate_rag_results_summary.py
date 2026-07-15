#!/usr/bin/env python3
"""Aggregate the evidence-conditioned (RAG) runs into `rag_results_summary.md`.

Step 1 of the analysis pipeline, for the evidence-conditioned setting. The twin of
`generate_noknowledge_results_summary.py` — same `summary.txt` parsing contract, same
output shape — differing mainly in how run directories are located: RAG runs are
discovered by globbing RAG_RESULTS_BASES for a per-model token, with `fixed_run_dir`
pinning the cases that cannot be globbed.

Reads : $ATTRIBENCH_RESULTS/{rag_together_batch,rag_together_parallel}/...
        $ATTRIBENCH_RESULTS/rag_openai_claude_batch/...   (per RUN_SPECS below)
        1_dataset_construction/datasets/*.csv              (quote counts)
Writes: $ATTRIBENCH_RESULTS/author_presence/rag_results_summary.md

Models marked "(subset)" were run on the 300-matching random subset (1,200 quotes)
rather than the full dataset, due to inference cost; these carry a dagger in the
paper's RAG figures.

IMPORTANT: RUN_SPECS below is a hardcoded manifest of the paper's own run
directories — edit it to point at yours. `parse_summary` here must stay identical to
its counterpart in `generate_noknowledge_results_summary.py`, or the two settings'
tables will silently diverge.
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
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(f"{_ATTRIBENCH_ROOT}")
RESULTS = Path(f"{_ATTRIBENCH_RESULTS}")
RAG_RESULTS_BASES = [
    RESULTS / "rag_together_batch",
    RESULTS / "rag_together_parallel",
]
OUT_MD = RESULTS / "author_presence" / "rag_results_summary.md"

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
    glob_token: str
    fixed_run_dir: Optional[Path] = None


RUN_SPECS: List[RunSpec] = [
    RunSpec("intersectional", "moonshotai/Kimi-K2.5", "running", "moonshotai_Kimi-K2_5"),
    RunSpec("intersectional", "deepseek-ai/DeepSeek-V3.1", "done", "deepseek-ai_DeepSeek-V3_1"),
    RunSpec("intersectional", "Qwen/Qwen3-Next-80B-A3B-Instruct", "done", "Qwen_Qwen3-Next-80B-A3B-Instruct"),
    RunSpec("intersectional", "Qwen/Qwen3.5-397B-A17B", "done", "Qwen_Qwen3_5-397B-A17B"),
    RunSpec("intersectional", "zai-org/GLM-5", "done", "zai-org_GLM-5"),
    RunSpec("intersectional", "openai/gpt-oss-120b", "done", "openai_gpt-oss-120b"),
    RunSpec("intersectional", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "done", "meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8"),
    RunSpec("intersectional", "mistralai/Mixtral-8x7B-Instruct-v0.1", "done", "mistralai_Mixtral-8x7B-Instruct-v0_1"),
    RunSpec(
        "intersectional",
        "openai/gpt-5.1 (subset)",
        "done",
        "",
        RESULTS
        / "rag_openai_claude_batch"
        / "prompt_attribution_rag_openai_gpt-5_1_intersectional_with_quotes_random_300matchings_20260322_203754",
    ),
    RunSpec(
        "intersectional",
        "anthropic/claude-4.6-sonnet (subset)",
        "done",
        "",
        RESULTS
        / "rag_openai_claude_batch"
        / "prompt_attribution_rag_claude_claude-sonnet-4-6_intersectional_with_quotes_random_300matchings_20260322_210031",
    ),
    RunSpec(
        "intersectional",
        "google/gemini-2.5-flash-lite",
        "done",
        "",
        RESULTS
        / "rag_openai_claude_batch"
        / "prompt_attribution_rag_gemini_gemini-2_5-flash-lite_intersectional_with_quotes_20260326_202653",
    ),
    RunSpec("multirace", "moonshotai/Kimi-K2.5", "running", "moonshotai_Kimi-K2_5"),
    RunSpec("multirace", "deepseek-ai/DeepSeek-V3.1", "done", "deepseek-ai_DeepSeek-V3_1"),
    RunSpec("multirace", "Qwen/Qwen3-Next-80B-A3B-Instruct", "done", "Qwen_Qwen3-Next-80B-A3B-Instruct"),
    RunSpec("multirace", "Qwen/Qwen3.5-397B-A17B", "done", "Qwen_Qwen3_5-397B-A17B"),
    RunSpec("multirace", "zai-org/GLM-5", "done", "zai-org_GLM-5"),
    RunSpec("multirace", "openai/gpt-oss-120b", "done", "openai_gpt-oss-120b"),
    RunSpec("multirace", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "done", "meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8"),
    RunSpec("multirace", "mistralai/Mixtral-8x7B-Instruct-v0.1", "running", "mistralai_Mixtral-8x7B-Instruct-v0_1"),
    RunSpec(
        "multirace",
        "openai/gpt-5.1 (subset)",
        "done",
        "",
        RESULTS
        / "rag_openai_claude_batch"
        / "prompt_attribution_rag_openai_gpt-5_1_multirace_with_quotes_random_300matchings_20260322_213040",
    ),
    RunSpec(
        "multirace",
        "anthropic/claude-4.6-sonnet (subset)",
        "done",
        "",
        RESULTS
        / "rag_openai_claude_batch"
        / "prompt_attribution_rag_claude_claude-sonnet-4-6_multirace_with_quotes_random_300matchings_20260322_210844",
    ),
    RunSpec(
        "multirace",
        "google/gemini-2.5-flash-lite",
        "done",
        "",
        RESULTS
        / "rag_openai_claude_batch"
        / "prompt_attribution_rag_gemini_gemini-2_5-flash-lite_multirace_with_quotes_20260327_085126",
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

    overall_rows: List[Dict] = []
    inter_correct: Dict[Tuple[str, str], float] = {}
    inter_no: Dict[Tuple[str, str], float] = {}
    race_correct: Dict[Tuple[str, str], float] = {}
    race_no: Dict[Tuple[str, str], float] = {}
    in_demo_by_race = False

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
                subgroup = norm_intersectional_subgroup(m.group(1))
                inter_correct[(cur_prompt, subgroup)] = float(m.group(2))

        if section == "inter_no" and cur_prompt:
            m = re.match(r"^\s{2}([A-Za-z_]+)\s+\d+/\d+ \(([0-9.]+)%\)", line)
            if m:
                subgroup = norm_intersectional_subgroup(m.group(1))
                inter_no[(cur_prompt, subgroup)] = float(m.group(2))

        if section == "demo" and cur_prompt and in_demo_by_race:
            m = re.match(r"^\s{2}([A-Za-z_]+):\s+([0-9.]+)% \(n=\d+\)", line)
            if m:
                subgroup = m.group(1).strip().lower().replace("-", "_")
                race_correct[(cur_prompt, subgroup)] = float(m.group(2))

        if section == "race_no" and cur_prompt:
            m = re.match(r"^\s{2}([A-Za-z_]+)\s+\d+/\d+ \(([0-9.]+)%\)", line)
            if m:
                subgroup = m.group(1).strip().lower().replace("-", "_")
                race_no[(cur_prompt, subgroup)] = float(m.group(2))

    overall_map: Dict[str, Dict[str, float]] = {}
    for row in overall_rows:
        overall_map.setdefault(row["prompt"], {})[row["metric"]] = row["value"]
    overall = []
    for prompt, vals in overall_map.items():
        overall.append(
            {
                "prompt": prompt,
                "correct_pct": vals.get("correct_pct"),
                "wrong_pct": vals.get("wrong_pct"),
                "no_author_pct": vals.get("no_author_pct"),
            }
        )

    subgroup = []
    if dataset == "intersectional" and (inter_correct or inter_no):
        keys = sorted(set(inter_correct.keys()) | set(inter_no.keys()))
        for prompt, sg in keys:
            correct = inter_correct.get((prompt, sg))
            no_author = inter_no.get((prompt, sg))
            wrong = None
            if correct is not None and no_author is not None:
                wrong = round(100.0 - correct - no_author, 3)
            subgroup.append(
                {
                    "prompt": prompt,
                    "subgroup": sg,
                    "correct_pct": correct,
                    "wrong_pct": wrong,
                    "no_author_pct": no_author,
                }
            )
    elif dataset == "multirace" and (race_correct or race_no):
        keys = sorted(set(race_correct.keys()) | set(race_no.keys()))
        for prompt, sg in keys:
            correct = race_correct.get((prompt, sg))
            no_author = race_no.get((prompt, sg))
            wrong = None
            if correct is not None and no_author is not None:
                wrong = round(100.0 - correct - no_author, 3)
            subgroup.append(
                {
                    "prompt": prompt,
                    "subgroup": sg,
                    "correct_pct": correct,
                    "wrong_pct": wrong,
                    "no_author_pct": no_author,
                }
            )

    return overall, subgroup


def fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return ""
    if abs(v) < 0.05:
        v = 0.0
    return f"{v:.1f}%"


def find_latest_run_dir(spec: RunSpec) -> Optional[Path]:
    if spec.fixed_run_dir is not None:
        return spec.fixed_run_dir if spec.fixed_run_dir.exists() else None
    patterns = [
        f"prompt_attribution_rag_together_batch_{spec.glob_token}_{spec.dataset}_with_quotes_labeled_self*",
        f"prompt_attribution_rag_together_parallel_{spec.glob_token}_{spec.dataset}_with_quotes_labeled_self*",
    ]
    candidates: List[Path] = []
    for base in RAG_RESULTS_BASES:
        if not base.exists():
            continue
        for pat in patterns:
            candidates.extend([p for p in base.glob(pat) if p.is_dir()])
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def count_completed_units(raw_outputs_path: Path) -> int:
    unit_prompt_outputs: Dict[Tuple[str, str], Dict[str, bool]] = {}
    raw = raw_outputs_path.read_bytes().replace(b"\x00", b"")
    text = io.StringIO(raw.decode("utf-8", errors="replace"))
    reader = csv.DictReader(text)
    for row in reader:
        quote_id = str(row.get("quote_id", "")).strip()
        run = str(row.get("run", "")).strip()
        if not quote_id or not run:
            continue

        base_prompt = str(row.get("base_prompt_type", "")).strip().lower()
        if not base_prompt:
            prompt_type = str(row.get("prompt_type", "")).strip().lower()
            if prompt_type.startswith("direct"):
                base_prompt = "direct"
            elif prompt_type.startswith("indirect_overt"):
                base_prompt = "indirect_overt"
            elif prompt_type.startswith("indirect"):
                base_prompt = "indirect"
        if base_prompt not in {"direct", "indirect", "indirect_overt"}:
            continue

        output_nonempty = bool(str(row.get("llm_output", "")).strip())
        key = (quote_id, run)
        if key not in unit_prompt_outputs:
            unit_prompt_outputs[key] = {"direct": False, "indirect": False, "indirect_overt": False}
        unit_prompt_outputs[key][base_prompt] = unit_prompt_outputs[key][base_prompt] or output_nonempty

    return sum(1 for p in unit_prompt_outputs.values() if p["direct"] and p["indirect"] and p["indirect_overt"])


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    dataset_sizes = {k: count_dataset_quotes(v) for k, v in DATASET_PATHS.items()}
    rows_status: List[Dict] = []
    rows_overall_inter: List[Dict] = []
    rows_subgroup_inter: List[Dict] = []
    rows_overall_multi: List[Dict] = []
    rows_subgroup_multi: List[Dict] = []
    running_notes: List[str] = []

    for spec in RUN_SPECS:
        run_dir = find_latest_run_dir(spec)
        config_path = run_dir / "config.json" if run_dir else None
        summary_path = run_dir / "analysis" / "summary.txt" if run_dir else None
        raw_path = run_dir / "raw_outputs.csv" if run_dir else None

        expected_units = dataset_sizes[spec.dataset] * 3
        if config_path and config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            expected_units = int(cfg.get("total_quotes", dataset_sizes[spec.dataset])) * int(cfg.get("runs", 3))

        complete_units = 0
        if raw_path and raw_path.exists():
            complete_units = count_completed_units(raw_path)
        pending_units = max(expected_units - complete_units, 0)

        status = spec.status
        if status == "done" and (not summary_path or not summary_path.exists()):
            status = "running"
        if pending_units == 0 and summary_path and summary_path.exists():
            status = "done"
        if status == "running" and (not summary_path or not summary_path.exists()):
            running_notes.append(f"`{spec.model}` for **{spec.dataset}** is marked running; no prompt-rate table row is included for it.")

        rows_status.append(
            {
                "dataset": spec.dataset,
                "model": spec.model,
                "status": status,
                "pending": pending_units,
                "complete": complete_units,
                "expected": expected_units,
                "source_csv": str(raw_path) if raw_path and raw_path.exists() else "",
                "summary": str(summary_path) if summary_path and summary_path.exists() else "",
            }
        )

        if not summary_path or not summary_path.exists():
            continue

        overall, subgroup = parse_summary(summary_path, spec.dataset)
        for o in overall:
            rec = {"model": spec.model, **o}
            if spec.dataset == "intersectional":
                rows_overall_inter.append(rec)
            else:
                rows_overall_multi.append(rec)

        for s in subgroup:
            rec = {"model": spec.model, **s}
            if spec.dataset == "intersectional":
                rows_subgroup_inter.append(rec)
            else:
                rows_subgroup_multi.append(rec)

    rows_overall_inter.sort(key=lambda r: (r["model"], PROMPT_ORDER.get(r["prompt"], 99)))
    rows_overall_multi.sort(key=lambda r: (r["model"], PROMPT_ORDER.get(r["prompt"], 99)))
    rows_subgroup_inter.sort(
        key=lambda r: (r["model"], PROMPT_ORDER.get(r["prompt"], 99), INTERSECTIONAL_SUBGROUP_ORDER.get(r["subgroup"], 99), r["subgroup"])
    )
    rows_subgroup_multi.sort(
        key=lambda r: (r["model"], PROMPT_ORDER.get(r["prompt"], 99), MULTIRACE_SUBGROUP_ORDER.get(r["subgroup"], 99), r["subgroup"])
    )

    lines: List[str] = []
    lines.append("# RAG Results Summary")
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
            [
                [r["model"], r["prompt"], r["subgroup"], fmt_pct(r["correct_pct"]), fmt_pct(r["wrong_pct"]), fmt_pct(r["no_author_pct"])]
                for r in rows_subgroup_inter
            ],
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
            [
                [r["model"], r["prompt"], r["subgroup"], fmt_pct(r["correct_pct"]), fmt_pct(r["wrong_pct"]), fmt_pct(r["no_author_pct"])]
                for r in rows_subgroup_multi
            ],
        )
    )
    lines.append("")
    lines.append("## Notes")
    for note in running_notes:
        lines.append(f"- {note}")
    lines.append("- Subgroup `% Wrong` is computed as `100 - %Correct - %NoAuthor` using subgroup rates parsed from summary sections.")
    lines.append(
        "- For multirace subgroup split, `%Correct` comes from `ACCURACY BY DEMOGRAPHIC GROUP` (`DIRECT/INDIRECT/INDIRECT_OVERT - By Race`) and `%NoAuthor` comes from `NO AUTHOR MENTIONED BY RACE` in each summary."
    )
    lines.append(
        "- For intersectional subgroup split, `%Correct` comes from `INTERSECTIONAL ANALYSIS (Gender × Race)` and `%NoAuthor` comes from `NO AUTHOR MENTIONED BY SUBGROUP (RACE x GENDER)` in each summary."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
