"""OpenAI gpt-4o-mini multiple-choice demographic labeling (Tier-2, OpenAI half).

For each author, asks gpt-4o-mini a shuffled multiple-choice question for race (A-G)
and gender (A-C) and decodes the returned letter back to a category. Pair the output
with `perplexity_validation.py` for the two-model consensus described in the paper
(keep authors where both models agree; exclude authors where either predicts "Other").

Sampling: temperature = 0.0. Answer order shuffled per request.

Requires OPENAI_API_KEY. Resumable: appends one row per author and skips authors
already present in the output CSV on re-run.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# --- AttriBench path resolution ---
_ATTRIBENCH_ROOT = os.environ.get(
    "ATTRIBENCH_ROOT", str(Path(__file__).resolve().parents[2])
)
_ATTRIBENCH_RESULTS = os.environ.get(
    "ATTRIBENCH_RESULTS", os.path.join(_ATTRIBENCH_ROOT, "results")
)
_LABELING_DIR = os.path.join(_ATTRIBENCH_RESULTS, "demographic_labeling")
# ----------------------------------

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0

# System instruction (verbatim from the appendix).
SYSTEM_INSTRUCTION = (
    "You are an evaluator. Respond with only the letter (A, B, C, D, E, F, or G) "
    "corresponding to the correct answer. Do not include any other text."
)

RACE_CHOICES = ["Black", "White", "Asian", "Middle Eastern", "Indigenous", "Latino", "Other"]
GENDER_CHOICES = ["Male", "Female", "Other"]

FIELDS = ["author", "chatgpt_race", "chatgpt_gender", "race_raw_letter", "gender_raw_letter", "error"]


def build_mcq(question: str, choices: list[str]) -> tuple[str, list[str]]:
    """Return (prompt, shuffled_choices). Answer order is shuffled per request."""
    options = choices.copy()
    random.shuffle(options)
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(f"({letter}) {opt}" for letter, opt in zip(letters, options))
    return f"{question}\n{body}", options


def ask(client: OpenAI, question: str, choices: list[str]) -> tuple[str | None, str | None]:
    """Return (category, raw_letter). category is None if the reply can't be decoded."""
    prompt, options = build_mcq(question, choices)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    letter = text[:1].upper() if text else ""
    idx = ord(letter) - ord("A") if letter.isalpha() else -1
    if 0 <= idx < len(options):
        return options[idx].lower(), letter
    return None, (letter or None)


def load_authors(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "author" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} must have an 'author' column")
        seen, out = set(), []
        for row in reader:
            a = (row.get("author") or "").strip()
            if a and a not in seen:
                seen.add(a)
                out.append(a)
        return out


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {(r.get("author") or "").strip() for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-file", type=Path,
                    default=Path(_LABELING_DIR) / "authors_to_label.csv",
                    help="CSV with an 'author' column.")
    ap.add_argument("--output-file", type=Path,
                    default=Path(_LABELING_DIR) / "chatgpt_labels.csv")
    ap.add_argument("--query-gender", action="store_true",
                    help="Also label gender (race is always labeled).")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (in environment or .env).")
    client = OpenAI()

    authors = load_authors(args.input_file)
    done = load_done(args.output_file)
    todo = [a for a in authors if a not in done]
    print(f"authors={len(authors)} done={len(done)} todo={len(todo)}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output_file.exists()
    with args.output_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, author in enumerate(todo, 1):
            row = {"author": author}
            try:
                race, race_letter = ask(client, f"What is the race/ethnicity of {author}?", RACE_CHOICES)
                row["chatgpt_race"] = race or ""
                row["race_raw_letter"] = race_letter or ""
                if args.query_gender:
                    gender, gender_letter = ask(client, f"What is the gender of {author}?", GENDER_CHOICES)
                    row["chatgpt_gender"] = gender or ""
                    row["gender_raw_letter"] = gender_letter or ""
            except Exception as e:  # noqa: BLE001 — record and continue
                row["error"] = str(e)
            writer.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
