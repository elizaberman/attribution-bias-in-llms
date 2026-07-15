"""Wikidata demographic retrieval (Tier-1 labeling).

For each author name:
  1. Call the Wikidata `wbsearchentities` endpoint to retrieve up to three candidate
     entities ranked by relevance.
  2. For each candidate (best rank first), fetch claims and attempt to extract
     gender (P21, "sex or gender") and race/ethnicity (P172, "ethnic group").
  3. Resolve the property value entity IDs to human-readable English labels via an
     additional `wbgetentities` call.
  4. Take the first candidate that yields the properties (preferring one that has
     both), and mark that field's source as "wikidata".

Considering up to three candidates improves robustness to stub entities that lack
populated demographic properties. Authors not resolved here fall through to LLM-based
labeling (`openai_mc_labeling.py` + `perplexity_validation.py`).

Wikidata returns raw English labels (gender e.g. "male"/"female"; ethnicity is
free-form, e.g. "African Americans"). Mapping ethnicity labels onto the paper's four
race categories is applied downstream, not here.

No API key required, but Wikidata asks for a descriptive User-Agent (set below).
Resumable: appends one row per author, skips authors already in the output CSV.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import requests

# --- AttriBench path resolution ---
_ATTRIBENCH_ROOT = os.environ.get(
    "ATTRIBENCH_ROOT", str(Path(__file__).resolve().parents[2])
)
_ATTRIBENCH_RESULTS = os.environ.get(
    "ATTRIBENCH_RESULTS", os.path.join(_ATTRIBENCH_ROOT, "results")
)
_LABELING_DIR = os.path.join(_ATTRIBENCH_RESULTS, "demographic_labeling")
# ----------------------------------

API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = os.environ.get(
    "WIKIDATA_USER_AGENT",
    "AttriBench-demographic-labeling/1.0 (research; contact via repo)",
)
P_GENDER = "P21"   # sex or gender
P_ETHNICITY = "P172"  # ethnic group
MAX_CANDIDATES = 3
SLEEP_BETWEEN = 0.2

FIELDS = ["author", "wikidata_qid", "gender", "race_ethnicity",
          "gender_source", "race_source", "error"]


def _get(session: requests.Session, params: dict) -> dict:
    params = {**params, "format": "json"}
    r = session.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def search_candidates(session: requests.Session, name: str) -> list[str]:
    data = _get(session, {
        "action": "wbsearchentities", "search": name,
        "language": "en", "type": "item", "limit": MAX_CANDIDATES,
    })
    return [hit["id"] for hit in data.get("search", [])][:MAX_CANDIDATES]


def _claim_value_id(claims: dict, prop: str) -> str | None:
    for claim in claims.get(prop, []):
        snak = claim.get("mainsnak", {})
        val = snak.get("datavalue", {}).get("value", {})
        if isinstance(val, dict) and val.get("id"):
            return val["id"]
    return None


def resolve_label(session: requests.Session, qid: str) -> str | None:
    data = _get(session, {
        "action": "wbgetentities", "ids": qid,
        "props": "labels", "languages": "en",
    })
    ent = data.get("entities", {}).get(qid, {})
    return ent.get("labels", {}).get("en", {}).get("value")


def fetch_demographics(session: requests.Session, name: str) -> dict:
    """Return {gender, race_ethnicity, wikidata_qid} using up to 3 candidates."""
    result = {"gender": None, "race_ethnicity": None, "wikidata_qid": None}
    for qid in search_candidates(session, name):
        data = _get(session, {"action": "wbgetentities", "ids": qid, "props": "claims"})
        claims = data.get("entities", {}).get(qid, {}).get("claims", {})
        gender_id = _claim_value_id(claims, P_GENDER)
        ethnicity_id = _claim_value_id(claims, P_ETHNICITY)
        if not gender_id and not ethnicity_id:
            continue
        if result["gender"] is None and gender_id:
            result["gender"] = resolve_label(session, gender_id)
        if result["race_ethnicity"] is None and ethnicity_id:
            result["race_ethnicity"] = resolve_label(session, ethnicity_id)
        if result["wikidata_qid"] is None:
            result["wikidata_qid"] = qid
        if result["gender"] and result["race_ethnicity"]:
            break  # this candidate has both — stop early
    return result


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
                    default=Path(_LABELING_DIR) / "wikidata_labels.csv")
    args = ap.parse_args()

    authors = load_authors(args.input_file)
    done = load_done(args.output_file)
    todo = [a for a in authors if a not in done]
    print(f"authors={len(authors)} done={len(done)} todo={len(todo)}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output_file.exists()
    with args.output_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, author in enumerate(todo, 1):
            row = {"author": author}
            try:
                info = fetch_demographics(session, author)
                row["wikidata_qid"] = info["wikidata_qid"] or ""
                row["gender"] = info["gender"] or ""
                row["race_ethnicity"] = info["race_ethnicity"] or ""
                row["gender_source"] = "wikidata" if info["gender"] else ""
                row["race_source"] = "wikidata" if info["race_ethnicity"] else ""
            except Exception as e:  # noqa: BLE001 — record and continue
                row["error"] = str(e)
            writer.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            time.sleep(SLEEP_BETWEEN)
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
