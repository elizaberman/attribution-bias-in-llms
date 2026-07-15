#!/usr/bin/env bash
#
# Zero-shot (no-evidence) sweep: every open-weight model x both datasets on Together AI.
#
# Wrapper around `parallel_together_exp.py` that reproduces the paper's no-evidence
# runs. Prefer it over calling the runner directly: it counts the dataset rows and
# passes --n-rows (the runner defaults to 25 and would otherwise silently process
# only the first 25 quotes), and it pins the paper's run label and sampling
# parameters (RUN_LABEL, temperature 0.7, top_p 0.95, reasoning off).
#
# Results -> $ATTRIBENCH_RESULTS/$RUN_LABEL/<model>_<dataset>_full_runs<N>.csv
# Logs    -> $ATTRIBENCH_RESULTS/author_presence/logs
#
# Every knob is env-overridable, e.g.:
#   RUNS=3 TEMPERATURE=0.7 MAX_WORKERS=4 ./run_together_models_full.sh
#
# NOTE: the MODELS array below is a partial selection left from the paper's staged
# reruns — entries are commented out because they were already complete, not because
# they are excluded from the paper. Uncomment to run the full sweep.
# NOTE: prepare_fresh_output archives any existing output CSV before rerunning.

# Resolve repo and results roots without any cluster-specific assumptions.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
: "${ATTRIBENCH_ROOT:=$( cd "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd )}"
: "${ATTRIBENCH_RESULTS:=$ATTRIBENCH_ROOT/results}"
export ATTRIBENCH_ROOT ATTRIBENCH_RESULTS
mkdir -p "$ATTRIBENCH_RESULTS"


set -euo pipefail

ROOT_DIR="${ATTRIBENCH_ROOT}"
SCRATCH_BASE="${ATTRIBENCH_RESULTS}/author_presence"
LOG_DIR="${SCRATCH_BASE}/logs"
RESULTS_BASE="${ATTRIBENCH_RESULTS}"

RUNS="${RUNS:-3}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_WORKERS="${MAX_WORKERS:-4}"
TARGET_RPM="${TARGET_RPM:-1200}"
FLUSH_EVERY="${FLUSH_EVERY:-50}"
RUN_LABEL="${RUN_LABEL:-together_models_full_parallel_reasoning_off_t07_p095}"
RESULTS_DIR="${RESULTS_BASE}/${RUN_LABEL}"

MULTIRACE_DATA="${MULTIRACE_DATA:-${ROOT_DIR}/1_dataset_construction/datasets/multirace_with_quotes.csv}"
INTERSECT_DATA="${INTERSECT_DATA:-${ROOT_DIR}/1_dataset_construction/datasets/intersectional_with_quotes.csv}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

echo "LOG_DIR=${LOG_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"

VENV_PATH="${VENV_PATH:-$ATTRIBENCH_ROOT/.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

if [[ -z "${TOGETHER_API_KEY:-}" && -z "${TOGETHERAI_API_KEY:-}" && -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

if [[ -z "${TOGETHER_API_KEY:-}" && -z "${TOGETHERAI_API_KEY:-}" ]]; then
  echo "ERROR: TOGETHER_API_KEY (or TOGETHERAI_API_KEY) is not set."
  exit 1
fi

count_rows() {
  local data_file="$1"
  DATA_FILE="${data_file}" python - <<'PY'
import csv
import os
from pathlib import Path
path = Path(os.environ["DATA_FILE"])
with path.open(newline="", encoding="utf-8") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

prepare_fresh_output() {
  local out_csv="$1"
  if [[ -f "${out_csv}" ]]; then
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local backup="${out_csv%.csv}.backup_${ts}.csv"
    mv "${out_csv}" "${backup}"
    echo "Archived existing output to ${backup}"
  fi
}

run_full_fresh() {
  local model_name="$1"
  local data_file="$2"
  local out_csv="$3"
  local tag="$4"

  if [[ ! -f "${data_file}" ]]; then
    echo "ERROR: data file not found: ${data_file}"
    exit 1
  fi

  local n_rows
  n_rows="$(count_rows "${data_file}")"

  echo ""
  echo "================================================"
  echo "Dataset: ${tag}"
  echo "Data file: ${data_file}"
  echo "Output CSV: ${out_csv}"
  echo "Rows: ${n_rows}"
  echo "Runs: ${RUNS}"
  echo "Model: ${model_name}"
  echo "Temperature: ${TEMPERATURE}"
  echo "Top-p: ${TOP_P}"
  echo "Max workers: ${MAX_WORKERS}"
  echo "Target RPM: ${TARGET_RPM}"
  echo "================================================"

  python "${SCRIPT_DIR}/parallel_together_exp.py" \
    --data-file "${data_file}" \
    --output "${out_csv}" \
    --model "${model_name}" \
    --n-rows "${n_rows}" \
    --runs "${RUNS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-workers "${MAX_WORKERS}" \
    --target-rpm "${TARGET_RPM}" \
    --flush-every "${FLUSH_EVERY}" \
    --progress-every "${PROGRESS_EVERY}"
}

MODELS=(
  #"moonshotai/Kimi-K2.5"
  "deepseek-ai/DeepSeek-V3.1"
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
  "Qwen/Qwen3.5-397B-A17B"
  "zai-org/GLM-5"
  "openai/gpt-oss-120b"
)

for MODEL_NAME in "${MODELS[@]}"; do
  MODEL_TAG="$(echo "${MODEL_NAME}" | tr '/.' '__')"
  MULTIRACE_OUT="${RESULTS_DIR}/${MODEL_TAG}_multirace_full_runs${RUNS}.csv"
  INTERSECT_OUT="${RESULTS_DIR}/${MODEL_TAG}_intersectional_full_runs${RUNS}.csv"

  prepare_fresh_output "${MULTIRACE_OUT}"
  prepare_fresh_output "${INTERSECT_OUT}"

  run_full_fresh "${MODEL_NAME}" "${MULTIRACE_DATA}" "${MULTIRACE_OUT}" "multirace_full"
  run_full_fresh "${MODEL_NAME}" "${INTERSECT_DATA}" "${INTERSECT_OUT}" "intersectional_full"
done

echo ""
echo "All Together full fresh runs completed."
