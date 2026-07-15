#!/usr/bin/env bash
#
# Evidence-conditioned (RAG) sweep: open-weight models x both datasets on Together AI.
#
# Wrapper around `rag_together_exp.py` (Together Batch API) that reproduces the
# paper's RAG runs for the open-weight models. The closed frontier APIs are handled
# by ../providers/run_batch_rag_openai_claude_gemini.sh.
#
# Results -> $ATTRIBENCH_RESULTS/rag_together_batch/
# Logs    -> $ATTRIBENCH_RESULTS/logs
#
# Needs TOGETHER_API_KEY (generation) and OPENAI_API_KEY (retrieval embeddings);
# both are read from the environment, else from the repo-root .env.
#
# NOTE: the MODELS array below is a partial selection left from the paper's staged
# reruns — the commented-out entries were already complete, not excluded from the
# paper. As written, only Llama-4 Maverick and Mixtral-8x7B will run; uncomment the
# rest for the full sweep.

# Resolve repo and results roots without any cluster-specific assumptions.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
: "${ATTRIBENCH_ROOT:=$( cd "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd )}"
: "${ATTRIBENCH_RESULTS:=$ATTRIBENCH_ROOT/results}"
export ATTRIBENCH_ROOT ATTRIBENCH_RESULTS
mkdir -p "$ATTRIBENCH_RESULTS"


set -euo pipefail

ROOT_DIR="${ATTRIBENCH_ROOT}"
HOME_RESULTS_BASE="${ATTRIBENCH_RESULTS}/rag_together_batch"
SCRATCH_LOG_DIR="${ATTRIBENCH_RESULTS}/logs"

RUNS="${RUNS:-3}"
SAVE_EVERY="${SAVE_EVERY:-4000}"
TOP_K="${TOP_K:-5}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-128}"
MAX_FULL_SIMS_N="${MAX_FULL_SIMS_N:-15000}"
CONTEXT_LABEL_MODE="${CONTEXT_LABEL_MODE:-labeled}"
EXCLUDE_SELF_MATCH="${EXCLUDE_SELF_MATCH:-false}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
POLL_EVERY_S="${POLL_EVERY_S:-60}"
MAX_REQUESTS_PER_BATCH="${MAX_REQUESTS_PER_BATCH:-48000}"

DATASETS=(
  "${ROOT_DIR}/1_dataset_construction/datasets/intersectional_with_quotes.csv"
  "${ROOT_DIR}/1_dataset_construction/datasets/multirace_with_quotes.csv"
)

MODELS=(
  # "deepseek-ai/DeepSeek-V3.1"
  # "Qwen/Qwen3-Next-80B-A3B-Instruct"
  # "Qwen/Qwen3.5-397B-A17B"
  # "zai-org/GLM-5"
  # "openai/gpt-oss-120b"
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
)

mkdir -p "${SCRATCH_LOG_DIR}" "${HOME_RESULTS_BASE}"

VENV_PATH="${VENV_PATH:-$ATTRIBENCH_ROOT/.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

ENV_FILE="${ROOT_DIR}/.env"

if [[ -z "${OPENAI_API_KEY:-}" && -f "$HOME/.openai_key" ]]; then
  export OPENAI_API_KEY="$(cat "$HOME/.openai_key")"
elif [[ -z "${OPENAI_API_KEY:-}" && -f "${ENV_FILE}" ]]; then
  OPENAI_API_KEY_LINE=$(grep -E '^OPENAI_API_KEY=' "${ENV_FILE}" || true)
  if [[ -n "${OPENAI_API_KEY_LINE}" ]]; then
    export OPENAI_API_KEY="${OPENAI_API_KEY_LINE#OPENAI_API_KEY=}"
  fi
fi

if [[ -z "${TOGETHER_API_KEY:-}" && -z "${TOGETHERAI_API_KEY:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set (required for embeddings)."
  exit 1
fi

if [[ -z "${TOGETHER_API_KEY:-}" && -z "${TOGETHERAI_API_KEY:-}" ]]; then
  echo "ERROR: TOGETHER_API_KEY (or TOGETHERAI_API_KEY) is not set."
  exit 1
fi

echo "SCRATCH_LOG_DIR=${SCRATCH_LOG_DIR}"
echo "HOME_RESULTS_BASE=${HOME_RESULTS_BASE}"
echo "RUNS=${RUNS}"
echo "SAVE_EVERY=${SAVE_EVERY}"
echo "TOP_K=${TOP_K}"
echo "CONTEXT_LABEL_MODE=${CONTEXT_LABEL_MODE}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "TOP_P=${TOP_P}"
echo "POLL_EVERY_S=${POLL_EVERY_S}"
echo "MAX_REQUESTS_PER_BATCH=${MAX_REQUESTS_PER_BATCH}"

for CSV_PATH in "${DATASETS[@]}"; do
  if [[ ! -f "${CSV_PATH}" ]]; then
    echo "ERROR: dataset not found: ${CSV_PATH}"
    exit 1
  fi

  CSV_BASENAME="$(basename "${CSV_PATH}")"
  if [[ "${CSV_BASENAME}" == *intersectional* ]]; then
    DATASET_TAG="intersectional_with_quotes"
  elif [[ "${CSV_BASENAME}" == *multirace* ]]; then
    DATASET_TAG="multirace_with_quotes"
  else
    DATASET_TAG="${CSV_BASENAME%.csv}"
  fi

  SUFFIX="${CONTEXT_LABEL_MODE}"
  if [[ "${EXCLUDE_SELF_MATCH}" != "true" ]]; then
    SUFFIX="${SUFFIX}_self"
  fi

  for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_TAG="$(echo "${MODEL_NAME}" | tr '/ .' '___')"
    OUT_DIR_BASE="${HOME_RESULTS_BASE}/prompt_attribution_rag_together_batch_${MODEL_TAG}_${DATASET_TAG}_${SUFFIX}"

    echo "============================================================"
    echo "CSV_PATH=${CSV_PATH}"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "OUT_DIR_BASE=${OUT_DIR_BASE}"
    echo "============================================================"

    RAG_ARGS=(
      --csv-path "${CSV_PATH}"
      --model "${MODEL_NAME}"
      --runs "${RUNS}"
      --save-every "${SAVE_EVERY}"
      --top-k "${TOP_K}"
      --embedding-model "${EMBEDDING_MODEL}"
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
      --max-full-sims-n "${MAX_FULL_SIMS_N}"
      --context-label-mode "${CONTEXT_LABEL_MODE}"
      --max-tokens "${MAX_TOKENS}"
      --temperature "${TEMPERATURE}"
      --top-p "${TOP_P}"
      --poll-every-s "${POLL_EVERY_S}"
      --max-requests-per-batch "${MAX_REQUESTS_PER_BATCH}"
      --out-dir "${OUT_DIR_BASE}"
    )

    if [[ "${EXCLUDE_SELF_MATCH}" == "true" ]]; then
      RAG_ARGS+=(--exclude-self-match)
    fi

    python "${SCRIPT_DIR}/rag_together_exp.py" "${RAG_ARGS[@]}"
  done
done

echo "All Together batch RAG runs completed."
