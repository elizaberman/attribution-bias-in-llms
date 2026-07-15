#!/usr/bin/env bash
#
# Evidence-conditioned (RAG) sweep: GPT-5.1, Claude 4.6 Sonnet, Gemini 2.5 Flash-Lite.
#
# Wrapper around `rag_batch_openai_claude_gemini.py` that reproduces the paper's RAG
# runs for the closed frontier APIs. Open-weight models are handled by
# ../runners/run_batched_rag_together_models.sh.
#
# Two env switches control scope:
#   DATASET_SELECTION = intersectional | multirace | both   (which datasets)
#   DATASET_MODE      = full | subset                       (which CSVs)
# Because these APIs are costly, the paper ran GPT-5.1 and Claude with
# DATASET_MODE=subset — the *_random_300matchings.csv files (1,200 quotes) — rather
# than the full datasets. Those are the models daggered in the paper's RAG figures.
#
# Results -> $ATTRIBENCH_RESULTS/rag_openai_claude_batch/
# Logs    -> $ATTRIBENCH_RESULTS/rag_openai_claude/logs
#
# Needs OPENAI_API_KEY (always — retrieval embeddings) plus the selected provider's
# key; read from the environment, else $HOME/.openai_key, else the repo-root .env.

# Resolve repo and results roots without any cluster-specific assumptions.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
: "${ATTRIBENCH_ROOT:=$( cd "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd )}"
: "${ATTRIBENCH_RESULTS:=$ATTRIBENCH_ROOT/results}"
export ATTRIBENCH_ROOT ATTRIBENCH_RESULTS
mkdir -p "$ATTRIBENCH_RESULTS"


set -euo pipefail

ROOT_DIR="${ATTRIBENCH_ROOT}"
PY_SCRIPT="${SCRIPT_DIR}/rag_batch_openai_claude_gemini.py"
OUT_BASE="${ATTRIBENCH_RESULTS}/rag_openai_claude_batch"
SCRATCH_LOG_DIR="${ATTRIBENCH_RESULTS}/rag_openai_claude/logs"
# Activate your virtualenv before running, or set VENV_PATH to point at one.
VENV_PATH="${VENV_PATH:-$ATTRIBENCH_ROOT/.venv}"
ENV_FILE="${ROOT_DIR}/.env"

RUNS="${RUNS:-3}"
SMOKE_ROWS="${SMOKE_ROWS:-0}"
TOP_K="${TOP_K:-5}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-128}"
MAX_FULL_SIMS_N="${MAX_FULL_SIMS_N:-15000}"
CONTEXT_LABEL_MODE="${CONTEXT_LABEL_MODE:-labeled}"
EXCLUDE_SELF_MATCH="${EXCLUDE_SELF_MATCH:-false}"
POLL_EVERY_S="${POLL_EVERY_S:-60}"
MAX_REQUESTS_PER_BATCH="${MAX_REQUESTS_PER_BATCH:-20000}"
PROVIDERS="${PROVIDERS:-openai,claude}"
OPENAI_ONLY="${OPENAI_ONLY:-false}"
CLAUDE_ONLY="${CLAUDE_ONLY:-false}"
GEMINI_ONLY="${GEMINI_ONLY:-false}"
DATASET_MODE="${DATASET_MODE:-full}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.1}"
OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.7}"
OPENAI_TOP_P="${OPENAI_TOP_P:-0.95}"
OPENAI_MAX_OUTPUT_TOKENS="${OPENAI_MAX_OUTPUT_TOKENS:-500}"
OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-none}"

CLAUDE_MODEL="${CLAUDE_MODEL:-claude-sonnet-4-6}"
CLAUDE_TEMPERATURE="${CLAUDE_TEMPERATURE:-0.7}"
CLAUDE_MAX_TOKENS="${CLAUDE_MAX_TOKENS:-500}"
CLAUDE_THINKING_BUDGET_TOKENS="${CLAUDE_THINKING_BUDGET_TOKENS:-0}"

GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash-lite}"
GEMINI_TEMPERATURE="${GEMINI_TEMPERATURE:-0.7}"
GEMINI_TOP_P="${GEMINI_TOP_P:-0.95}"
GEMINI_MAX_OUTPUT_TOKENS="${GEMINI_MAX_OUTPUT_TOKENS:-500}"
GEMINI_MAX_CONCURRENCY="${GEMINI_MAX_CONCURRENCY:-40}"
GEMINI_RATE_LIMIT_RPM="${GEMINI_RATE_LIMIT_RPM:-150}"

if [[ "${DATASET_MODE}" == "subset" ]]; then
  DEFAULT_DATASET_INTERSECTIONAL="${ROOT_DIR}/1_dataset_construction/datasets/intersectional_with_quotes_random_300matchings.csv"
  DEFAULT_DATASET_MULTIRACE="${ROOT_DIR}/1_dataset_construction/datasets/multirace_with_quotes_random_300matchings.csv"
elif [[ "${DATASET_MODE}" == "full" ]]; then
  DEFAULT_DATASET_INTERSECTIONAL="${ROOT_DIR}/1_dataset_construction/datasets/intersectional_with_quotes.csv"
  DEFAULT_DATASET_MULTIRACE="${ROOT_DIR}/1_dataset_construction/datasets/multirace_with_quotes.csv"
else
  echo "ERROR: DATASET_MODE must be one of: full, subset"
  exit 1
fi

DATASET_INTERSECTIONAL="${DATASET_INTERSECTIONAL:-${DEFAULT_DATASET_INTERSECTIONAL}}"
DATASET_MULTIRACE="${DATASET_MULTIRACE:-${DEFAULT_DATASET_MULTIRACE}}"

RETRIEVAL_CORPUS_INTERSECTIONAL="${RETRIEVAL_CORPUS_INTERSECTIONAL:-${DATASET_INTERSECTIONAL}}"
RETRIEVAL_CORPUS_MULTIRACE="${RETRIEVAL_CORPUS_MULTIRACE:-${DATASET_MULTIRACE}}"

REUSE_RESULTS_CSV_INTERSECTIONAL="${REUSE_RESULTS_CSV_INTERSECTIONAL:-}"
REUSE_RESULTS_CSV_MULTIRACE="${REUSE_RESULTS_CSV_MULTIRACE:-}"
DATASET_SELECTION="${DATASET_SELECTION:-both}"

case "${DATASET_SELECTION}" in
  intersectional)
    DATASETS=("${DATASET_INTERSECTIONAL}")
    ;;
  multirace)
    DATASETS=("${DATASET_MULTIRACE}")
    ;;
  both)
    DATASETS=("${DATASET_INTERSECTIONAL}" "${DATASET_MULTIRACE}")
    ;;
  *)
    echo "ERROR: DATASET_SELECTION must be one of: intersectional, multirace, both"
    exit 1
    ;;
esac

mkdir -p "${SCRATCH_LOG_DIR}" "${OUT_BASE}"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Missing venv: ${VENV_PATH}"
  exit 1
fi
source "${VENV_PATH}/bin/activate"

if [[ -z "${OPENAI_API_KEY:-}" && -f "$HOME/.openai_key" ]]; then
  export OPENAI_API_KEY="$(cat "$HOME/.openai_key")"
elif [[ -z "${OPENAI_API_KEY:-}" && -f "${ENV_FILE}" ]]; then
  OPENAI_API_KEY_LINE=$(grep -E '^OPENAI_API_KEY=' "${ENV_FILE}" || true)
  if [[ -n "${OPENAI_API_KEY_LINE}" ]]; then
    export OPENAI_API_KEY="${OPENAI_API_KEY_LINE#OPENAI_API_KEY=}"
  fi
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" && -f "${ENV_FILE}" ]]; then
  ANTHROPIC_API_KEY_LINE=$(grep -E '^ANTHROPIC_API_KEY=' "${ENV_FILE}" || true)
  if [[ -n "${ANTHROPIC_API_KEY_LINE}" ]]; then
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LINE#ANTHROPIC_API_KEY=}"
  fi
fi

if [[ -z "${GEMINI_API_KEY:-}" && -f "${ENV_FILE}" ]]; then
  GEMINI_API_KEY_LINE=$(grep -E '^GEMINI_API_KEY=' "${ENV_FILE}" || true)
  if [[ -n "${GEMINI_API_KEY_LINE}" ]]; then
    export GEMINI_API_KEY="${GEMINI_API_KEY_LINE#GEMINI_API_KEY=}"
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set (required for embeddings + OpenAI runs)."
  exit 1
fi

echo "RUNS=${RUNS}"
echo "SMOKE_ROWS=${SMOKE_ROWS}"
echo "TOP_K=${TOP_K}"
echo "CONTEXT_LABEL_MODE=${CONTEXT_LABEL_MODE}"
echo "MAX_REQUESTS_PER_BATCH=${MAX_REQUESTS_PER_BATCH}"
echo "PROVIDERS=${PROVIDERS}"
echo "OPENAI_ONLY=${OPENAI_ONLY}"
echo "CLAUDE_ONLY=${CLAUDE_ONLY}"
echo "GEMINI_ONLY=${GEMINI_ONLY}"
echo "DATASET_MODE=${DATASET_MODE}"
echo "OPENAI_MODEL=${OPENAI_MODEL}"
echo "OPENAI_TEMPERATURE=${OPENAI_TEMPERATURE}"
echo "OPENAI_TOP_P=${OPENAI_TOP_P}"
echo "OPENAI_MAX_OUTPUT_TOKENS=${OPENAI_MAX_OUTPUT_TOKENS}"
echo "OPENAI_REASONING_EFFORT=${OPENAI_REASONING_EFFORT}"
echo "CLAUDE_MODEL=${CLAUDE_MODEL}"
echo "CLAUDE_TEMPERATURE=${CLAUDE_TEMPERATURE}"
echo "CLAUDE_MAX_TOKENS=${CLAUDE_MAX_TOKENS}"
echo "CLAUDE_THINKING_BUDGET_TOKENS=${CLAUDE_THINKING_BUDGET_TOKENS}"
echo "GEMINI_MODEL=${GEMINI_MODEL}"
echo "GEMINI_TEMPERATURE=${GEMINI_TEMPERATURE}"
echo "GEMINI_TOP_P=${GEMINI_TOP_P}"
echo "GEMINI_MAX_OUTPUT_TOKENS=${GEMINI_MAX_OUTPUT_TOKENS}"
echo "GEMINI_MAX_CONCURRENCY=${GEMINI_MAX_CONCURRENCY}"
echo "GEMINI_RATE_LIMIT_RPM=${GEMINI_RATE_LIMIT_RPM}"
echo "DATASET_INTERSECTIONAL=${DATASET_INTERSECTIONAL}"
echo "DATASET_MULTIRACE=${DATASET_MULTIRACE}"
echo "RETRIEVAL_CORPUS_INTERSECTIONAL=${RETRIEVAL_CORPUS_INTERSECTIONAL}"
echo "RETRIEVAL_CORPUS_MULTIRACE=${RETRIEVAL_CORPUS_MULTIRACE}"
echo "REUSE_RESULTS_CSV_INTERSECTIONAL=${REUSE_RESULTS_CSV_INTERSECTIONAL}"
echo "REUSE_RESULTS_CSV_MULTIRACE=${REUSE_RESULTS_CSV_MULTIRACE}"
echo "DATASET_SELECTION=${DATASET_SELECTION}"

if [[ "${OPENAI_ONLY}" == "true" ]]; then
  PROVIDERS="openai"
fi
if [[ "${CLAUDE_ONLY}" == "true" ]]; then
  PROVIDERS="claude"
fi
if [[ "${GEMINI_ONLY}" == "true" ]]; then
  PROVIDERS="gemini"
fi

IFS=',' read -r -a PROVIDER_LIST <<< "${PROVIDERS}"

for CSV_PATH in "${DATASETS[@]}"; do
  if [[ ! -f "${CSV_PATH}" ]]; then
    echo "ERROR: dataset not found: ${CSV_PATH}"
    exit 1
  fi

  if [[ "${CSV_PATH}" == "${DATASET_INTERSECTIONAL}" ]]; then
    RETRIEVAL_CSV="${RETRIEVAL_CORPUS_INTERSECTIONAL}"
    REUSE_CSV="${REUSE_RESULTS_CSV_INTERSECTIONAL}"
  elif [[ "${CSV_PATH}" == "${DATASET_MULTIRACE}" ]]; then
    RETRIEVAL_CSV="${RETRIEVAL_CORPUS_MULTIRACE}"
    REUSE_CSV="${REUSE_RESULTS_CSV_MULTIRACE}"
  else
    RETRIEVAL_CSV="${CSV_PATH}"
    REUSE_CSV=""
  fi

  if [[ ! -f "${RETRIEVAL_CSV}" ]]; then
    echo "ERROR: retrieval corpus not found: ${RETRIEVAL_CSV}"
    exit 1
  fi

  COMMON_ARGS=(
    --csv-path "${CSV_PATH}"
    --retrieval-corpus-csv "${RETRIEVAL_CSV}"
    --runs "${RUNS}"
    --top-k "${TOP_K}"
    --embedding-model "${EMBEDDING_MODEL}"
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
    --max-full-sims-n "${MAX_FULL_SIMS_N}"
    --context-label-mode "${CONTEXT_LABEL_MODE}"
    --poll-every-s "${POLL_EVERY_S}"
    --max-requests-per-batch "${MAX_REQUESTS_PER_BATCH}"
    --out-dir "${OUT_BASE}"
  )

  if [[ "${SMOKE_ROWS}" =~ ^[0-9]+$ ]] && [[ "${SMOKE_ROWS}" -gt 0 ]]; then
    COMMON_ARGS+=(--subset-size "${SMOKE_ROWS}")
  fi

  if [[ -n "${REUSE_CSV}" ]]; then
    if [[ ! -f "${REUSE_CSV}" ]]; then
      echo "ERROR: reuse results CSV not found: ${REUSE_CSV}"
      exit 1
    fi
    COMMON_ARGS+=(--reuse-results-csv "${REUSE_CSV}")
  fi

  if [[ "${EXCLUDE_SELF_MATCH}" == "true" ]]; then
    COMMON_ARGS+=(--exclude-self-match)
  fi

  for PROVIDER in "${PROVIDER_LIST[@]}"; do
    case "${PROVIDER}" in
      openai)
        echo "============================================================"
        echo "Dataset: ${CSV_PATH}"
        echo "Retrieval corpus: ${RETRIEVAL_CSV}"
        echo "Reuse CSV: ${REUSE_CSV}"
        echo "Running provider=openai model=${OPENAI_MODEL}"
        echo "============================================================"

        python "${PY_SCRIPT}" \
          --provider openai \
          --model "${OPENAI_MODEL}" \
          --openai-temperature "${OPENAI_TEMPERATURE}" \
          --openai-top-p "${OPENAI_TOP_P}" \
          --openai-max-output-tokens "${OPENAI_MAX_OUTPUT_TOKENS}" \
          --openai-reasoning-effort "${OPENAI_REASONING_EFFORT}" \
          "${COMMON_ARGS[@]}"
        ;;
      claude)
        if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
          echo "ERROR: ANTHROPIC_API_KEY is not set (required for Claude runs)."
          exit 1
        fi
        echo "============================================================"
        echo "Dataset: ${CSV_PATH}"
        echo "Retrieval corpus: ${RETRIEVAL_CSV}"
        echo "Reuse CSV: ${REUSE_CSV}"
        echo "Running provider=claude model=${CLAUDE_MODEL}"
        echo "============================================================"

        python "${PY_SCRIPT}" \
          --provider claude \
          --model "${CLAUDE_MODEL}" \
          --claude-temperature "${CLAUDE_TEMPERATURE}" \
          --claude-max-tokens "${CLAUDE_MAX_TOKENS}" \
          --claude-thinking-budget-tokens "${CLAUDE_THINKING_BUDGET_TOKENS}" \
          "${COMMON_ARGS[@]}"
        ;;
      gemini)
        if [[ -z "${GEMINI_API_KEY:-}" ]]; then
          echo "ERROR: GEMINI_API_KEY is not set (required for Gemini runs)."
          exit 1
        fi
        echo "============================================================"
        echo "Dataset: ${CSV_PATH}"
        echo "Retrieval corpus: ${RETRIEVAL_CSV}"
        echo "Reuse CSV: ${REUSE_CSV}"
        echo "Running provider=gemini model=${GEMINI_MODEL}"
        echo "============================================================"

        python "${PY_SCRIPT}" \
          --provider gemini \
          --model "${GEMINI_MODEL}" \
          --gemini-temperature "${GEMINI_TEMPERATURE}" \
          --gemini-top-p "${GEMINI_TOP_P}" \
          --gemini-max-output-tokens "${GEMINI_MAX_OUTPUT_TOKENS}" \
          --gemini-max-concurrency "${GEMINI_MAX_CONCURRENCY}" \
          --gemini-rate-limit-rpm "${GEMINI_RATE_LIMIT_RPM}" \
          "${COMMON_ARGS[@]}"
        ;;
      *)
        echo "ERROR: unsupported provider '${PROVIDER}' in PROVIDERS=${PROVIDERS}"
        exit 1
        ;;
    esac
  done
done

echo "All requested RAG batch runs completed."
