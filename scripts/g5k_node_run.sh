#!/usr/bin/env bash
# Runs on a reserved Grid'5000 GPU node: benchmarks every rostered model-language
# pair and checkpoints each result as soon as it is produced.
#
# Environment:
#   LRB_ROOT        project checkout on the node        (default: $HOME/benchmark-llms-landuse-relevance)
#   LRB_DATA_ROOT   vendored multilingual data root     (default: $LRB_ROOT/data/translations)
#   LRB_RESULTS     directory for run results           (default: $LRB_ROOT/results)
#   LRB_BATCH_SIZE  prompts per forward pass            (default: 16)
#   LRB_MAX_NEW_TOKENS  generation budget per prompt     (default: 4096)
#   LRB_SHARD_INDEX shard index among deterministic pairs (default: 0)
#   LRB_SHARD_COUNT number of deterministic pair shards   (default: 1)
#   LRB_MODEL_ID    optional single model to run instead of the full roster
#   LRB_DRY_RUN     print sizing information and exit     (default: 0)
#   HF_HOME         Hugging Face cache                  (default: node-local /tmp scratch)
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"   # oarsub runs a non-login shell
LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
LRB_DATA_ROOT="${LRB_DATA_ROOT:-$LRB_ROOT/data/translations}"
LRB_RESULTS="${LRB_RESULTS:-$LRB_ROOT/results}"
LRB_BATCH_SIZE="${LRB_BATCH_SIZE:-16}"
LRB_MAX_NEW_TOKENS="${LRB_MAX_NEW_TOKENS:-4096}"
LRB_SHARD_INDEX="${LRB_SHARD_INDEX:-0}"
LRB_SHARD_COUNT="${LRB_SHARD_COUNT:-1}"
LRB_MODEL_ID="${LRB_MODEL_ID:-}"
LRB_DRY_RUN="${LRB_DRY_RUN:-0}"
GTE_MODEL_ID="Alibaba-NLP/gte-multilingual-reranker-base"
GTE_TRANSFORMERS_VERSION="5.11.0"
GLINER2_MODEL_ID="fastino/gliner2.5-multi-v1"
GLINER2_VERSION="2.0.0"
GLICLASS_MODEL_ID="knowledgator/gliclass-multilang-mini"
GGUF_MODEL_ID="unsloth/Qwen3.8-27B-GGUF@UD-IQ2_XXS"
LLAMA_CPP_PYTHON_VERSION="0.3.35"

if [[ "${1:-}" == "--dry-run" ]]; then
  LRB_DRY_RUN=1
  shift
fi
if (( $# > 0 )); then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ! "$LRB_SHARD_INDEX" =~ ^[0-9]+$ || ! "$LRB_SHARD_COUNT" =~ ^[1-9][0-9]*$ \
  || "$LRB_SHARD_INDEX" -ge "$LRB_SHARD_COUNT" ]]; then
  echo "invalid LRB_SHARD_INDEX/LRB_SHARD_COUNT: $LRB_SHARD_INDEX/$LRB_SHARD_COUNT" >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-/tmp/$USER/hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
# Laya's loader probes TensorFlow unless this is disabled; the benchmark is
# PyTorch-only and should not spend startup time importing a second runtime.
export USE_TF=0

cd "$LRB_ROOT"
mkdir -p "$LRB_RESULTS" "$HF_HOME"

# GTE's remote model code calls get_extended_attention_mask, absent from the
# locked Transformers 5.17 runtime. Keep its compatible runtime isolated.
# gliner2 pins Transformers<5 and llama.cpp needs a CUDA build: both get their own
# environment so the locked runtime every other model uses is never touched.
case "$LRB_MODEL_ID" in
  "$GTE_MODEL_ID") export UV_PROJECT_ENVIRONMENT="$LRB_ROOT/.venv-gte-${OAR_JOB_ID:-manual}" ;;
  "$GLINER2_MODEL_ID") export UV_PROJECT_ENVIRONMENT="$LRB_ROOT/.venv-gliner2-${OAR_JOB_ID:-manual}" ;;
  "$GGUF_MODEL_ID") export UV_PROJECT_ENVIRONMENT="$LRB_ROOT/.venv-gguf-${OAR_JOB_ID:-manual}" ;;
esac

echo "== node: $(hostname)  job: ${OAR_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

extras=(--extra inference)
if [[ "$LRB_MODEL_ID" == "$GLICLASS_MODEL_ID" ]]; then
  extras+=(--extra scoring)
fi
uv sync "${extras[@]}" --frozen --no-dev
if [[ "$LRB_MODEL_ID" == "$GLINER2_MODEL_ID" ]]; then
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    "gliner2[local]==$GLINER2_VERSION" "transformers>=4.38,<5" protobuf sentencepiece
  echo "== runtime: gliner2 $GLINER2_VERSION (isolated Transformers 4)"
fi
if [[ "$LRB_MODEL_ID" == "$GGUF_MODEL_ID" ]]; then
  module load cuda 2>/dev/null || true
  CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    --no-binary-package llama-cpp-python "llama-cpp-python==$LLAMA_CPP_PYTHON_VERSION" jinja2
  echo "== runtime: llama-cpp-python $LLAMA_CPP_PYTHON_VERSION (CUDA build)"
fi
if [[ "$LRB_MODEL_ID" == "$GTE_MODEL_ID" ]]; then
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    "transformers==$GTE_TRANSFORMERS_VERSION"
  installed_transformers_version="$(
    "$UV_PROJECT_ENVIRONMENT/bin/python" -c \
      'from importlib.metadata import version; print(version("transformers"))'
  )"
  if [[ "$installed_transformers_version" != "$GTE_TRANSFORMERS_VERSION" ]]; then
    echo "expected Transformers $GTE_TRANSFORMERS_VERSION, found $installed_transformers_version" >&2
    exit 1
  fi
  echo "== runtime: Transformers $installed_transformers_version (GTE compatibility)"
fi

mapfile -t language_rows < <(uv run --no-sync lrb languages --data-root "$LRB_DATA_ROOT")
if (( ${#language_rows[@]} == 0 )); then
  echo "no active languages found under $LRB_DATA_ROOT" >&2
  exit 1
fi

languages=()
rows_per_language=""
for language_row in "${language_rows[@]}"; do
  IFS=$'\t' read -r language row_count <<<"$language_row"
  if [[ -z "$language" || ! "$row_count" =~ ^[0-9]+$ ]]; then
    echo "invalid language inventory row: $language_row" >&2
    exit 1
  fi
  if [[ "$row_count" != "300" ]]; then
    echo "language $language has $row_count rows; expected 300" >&2
    exit 1
  fi
  if [[ -z "$rows_per_language" ]]; then
    rows_per_language="$row_count"
  elif [[ "$rows_per_language" != "$row_count" ]]; then
    echo "language row counts are not uniform" >&2
    exit 1
  fi
  languages+=("$language")
done

mapfile -t models < <(uv run --no-sync lrb models | cut -f1)
mapfile -t scoring_models < <(uv run --no-sync lrb scorers | cut -f1)
# A scoring model is benchmarked by `lrb score`, not `lrb run`: it is never prompted for
# text. Dispatch on which roster the id belongs to so the submitter does not have to.
is_scoring=0
if [[ -n "$LRB_MODEL_ID" ]]; then
  for model in "${scoring_models[@]}"; do
    if [[ "$model" == "$LRB_MODEL_ID" ]]; then
      is_scoring=1
      break
    fi
  done
  if (( is_scoring == 0 )); then
    model_found=0
    for model in "${models[@]}"; do
      if [[ "$model" == "$LRB_MODEL_ID" ]]; then
        model_found=1
        break
      fi
    done
    if (( model_found == 0 )); then
      echo "unknown LRB_MODEL_ID: $LRB_MODEL_ID" >&2
      exit 2
    fi
  fi
  models=("$LRB_MODEL_ID")
fi
pair_count=$(( ${#models[@]} * ${#languages[@]} ))
total_rows=$(( rows_per_language * ${#languages[@]} ))
estimated_prompts=$(( rows_per_language * pair_count ))
assigned_pair_count=0
for (( pair_index = 0; pair_index < pair_count; pair_index++ )); do
  if (( pair_index % LRB_SHARD_COUNT == LRB_SHARD_INDEX )); then
    assigned_pair_count=$(( assigned_pair_count + 1 ))
  fi
done
echo "== sizing: model=${LRB_MODEL_ID:-all} languages=${#languages[@]} rows=$total_rows pairs=$pair_count assigned_pairs=$assigned_pair_count prompts=$estimated_prompts batch_size=$LRB_BATCH_SIZE max_new_tokens=$LRB_MAX_NEW_TOKENS shard=$LRB_SHARD_INDEX/$LRB_SHARD_COUNT"

if [[ "$LRB_DRY_RUN" == "1" ]]; then
  exit 0
fi

if (( is_scoring == 1 )); then
  # Each scorer declares its own prompt file; override only when asked to.
  uv run --no-sync lrb score "$LRB_MODEL_ID" \
    --data-root "$LRB_DATA_ROOT" \
    ${LRB_SCORER_PROMPT:+--prompt "$LRB_SCORER_PROMPT"} \
    --out "$LRB_RESULTS" \
    --batch-size "$LRB_BATCH_SIZE" \
    --shard-index "$LRB_SHARD_INDEX" \
    --shard-count "$LRB_SHARD_COUNT"
elif [[ -n "$LRB_MODEL_ID" ]]; then
  uv run --no-sync lrb run "$LRB_MODEL_ID" \
    --data-root "$LRB_DATA_ROOT" \
    --prompt data/prompt.txt \
    --out "$LRB_RESULTS" \
    --batch-size "$LRB_BATCH_SIZE" \
    --max-new-tokens "$LRB_MAX_NEW_TOKENS" \
    --shard-index "$LRB_SHARD_INDEX" \
    --shard-count "$LRB_SHARD_COUNT"
else
  uv run --no-sync lrb run-all \
    --data-root "$LRB_DATA_ROOT" \
    --prompt data/prompt.txt \
    --out "$LRB_RESULTS" \
    --batch-size "$LRB_BATCH_SIZE" \
    --max-new-tokens "$LRB_MAX_NEW_TOKENS" \
    --shard-index "$LRB_SHARD_INDEX" \
    --shard-count "$LRB_SHARD_COUNT"
fi

uv run --no-sync lrb report --results-dir "$LRB_RESULTS"
