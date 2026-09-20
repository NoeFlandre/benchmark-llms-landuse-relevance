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
#   LRB_DRY_RUN     print sizing information and exit     (default: 0)
#   HF_HOME         Hugging Face cache                  (default: node-local /tmp scratch)
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"   # oarsub runs a non-login shell
LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
LRB_DATA_ROOT="${LRB_DATA_ROOT:-$LRB_ROOT/data/translations}"
LRB_RESULTS="${LRB_RESULTS:-$LRB_ROOT/results}"
LRB_BATCH_SIZE="${LRB_BATCH_SIZE:-16}"
LRB_MAX_NEW_TOKENS="${LRB_MAX_NEW_TOKENS:-4096}"
LRB_DRY_RUN="${LRB_DRY_RUN:-0}"

if [[ "${1:-}" == "--dry-run" ]]; then
  LRB_DRY_RUN=1
  shift
fi
if (( $# > 0 )); then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-/tmp/$USER/hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "$LRB_ROOT"
mkdir -p "$LRB_RESULTS" "$HF_HOME"

echo "== node: $(hostname)  job: ${OAR_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv sync --extra inference --frozen --no-dev

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
pair_count=$(( ${#models[@]} * ${#languages[@]} ))
total_rows=$(( rows_per_language * ${#languages[@]} ))
estimated_prompts=$(( rows_per_language * pair_count ))
echo "== sizing: languages=${#languages[@]} rows=$total_rows pairs=$pair_count prompts=$estimated_prompts batch_size=$LRB_BATCH_SIZE max_new_tokens=$LRB_MAX_NEW_TOKENS"

if [[ "$LRB_DRY_RUN" == "1" ]]; then
  exit 0
fi

for model in "${models[@]}"; do
  for language in "${languages[@]}"; do
    target="$LRB_RESULTS/$language/${model//\//__}.json"
    if [[ -s "$target" ]]; then
      echo "== skip $model [$language] (result already checkpointed)"
      continue
    fi
    echo "== run $model [$language]"
    uv run --no-sync lrb run "$model" \
      --data-root "$LRB_DATA_ROOT" \
      --language "$language" \
      --prompt data/prompt.txt \
      --out "$LRB_RESULTS" \
      --batch-size "$LRB_BATCH_SIZE" \
      --max-new-tokens "$LRB_MAX_NEW_TOKENS"
  done
done

uv run --no-sync lrb report --results-dir "$LRB_RESULTS"
