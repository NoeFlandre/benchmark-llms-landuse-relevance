#!/usr/bin/env bash
# Runs on a reserved Grid'5000 GPU node: benchmarks every rostered model and
# checkpoints each result as soon as it is produced.
#
# Environment:
#   LRB_ROOT        project checkout on the node        (default: $HOME/benchmark-llms-landuse-relevance)
#   LRB_RESULTS     directory for run results           (default: $LRB_ROOT/results)
#   LRB_BATCH_SIZE  prompts per forward pass            (default: 16)
#   LRB_MAX_NEW_TOKENS  generation budget per prompt     (default: 1024)
#   HF_HOME         Hugging Face cache                  (default: node-local /tmp scratch)
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"   # oarsub runs a non-login shell
LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
LRB_RESULTS="${LRB_RESULTS:-$LRB_ROOT/results}"
LRB_BATCH_SIZE="${LRB_BATCH_SIZE:-16}"
LRB_MAX_NEW_TOKENS="${LRB_MAX_NEW_TOKENS:-1024}"
export HF_HOME="${HF_HOME:-/tmp/$USER/hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "$LRB_ROOT"
mkdir -p "$LRB_RESULTS" "$HF_HOME"

echo "== node: $(hostname)  job: ${OAR_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv sync --extra inference --frozen --no-dev

for model in $(uv run --no-sync lrb models | cut -f1); do
  target="$LRB_RESULTS/${model//\//__}.json"
  if [[ -s "$target" ]]; then
    echo "== skip $model (result already checkpointed)"
    continue
  fi
  echo "== run $model"
  uv run --no-sync lrb run "$model" \
    --benchmark data/benchmark.csv \
    --prompt data/prompt.txt \
    --out "$LRB_RESULTS" \
    --batch-size "$LRB_BATCH_SIZE" \
    --max-new-tokens "$LRB_MAX_NEW_TOKENS"
done

uv run --no-sync lrb report --results-dir "$LRB_RESULTS"
