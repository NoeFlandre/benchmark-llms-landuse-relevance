#!/usr/bin/env bash
# Runs on a reserved Grid'5000 GPU node: benchmarks every rostered run and
# checkpoints each result as soon as it is produced.
#
# Two runtimes, two environments: SGLang pins its own torch and transformers, so the
# Transformers runs go first under the `inference` extra, then the environment is
# re-synced with the `speculative` extra for the SGLang runs (plain and DSpark).
#
# Environment:
#   LRB_ROOT        project checkout on the node        (default: $HOME/benchmark-llms-landuse-relevance)
#   LRB_RESULTS     directory for run results           (default: $LRB_ROOT/results)
#   LRB_BATCH_SIZE  prompts per forward pass            (default: each run's roster setting)
#   LRB_MAX_NEW_TOKENS  generation budget per prompt     (default: 4096)
#   LRB_RUNTIMES    runtimes to run, in order           (default: "transformers sglang")
#   LRB_ONLY        regex; run only matching run names  (default: every run)
#   HF_HOME         Hugging Face cache                  (default: node-local /tmp scratch)
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"   # oarsub runs a non-login shell
LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
LRB_RESULTS="${LRB_RESULTS:-$LRB_ROOT/results}"
LRB_BATCH_SIZE="${LRB_BATCH_SIZE:-}"
LRB_MAX_NEW_TOKENS="${LRB_MAX_NEW_TOKENS:-4096}"
LRB_RUNTIMES="${LRB_RUNTIMES:-transformers sglang}"
LRB_ONLY="${LRB_ONLY:-.}"
export HF_HOME="${HF_HOME:-/tmp/$USER/hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "$LRB_ROOT"
mkdir -p "$LRB_RESULTS" "$HF_HOME"

echo "== node: $(hostname)  job: ${OAR_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

batch_args=()
if [[ -n "$LRB_BATCH_SIZE" ]]; then
  batch_args=(--batch-size "$LRB_BATCH_SIZE")
fi

for runtime in $LRB_RUNTIMES; do
  case "$runtime" in
    transformers) extra=inference ;;
    sglang) extra=speculative ;;
    *) echo "unknown runtime $runtime" >&2; exit 2 ;;
  esac
  echo "== environment: $runtime (extra: $extra)"
  uv sync --extra "$extra" --frozen --no-dev

  uv run --no-sync lrb models | awk -F'\t' -v r="$runtime" '$3 == r { print $1 }' \
    | grep -E "$LRB_ONLY" | while read -r name; do
    target="$LRB_RESULTS/${name//\//__}.json"
    if [[ -s "$target" ]]; then
      echo "== skip $name (result already checkpointed)"
      continue
    fi
    echo "== run $name"
    uv run --no-sync lrb run "$name" \
      --benchmark data/benchmark.csv \
      --prompt data/prompt.txt \
      --out "$LRB_RESULTS" \
      --max-new-tokens "$LRB_MAX_NEW_TOKENS" \
      "${batch_args[@]}"
  done
done

# A same-runtime DSpark mismatch makes `report` exit non-zero; keep the job's
# results either way and surface the failure in the OAR log.
uv run --no-sync lrb report --results-dir "$LRB_RESULTS"
