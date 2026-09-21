#!/usr/bin/env bash
# Submits one deterministic shard to one GPU on a Grid'5000 site frontend.
# Usage: scripts/g5k_submit.sh [walltime] [--dry-run]
set -euo pipefail

WALLTIME="1:00"
DRY_RUN=0
for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=1 ;;
    *) WALLTIME="$argument" ;;
  esac
done
LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
LRB_RESULTS="${LRB_RESULTS:-$LRB_ROOT/results-live}"
LRB_MODEL_ID="${LRB_MODEL_ID:-}"
LRB_GPU_FILTER="${LRB_GPU_FILTER:-gpu_compute_capability_major>=7 AND gpu_mem>=23040}"
LRB_JOB_TYPE="${LRB_JOB_TYPE:-night}"
LRB_SHARD_INDEX="${LRB_SHARD_INDEX:-0}"
LRB_SHARD_COUNT="${LRB_SHARD_COUNT:-1}"

echo "usagepolicycheck -t"
echo "oarsub type=$LRB_JOB_TYPE host=1/gpu=1 walltime=$WALLTIME shard=$LRB_SHARD_INDEX/$LRB_SHARD_COUNT model=${LRB_MODEL_ID:-all}"

if (( DRY_RUN == 1 )); then
  exit 0
fi

usagepolicycheck -t

oarsub -t "$LRB_JOB_TYPE" -p "$LRB_GPU_FILTER" -l "host=1/gpu=1,walltime=$WALLTIME" \
  -O "$LRB_ROOT/oar.%jobid%.out" \
  -E "$LRB_ROOT/oar.%jobid%.err" \
  "LRB_ROOT='$LRB_ROOT' LRB_MODEL_ID='$LRB_MODEL_ID' LRB_SHARD_INDEX=$LRB_SHARD_INDEX LRB_SHARD_COUNT=$LRB_SHARD_COUNT LRB_RESULTS='$LRB_RESULTS' $LRB_ROOT/scripts/g5k_node_run.sh"

oarstat -u
