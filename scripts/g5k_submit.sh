#!/usr/bin/env bash
# Submits one deterministic shard to one GPU on a Grid'5000 site frontend.
# Usage: scripts/g5k_submit.sh [walltime] [--dry-run]
#
# Every LRB_* setting listed in scripts/g5k_forward_env.sh is written into the job
# command. LRB_JOB_TYPES (space-separated, default "night") becomes repeated -t
# flags; LRB_QUEUE adds -q; LRB_GPU_FILTER overrides the resource filter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=g5k_forward_env.sh
source "$SCRIPT_DIR/g5k_forward_env.sh"

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
LRB_GPU_FILTER="${LRB_GPU_FILTER:-gpu_compute_capability_major>=7 AND gpu_mem>=23040}"
LRB_SHARD_INDEX="${LRB_SHARD_INDEX:-0}"
LRB_SHARD_COUNT="${LRB_SHARD_COUNT:-1}"

type_arguments=()
while IFS= read -r type_argument; do
  type_arguments+=("$type_argument")
done < <(lrb_oarsub_type_arguments)
job_command="$(lrb_forwarded_environment)$(printf '%q' "$LRB_ROOT/scripts/g5k_node_run.sh")"
oarsub_command=(
  oarsub "${type_arguments[@]}" -p "$LRB_GPU_FILTER" -l "host=1/gpu=1,walltime=$WALLTIME"
  -O "$LRB_ROOT/oar.%jobid%.out"
  -E "$LRB_ROOT/oar.%jobid%.err"
  "$job_command"
)

echo "usagepolicycheck -t"
echo "shard=$LRB_SHARD_INDEX/$LRB_SHARD_COUNT model=${LRB_MODEL_ID:-all}"
printf '%q ' "${oarsub_command[@]}"
printf '\n'

if (( DRY_RUN == 1 )); then
  exit 0
fi

usagepolicycheck -t
"${oarsub_command[@]}"
oarstat -u
