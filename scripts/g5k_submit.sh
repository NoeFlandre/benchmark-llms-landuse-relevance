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
ARRAY_COUNT=""
POSITIONAL_WALLTIME=0
while (( $# > 0 )); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --array)
      [[ $# -ge 2 ]] || { echo "--array needs a positive integer" >&2; exit 2; }
      ARRAY_COUNT="$2"
      shift 2
      ;;
    *)
      if (( POSITIONAL_WALLTIME == 1 )); then
        echo "usage: $0 [walltime] [--array COUNT] [--dry-run]" >&2
        exit 2
      fi
      WALLTIME="$1"
      POSITIONAL_WALLTIME=1
      shift
      ;;
  esac
done
if [[ -n "$ARRAY_COUNT" && ! "$ARRAY_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "--array must be a positive integer" >&2
  exit 2
fi
LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
LRB_RESULTS="${LRB_RESULTS:-$LRB_ROOT/results-live}"
LRB_GPU_FILTER="${LRB_GPU_FILTER:-gpu_compute_capability_major>=7 AND gpu_mem>=23040}"
LRB_SHARD_COUNT="${LRB_SHARD_COUNT:-${ARRAY_COUNT:-1}}"
if [[ -n "$ARRAY_COUNT" ]]; then
  if [[ -n "${LRB_SHARD_INDEX:-}" ]]; then
    echo "--array cannot be combined with an explicit LRB_SHARD_INDEX" >&2
    exit 2
  fi
  if [[ "$LRB_SHARD_COUNT" != "$ARRAY_COUNT" ]]; then
    echo "LRB_SHARD_COUNT must match --array COUNT" >&2
    exit 2
  fi
fi

type_arguments=()
while IFS= read -r type_argument; do
  type_arguments+=("$type_argument")
done < <(lrb_oarsub_type_arguments)
job_command="$(lrb_forwarded_environment)$(printf '%q' "$LRB_ROOT/scripts/g5k_node_run.sh")"
oarsub_command=(
  oarsub "${type_arguments[@]}" -p "$LRB_GPU_FILTER" -l "host=1/gpu=1,walltime=$WALLTIME"
  -O "$LRB_ROOT/oar.%jobid%.out"
  -E "$LRB_ROOT/oar.%jobid%.err"
)
if [[ -n "$ARRAY_COUNT" ]]; then
  oarsub_command+=(--array "$ARRAY_COUNT")
fi
oarsub_command+=("$job_command")

echo "usagepolicycheck -t"
if [[ -n "$ARRAY_COUNT" ]]; then
  echo "shard=OAR_ARRAY_INDEX/$LRB_SHARD_COUNT model=${LRB_MODEL_ID:-all}"
else
  echo "shard=${LRB_SHARD_INDEX:-0}/$LRB_SHARD_COUNT model=${LRB_MODEL_ID:-all}"
fi
printf '%q ' "${oarsub_command[@]}"
printf '\n'

if (( DRY_RUN == 1 )); then
  exit 0
fi

usagepolicycheck -t
"${oarsub_command[@]}"
oarstat -u
