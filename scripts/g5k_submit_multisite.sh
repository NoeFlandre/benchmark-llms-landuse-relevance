#!/usr/bin/env bash
# Submit deterministic weighted shards to explicitly configured Grid'5000 sites.
# Example: scripts/g5k_submit_multisite.sh --sites-config scripts/g5k_sites.json --dry-run
#
# Every LRB_* setting listed in scripts/g5k_forward_env.sh is written into each job
# command (LRB_ROOT/LRB_RESULTS come from LRB_REMOTE_ROOT/LRB_REMOTE_RESULTS and the
# shard variables are computed). LRB_JOB_TYPES (space-separated, default "night")
# becomes repeated -t flags and LRB_QUEUE adds -q. A site entry in the config may
# override both with "job_types" and "queue".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=g5k_forward_env.sh
source "$SCRIPT_DIR/g5k_forward_env.sh"

WALLTIME="${LRB_WALLTIME:-1:00}"
REMOTE_ROOT="${LRB_REMOTE_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
REMOTE_RESULTS="${LRB_REMOTE_RESULTS:-$REMOTE_ROOT/results-live}"
GPU_FILTER="${LRB_GPU_FILTER:-gpu_compute_capability_major>=7 AND gpu_mem>=23040}"
SITES_CONFIG="${LRB_SITES_CONFIG:-}"
DRY_RUN=0

while (( $# > 0 )); do
  case "$1" in
    --sites-config)
      [[ $# -ge 2 ]] || { echo "--sites-config needs a path" >&2; exit 2; }
      SITES_CONFIG="$2"
      shift 2
      ;;
    --walltime)
      [[ $# -ge 2 ]] || { echo "--walltime needs a value" >&2; exit 2; }
      WALLTIME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "usage: $0 --sites-config PATH [--walltime HH:MM] [--dry-run]" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SITES_CONFIG" || ! -f "$SITES_CONFIG" ]]; then
  echo "an explicit --sites-config file is required" >&2
  exit 2
fi
command -v jq >/dev/null || { echo "jq is required to read the site list" >&2; exit 2; }

site_count="$(jq -er '.sites | length' "$SITES_CONFIG")"
if [[ "$site_count" -lt 1 ]]; then
  echo "site list is empty" >&2
  exit 2
fi
total_weight=0
for (( site_index = 0; site_index < site_count; site_index++ )); do
  weight="$(jq -er ".sites[$site_index].weight" "$SITES_CONFIG")"
  if [[ ! "$weight" =~ ^[1-9][0-9]*$ ]]; then
    echo "site weight must be a positive integer: $weight" >&2
    exit 2
  fi
  total_weight=$(( total_weight + weight ))
done

echo "== multisite plan: sites=$site_count weighted_shards=$total_weight walltime=$WALLTIME"
shard_index=0
for (( site_index = 0; site_index < site_count; site_index++ )); do
  site_name="$(jq -er ".sites[$site_index].name" "$SITES_CONFIG")"
  frontend="$(jq -er ".sites[$site_index].frontend" "$SITES_CONFIG")"
  weight="$(jq -er ".sites[$site_index].weight" "$SITES_CONFIG")"
  site_job_types="$(jq -r ".sites[$site_index].job_types // empty" "$SITES_CONFIG")"
  site_queue="$(jq -r ".sites[$site_index].queue // empty" "$SITES_CONFIG")"
  type_arguments=()
  while IFS= read -r type_argument; do
    type_arguments+=("$type_argument")
  done < <(
    LRB_JOB_TYPES="${site_job_types:-${LRB_JOB_TYPES:-${LRB_JOB_TYPE:-night}}}" \
      LRB_QUEUE="${site_queue:-${LRB_QUEUE:-}}" lrb_oarsub_type_arguments
  )
  for (( site_slot = 0; site_slot < weight; site_slot++ )); do
    job_environment="$(
      LRB_ROOT="$REMOTE_ROOT" LRB_RESULTS="$REMOTE_RESULTS" \
        LRB_SHARD_INDEX="$shard_index" LRB_SHARD_COUNT="$total_weight" \
        lrb_forwarded_environment
    )"
    job_command="$job_environment$(printf '%q' "$REMOTE_ROOT/scripts/g5k_node_run.sh")"
    oarsub_command=(
      oarsub "${type_arguments[@]}" -p "$GPU_FILTER" -l "host=1/gpu=1,walltime=$WALLTIME"
      -O "$REMOTE_ROOT/oar.$site_name.%jobid%.out"
      -E "$REMOTE_ROOT/oar.$site_name.%jobid%.err"
      "$job_command"
    )
    # The frontend's login shell parses this line once; oarsub's job shell parses
    # job_command a second time, hence the two levels of printf %q.
    remote_command="cd $(printf '%q' "$REMOTE_ROOT") && $(printf '%q ' "${oarsub_command[@]}")"
    echo "[$site_name shard $shard_index/$total_weight] ssh $frontend: usagepolicycheck -t"
    echo "[$site_name shard $shard_index/$total_weight] ssh $frontend: $remote_command"
    if (( DRY_RUN == 0 )); then
      ssh "$frontend" "usagepolicycheck -t"
      ssh "$frontend" "$remote_command"
    fi
    shard_index=$(( shard_index + 1 ))
  done
done
