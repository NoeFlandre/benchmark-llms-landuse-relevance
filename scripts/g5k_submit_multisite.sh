#!/usr/bin/env bash
# Submit deterministic weighted shards to explicitly configured Grid'5000 sites.
# Example: scripts/g5k_submit_multisite.sh --sites-config scripts/g5k_sites.json --dry-run
set -euo pipefail

WALLTIME="${LRB_WALLTIME:-1:00}"
REMOTE_ROOT="${LRB_REMOTE_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
REMOTE_RESULTS="${LRB_REMOTE_RESULTS:-$REMOTE_ROOT/results-live}"
GPU_FILTER="${LRB_GPU_FILTER:-gpu_compute_capability_major>=7 AND gpu_mem>=23040}"
JOB_TYPE="${LRB_JOB_TYPE:-night}"
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

echo "== multisite plan: sites=$site_count weighted_shards=$total_weight walltime=$WALLTIME type=$JOB_TYPE"
shard_index=0
for (( site_index = 0; site_index < site_count; site_index++ )); do
  site_name="$(jq -er ".sites[$site_index].name" "$SITES_CONFIG")"
  frontend="$(jq -er ".sites[$site_index].frontend" "$SITES_CONFIG")"
  weight="$(jq -er ".sites[$site_index].weight" "$SITES_CONFIG")"
  for (( site_slot = 0; site_slot < weight; site_slot++ )); do
    remote_command="cd '$REMOTE_ROOT' && LRB_ROOT='$REMOTE_ROOT' LRB_SHARD_INDEX=$shard_index LRB_SHARD_COUNT=$total_weight LRB_RESULTS='$REMOTE_RESULTS' LRB_JOB_TYPE='$JOB_TYPE' oarsub -t '$JOB_TYPE' -p '$GPU_FILTER' -l 'host=1/gpu=1,walltime=$WALLTIME' -O '$REMOTE_ROOT/oar.$site_name.%jobid%.out' -E '$REMOTE_ROOT/oar.$site_name.%jobid%.err' '$REMOTE_ROOT/scripts/g5k_node_run.sh'"
    echo "[$site_name shard $shard_index/$total_weight] ssh $frontend: usagepolicycheck -t"
    echo "[$site_name shard $shard_index/$total_weight] ssh $frontend: $remote_command"
    if (( DRY_RUN == 0 )); then
      ssh "$frontend" "usagepolicycheck -t"
      ssh "$frontend" "$remote_command"
    fi
    shard_index=$(( shard_index + 1 ))
  done
done
