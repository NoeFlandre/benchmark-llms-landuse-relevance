#!/usr/bin/env bash
# Submits the benchmark to one GPU on a Grid'5000 site frontend.
# Usage: scripts/g5k_submit.sh [walltime]   e.g. scripts/g5k_submit.sh 4:00
#
# oarsub does not forward the caller's environment, so the node-run settings below
# are passed explicitly when set:
#   LRB_ONLY, LRB_RUNTIMES, LRB_MAX_NEW_TOKENS, LRB_BATCH_SIZE, LRB_RESULTS
#   LRB_OAR_PROPERTIES  optional `oarsub -p` resource filter, e.g. a GPU model
set -euo pipefail

WALLTIME="${1:-1:00}"
LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"

usagepolicycheck -t

forward=("LRB_ROOT=$LRB_ROOT")
for var in LRB_ONLY LRB_RUNTIMES LRB_MAX_NEW_TOKENS LRB_BATCH_SIZE LRB_RESULTS; do
  if [[ -n "${!var:-}" ]]; then
    forward+=("$var=$(printf '%q' "${!var}")")
  fi
done

properties=()
if [[ -n "${LRB_OAR_PROPERTIES:-}" ]]; then
  properties=(-p "$LRB_OAR_PROPERTIES")
fi

oarsub -l "host=1/gpu=1,walltime=$WALLTIME" \
  "${properties[@]}" \
  -O "$LRB_ROOT/oar.%jobid%.out" \
  -E "$LRB_ROOT/oar.%jobid%.err" \
  "env ${forward[*]} $LRB_ROOT/scripts/g5k_node_run.sh"

oarstat -u
