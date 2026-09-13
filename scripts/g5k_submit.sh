#!/usr/bin/env bash
# Submits the benchmark to one GPU on a Grid'5000 site frontend.
# Usage: scripts/g5k_submit.sh [walltime]   e.g. scripts/g5k_submit.sh 1:30
set -euo pipefail

WALLTIME="${1:-1:00}"
LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"

usagepolicycheck -t

oarsub -l "host=1/gpu=1,walltime=$WALLTIME" \
  -O "$LRB_ROOT/oar.%jobid%.out" \
  -E "$LRB_ROOT/oar.%jobid%.err" \
  "$LRB_ROOT/scripts/g5k_node_run.sh"

oarstat -u
