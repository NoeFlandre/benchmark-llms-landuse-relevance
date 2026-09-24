#!/usr/bin/env bash
# Compatibility wrapper for the active multilingual benchmark. The active release
# has one published generation budget and one nested checkpoint tree.
set -euo pipefail

LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
HEADLINE_TOKENS="${HEADLINE_TOKENS:-4096}"

LRB_MAX_NEW_TOKENS="$HEADLINE_TOKENS" \
LRB_RESULTS="$LRB_ROOT/results" \
  "$LRB_ROOT/scripts/g5k_node_run.sh"
