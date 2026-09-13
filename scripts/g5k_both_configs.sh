#!/usr/bin/env bash
# Runs both published configurations in one GPU reservation:
#   results/                  the headline run, budget large enough to let a model finish
#   results/strict-8-tokens/  the output-contract run, eight tokens
# See docs/adr/0005-generation-budget.md for why both are published.
set -euo pipefail

LRB_ROOT="${LRB_ROOT:-$HOME/benchmark-llms-landuse-relevance}"
HEADLINE_TOKENS="${HEADLINE_TOKENS:-4096}"

LRB_MAX_NEW_TOKENS=8 \
LRB_RESULTS="$LRB_ROOT/results/strict-8-tokens" \
  "$LRB_ROOT/scripts/g5k_node_run.sh"

LRB_MAX_NEW_TOKENS="$HEADLINE_TOKENS" \
LRB_RESULTS="$LRB_ROOT/results" \
  "$LRB_ROOT/scripts/g5k_node_run.sh"
