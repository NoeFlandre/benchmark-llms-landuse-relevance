# shellcheck shell=bash
# Shared by g5k_submit.sh and g5k_submit_multisite.sh. oarsub does not export the
# submitting shell's environment, so every setting g5k_node_run.sh reads must be
# written into the job command itself.

LRB_FORWARDED_VARIABLES=(
  LRB_ROOT
  LRB_RESULTS
  LRB_MODEL_ID
  LRB_LANGUAGES
  LRB_SHARD_INDEX
  LRB_SHARD_COUNT
  LRB_BATCH_SIZE
  LRB_MAX_NEW_TOKENS
  LRB_DATA_ROOT
  LRB_SCORER_PROMPT
  LRB_EXPECTED_ROWS
  LRB_CUDA_MODULE
)

# Prints `NAME=value ` for every forwarded variable that is set and non-empty, each value quoted
# with printf %q so spaces and shell metacharacters survive the job shell.
lrb_forwarded_environment() {
  local name
  for name in "${LRB_FORWARDED_VARIABLES[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      printf '%s=%q ' "$name" "${!name}"
    fi
  done
}

# Prints one `-t TYPE` pair per word of LRB_JOB_TYPES (default: LRB_JOB_TYPE, else
# night), then `-q QUEUE` when LRB_QUEUE is set; one argument per line. Lyon A100s,
# for instance, need LRB_JOB_TYPES="exotic" (no night queue there).
lrb_oarsub_type_arguments() {
  local job_type
  for job_type in ${LRB_JOB_TYPES:-${LRB_JOB_TYPE:-night}}; do
    printf '%s\n' -t "$job_type"
  done
  if [[ -n "${LRB_QUEUE:-}" ]]; then
    printf '%s\n' -q "$LRB_QUEUE"
  fi
}
