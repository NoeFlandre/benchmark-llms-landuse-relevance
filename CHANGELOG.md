# Changelog

## Unreleased

- CI: pin Docker base images by digest, define `UV_VERSION` once, and enable
  Dependabot updates for the Docker ecosystem.
- CLI: add help text to the shard, seed, token, `--keep-going` and `--benchmark-name`
  options, and add `--no-skip-existing` to `lrb run-all`. Option names are unchanged.
- Document the `--only`, `--runtime`, `--skip-existing`, `--no-skip-existing`,
  `--keep-going`, `--throughput` and `--json` flags in the README.

## [0.2.0] - 2026-09-26

### Scoring changes

- Prefer exact and leading yes/no answers before the reasoning-first final-token
  fallback. Store the selected parse mode. Rescore the saved predictions from the raw
  generations.
- Reject run files in which the metrics do not agree with the predictions. Reject run
  files that contain duplicate item ids.
- Refuse mixed benchmark, prompt, generation-budget, or decoding settings, unless the
  user allows them. Put the provenance of each run in the dataset card.
- Add Wilson score intervals, deterministic bootstrap intervals, and paired exact
  McNemar comparisons to the results and cards.

### Other changes

- Require full matching coverage for speculative agreement. Pin the draft model
  revisions when the roster does not resolve them.
- Add Docker runtime targets, a contributor workflow, CI integration smoke tests,
  citation metadata, and documentation of the historical results.
