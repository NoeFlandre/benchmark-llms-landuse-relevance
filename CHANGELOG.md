# Changelog

## Unreleased

- Security: scoring models no longer run their Hub checkpoint's remote code unless their
  roster entry opts in. An opt-in needs a full 40-character lowercase commit SHA as the
  revision; any other revision is refused before any download. The SHA pins only the
  model repository. Code that an `auto_map` entry names in another Hub repository is
  still fetched at that repository's default branch, so a model whose `auto_map` names
  another repo is not safe to opt in until that repo is pinned too. No roster entry opts
  in yet, so `Alibaba-NLP/gte-multilingual-reranker-base` (its `auto_map` names
  `Alibaba-NLP/new-impl`) and `LiquidAI/LFM2.5-Encoder-350M` fail to load until a
  decision is made (issue #120).
- CI: pin Docker base images by digest, define `UV_VERSION` once, and enable
  Dependabot updates for the Docker ecosystem.
- Replace the obfuscated lazy imports of optional runtimes with one `require` helper that
  names the missing extra in its ImportError.
- Report an installed optional runtime that fails to import (or whose dependency is
  missing) with its original error, instead of saying the runtime is not installed.
- CLI: add help text to the shard, seed, token, `--keep-going` and `--benchmark-name`
  options, and add `--no-skip-existing` to `lrb run-all`. Option names are unchanged.
- Packaging: ship the `py.typed` marker, add project URLs, use an SPDX license
  expression, and test that the pyproject, package and CITATION.cff versions agree.
- Document the `--only`, `--runtime`, `--skip-existing`, `--no-skip-existing`,
  `--keep-going`, `--throughput` and `--json` flags in the README.
- Report a run file whose JSON top level is not an object (for example `[]`) as an
  invalid run result that names the file, instead of an `AttributeError` traceback.
- CLI: `lrb run-all --keep-going` logs the traceback of each unexpected failure, reports
  input problems as "rejected" rather than "failed", and ends with a summary of the pairs
  that did not complete.
- Grid'5000 scripts: one site-config parser serves the collector and the multisite status
  script. The status script now rejects a sites file with blank or duplicate site names,
  as the collector already did.

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
