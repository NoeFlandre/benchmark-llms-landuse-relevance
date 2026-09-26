# Changelog

## [0.2.0] - 2026-09-26

- Prefer exact and leading yes/no answers before the reasoning-first final-token fallback;
  store the selected parse mode and rescore saved predictions from raw generations.
- Reject run files whose metrics disagree with predictions or contain duplicate item ids.
- Refuse mixed benchmark, prompt, generation-budget, or decoding settings unless explicitly
  allowed; include per-run provenance in the dataset card.
- Require full matching coverage for speculative agreement and pin draft model revisions
  when the roster leaves them unresolved.
- Add Wilson score intervals, deterministic bootstrap intervals, and paired exact McNemar
  comparisons to results and cards.
- Add Docker runtime targets, contributor workflow, CI integration smoke tests, citation
  metadata, and historical-results documentation.
