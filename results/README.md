# Active results

Active result files are written to `results/<language>/<model>.json`. Each file
contains one complete model-language run and carries the benchmark language in
its metadata and every leaderboard row.

Use `uv run lrb report` to read active result files and write:

- `results/leaderboard.csv` — one detailed row per model-language pair;
- `results/aggregates.csv` — model-level macro metrics and F1 spread across
  languages;
- `results/threshold_sweep.csv` — the threshold grid for every scoring
  model-language pair;
- `results/scoring_summary.csv` — one best-operating-point row per scoring model.

The `results/archive/` subtree contains historical provenance only. Active
readers and publication refuse to consume it.
