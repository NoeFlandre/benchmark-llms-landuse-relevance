# Active results

The tool writes active result files to `results/<language>/<model>.json`. Each
file contains one complete model-language run. The file has the benchmark
language in its metadata and in each leaderboard row.

To read the active result files, run `uv run lrb report`. The command writes
these files:

- `results/leaderboard.csv`: one detailed row for each model-language pair.
- `results/aggregates.csv`: macro metrics for each model, and the F1 spread
  across languages.
- `results/threshold_sweep.csv`: the threshold grid for each scoring
  model-language pair.
- `results/scoring_summary.csv`: one best-operating-point row for each scoring
  model.

The `results/archive/` subtree contains historical provenance only. Active
readers and publication refuse to use it.
