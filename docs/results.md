# Results

Published to
[NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance)
as a Hugging Face dataset: one JSON per model plus `leaderboard.csv`.

Each run file holds the full metadata needed to audit it — model revision, prompt and
benchmark sha256, decoding settings, seed, host duration, and the source commit — then
every prediction with its raw generation, and finally the metrics computed from exactly
those predictions.

To reproduce a published number:

```bash
uv run lrb run LiquidAI/LFM2.5-350M --revision <revision from the run file>
uv run lrb report
```

The live leaderboard is the dataset card; it is regenerated from the run files by
`lrb publish`, so it cannot drift from the predictions it summarises.
