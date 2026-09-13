# Results

Published to
[NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance)
as a Hugging Face dataset: one JSON per model plus `leaderboard.csv`.

Two configurations are published, because they answer different questions.

| folder | budget | question it answers |
|---|---|---|
| (root) | 1024 tokens | Does the model know the answer? |
| `strict-8-tokens/` | 8 tokens | Does the model obey "output only the token"? |

The two compliant models score identically under both. The two that open with an
analysis preamble score zero under the strict budget — see
[ADR-0005](adr/0005-generation-budget.md) for why both numbers are kept.

## Reading a run file

Each run file holds the metadata needed to audit it — model revision, prompt and
benchmark sha256, decoding settings, seed, host duration, and the source commit — then
every prediction with its raw generation, and finally the metrics computed from exactly
those predictions.

Each prediction carries `truncated`. A truncated generation is scored unparsed however
many verdict words appear in it: the model ran out of budget mid-thought and never
committed. `unparsed_rate` therefore covers both "never answered" and "was cut off",
and the flag says which.

## Reproducing a number

```bash
uv run lrb run LiquidAI/LFM2.5-2.6B --revision <revision from the run file>
uv run lrb report
```

Decoding is greedy and every input is pinned by digest, so the same revision reproduces
the same generations.

The live leaderboard is the dataset card; it is regenerated from the run files by
`lrb publish`, so it cannot drift from the predictions it summarises.
