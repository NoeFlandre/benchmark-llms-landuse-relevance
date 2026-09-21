# Results

Published runs belong to the `v3-multilingual` benchmark set in
[NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance).
Each model-language checkpoint is explicit, and the public release uses a 4096-token
generation budget so the benchmark measures classification rather than output-length
compliance.

## Reading a run file

Each run file holds the metadata needed to audit it — language, model revision, prompt
and benchmark sha256, decoding settings, seed, host duration, and the source commit —
then every prediction with its raw generation, and finally the metrics computed from
exactly those predictions.

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

The dataset card is a deterministic model-level aggregate table. It includes one row
per model, with `language_count` showing how many language checkpoints contributed;
`lrb publish` discovers result JSONs recursively and recomputes every aggregate from
their predictions.
