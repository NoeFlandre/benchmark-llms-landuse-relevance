# Results

Published to
[NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance)
and its [Hugging Face bucket](https://huggingface.co/buckets/NoeFlandre/benchmark-llms-landuse-relevance).
All published models are in `standard/models/`, with their aggregate scores in
`standard/leaderboard.csv`. The public release uses a 4096-token generation budget so
the benchmark measures classification rather than output-length compliance.

In this repository, the four JSON files directly under `results/` are the current
committed Transformers runs. The dated `more-models-20260913/`, `qwen3-20260913/`, and
`recent-models-20260913/` folders preserve 14 earlier model runs; they are historical
snapshots, not the current CLI roster. `lrb report` reads direct files only. The committed
root `README.md` card and `leaderboard.csv` summarize all result JSON files recursively,
matching `lrb publish`'s recursive discovery.

Runs are comparable only when benchmark hash, prompt hash, token budget, and decoding
match. `report` and `publish` stop on a mismatch and identify the offending run; use
`--allow-mixed` only when the card's per-run settings make the distinction clear.

## Reading a run file

Each run file holds metadata needed to audit it — model revision, prompt and benchmark
sha256, decoding settings, seed, host duration, source commit, and package version — then
every prediction with its raw generation and `parse_mode`, followed by metrics derived
from exactly those predictions.

`generation_mode` distinguishes static Transformers batching, Transformers continuous
batching, and SGLang throughput runs. The performance variants use separate run names,
so they do not overwrite the existing baselines. Reproduce them with, for example:

```bash
uv run lrb run LiquidAI/LFM2.5-2.6B --continuous-batching --batch-size 154
uv run lrb run LiquidAI/LFM2.5-2.6B@sglang --throughput --batch-size 8
```

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
the same generations. SGLang and continuous-batching rows report per-request latency
when available; static batched Transformers rows report batch-call latency.

The leaderboard adds Wilson 95% intervals for accuracy, precision and recall; seeded
bootstrap intervals for F1 and Matthews correlation; and exact paired McNemar p-values
against the top-F1 run when full coverage is identical.

The dataset card is a deterministic table of every published run. `lrb publish`
discovers result JSONs recursively and recomputes each row from its predictions.
