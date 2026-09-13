# benchmark-llms-landuse-relevance

Do small open-weight LLMs know when a sentence about a place says something a
satellite could see?

154 adjudicated sentences from Wikipedia articles and institutional websites, each
labelled `yes` if it carries land-use, land-cover, or geographic-environment signal —
vegetation, water, terrain, buildings, roads, mining, managed land — and `no` if it
only concerns history, administration, people, or events. One prompt, one token of
expected output, four Liquid AI LFM2.5 models, scored end to end on a Grid'5000 GPU.

- **Code:** this repository
- **Results:** [NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance)
- **Docs:** `make docs`, or [the published site](https://noeflandre.github.io/benchmark-llms-landuse-relevance/)

## Quick start

```bash
uv sync --all-extras
uv run lrb models                      # the roster
uv run lrb run LiquidAI/LFM2.5-350M    # one model
uv run lrb run-all                     # every model
uv run lrb report                      # leaderboard from stored runs
uv run lrb publish NoeFlandre/benchmark-llms-landuse-relevance
```

Without a GPU, `lrb report` and `lrb models` still work — `torch` is imported only
when a model is actually loaded.

## What gets measured

Positive class is `yes`. Each run reports accuracy, precision, recall, F1, balanced
accuracy, Matthews correlation, and `unparsed_rate`.

Decoding is greedy, so a run replays exactly. The verdict is the last standalone
`yes`/`no` in the generation — two of these models open with an analysis preamble that
restates the prompt's own rubric, and reading from the front scores that restatement as
the answer. A generation that exhausted its token budget without stopping carries no
verdict at all, whatever words appear in it. Either failure is counted as an error
rather than dropped or coerced, and the raw text is kept, so another convention can be
recomputed from the published results without re-running the models. See
[ADR-0002](docs/adr/0002-unparsed-as-error.md) and
[ADR-0005](docs/adr/0005-generation-budget.md).

Two configurations are published: the headline run at a 1024-token budget, which asks
whether the model knows the answer, and a strict eight-token run under
`strict-8-tokens/`, which asks whether it obeys "output only the token".

Every result file pins the model revision, the prompt sha256, the benchmark sha256, the
decoding settings, the seed, and the source commit.

## Models

| model | parameters |
|---|---|
| `LiquidAI/LFM2.5-350M` | 0.35B |
| `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2B |
| `LiquidAI/LFM2.5-2.6B` | 2.7B |
| `LiquidAI/LFM2.5-8B-A1B` | 8.5B total, ~1B active (MoE) |

## Grid'5000

```bash
ssh nancy
git clone https://github.com/NoeFlandre/benchmark-llms-landuse-relevance.git
cd benchmark-llms-landuse-relevance
usagepolicycheck -t
scripts/g5k_submit.sh 1:00
```

The node script checkpoints each model's result as it finishes and skips models already
done, so an overrun job is resumed rather than repeated. See
[docs/grid5000.md](docs/grid5000.md).

## Development

```bash
make check     # ruff → ty → unit → property → acceptance → architecture → CRAP
make mutation  # mutation testing over the domain
make docker    # reproducible runtime image
```

Three layers — `cli → adapters → domain` — enforced by `import-linter` and by a test
that fails the build if a domain module reaches for the filesystem or a model runtime.
Everything above the unit level substitutes a scripted generator through the one-method
`TextGenerator` protocol, which is why the acceptance suite runs in seconds without a
GPU. See [docs/architecture.md](docs/architecture.md) and
[docs/debt.md](docs/debt.md) for the known weaknesses.

## Licence

MIT. The benchmark sentences are quoted from their cited sources; `source_url` is
carried on every row.
