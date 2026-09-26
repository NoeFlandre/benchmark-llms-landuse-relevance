# benchmark-llms-landuse-relevance

Do small open-weight LLMs know when a sentence about a place says something a
satellite could see?

154 adjudicated sentences from Wikipedia articles and institutional websites, each
labelled `yes` if it carries land-use, land-cover, or geographic-environment signal —
vegetation, water, terrain, buildings, roads, mining, managed land — and `no` if it
only concerns history, administration, people, or events. One prompt, one token of
expected output, five Liquid AI LFM2.5 models across 13 generation/runtime
configurations — including four DSpark speculative-decoding runs — scored and timed end
to end on a Grid'5000 GPU.

- **Code:** this repository
- **Results:** [NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance)
- **Docs:** `make docs`, or [the published site](https://noeflandre.github.io/benchmark-llms-landuse-relevance/)
- **Citation:** [CITATION.cff](CITATION.cff)

## Quick start

```bash
uv sync --extra inference --extra publish
uv run lrb models                      # the roster
uv run lrb run LiquidAI/LFM2.5-350M    # one model
uv sync --extra speculative            # SGLang runtime (conflicts with `inference`)
uv run lrb run LiquidAI/LFM2.5-VL-3B+DSpark
uv run lrb run-all                     # every model
uv run lrb report                      # leaderboard from stored runs
uv run lrb publish NoeFlandre/benchmark-llms-landuse-relevance
```

Without a GPU, `lrb report` and `lrb models` still work — `torch` is imported only
when a model is actually loaded.

### Scripting the CLI

- `lrb --version` prints the package version; `lrb --install-completion` sets up shell
  completion. Every command's `--help` ends with examples.
- `--json` on `models`, `run` and `report` prints machine-readable output, e.g.
  `lrb report --json | python -m json.tool`.
- `-v` logs model loading and per-batch progress to stderr (`-vv` for debug detail);
  `-q` shows errors only. Warnings are shown by default.
- `run-all` filters and resumes: `--only REGEX`, `--runtime transformers|sglang`
  (repeatable), `--skip-existing` (skip runs whose result file is already written) and
  `--keep-going` (continue past a failed run, exit 1 at the end). `lrb models --runtime`
  filters the roster the same way.
- `lrb publish REPO --dry-run` prints the target repo, its visibility, the files that
  would be uploaded and the generated card, without calling the Hub or writing a file.
  `--commit-message` sets the upload's message. A missing `publish` extra or a Hub
  auth/network error exits 1 with a one-line message instead of a traceback.
- `run` and `run-all` accept `--results-dir` as an alias of `--out`, matching `report`
  and `publish`.
- `run` can opt into `--continuous-batching` for text-only Transformers models or
  `--throughput` for multi-request SGLang runs. Give continuous batching a
  `--batch-size` equal to the worklist size to submit the full benchmark together;
  SGLang throughput mode requires `--batch-size` greater than one. Each variant has a
  separate run name and records its generation mode.

## What gets measured

Positive class is `yes`. Each run reports accuracy, precision, recall, F1, balanced
accuracy, Matthews correlation, and `unparsed_rate`, and alongside them how fast the
answers came: wall time, sentences/s, per-sentence latency (mean, p50, p95), generated
tokens and output tokens/s, and for speculative runs the mean accept length and draft
accept rate ([ADR-0006](docs/adr/0006-speed-measurement.md)). Accuracy, precision and
recall include Wilson 95% intervals; F1 and Matthews correlation include seeded bootstrap
intervals. Paired exact McNemar p-values compare each run with the top-F1 run when both
cover the same full benchmark.

Decoding is greedy, so a run replays exactly. Verdict parsing prefers an exact answer,
then an answer at the start, then the last standalone `yes`/`no` for reasoning-first
outputs. Each prediction records which rule was used. A generation that exhausted its
token budget without stopping carries no verdict, whatever words appear in it. Raw text
is retained, so scores can be recomputed without loading the models. See
[ADR-0002](docs/adr/0002-unparsed-as-error.md),
[ADR-0008](docs/adr/0008-verdict-parsing.md), and
[ADR-0005](docs/adr/0005-generation-budget.md).

The published benchmark uses a 4096-token generation budget so models have enough room
to reach their yes/no verdict without turning the score into a test of output length.

Every result file pins the model revision, the prompt sha256, the benchmark sha256, the
decoding settings, the seed, the source commit, and the package version.

## Saved results

The current CLI roster contains 13 generation/runtime configurations. The root of
`results/` currently stores four current Transformers runs; the dated subfolders retain
14 earlier model runs as historical snapshots. `lrb report` reads the root files only,
while the committed `results/README.md` card and `results/leaderboard.csv` summarize all
run JSON files recursively. Use separate directories for different benchmark settings;
`report` and `publish` refuse mixed benchmark, prompt, budget or decoding settings unless
`--allow-mixed` is supplied. See [the results guide](docs/results.md).

## Models

| run | runtime | parameters |
|---|---|---|
| `LiquidAI/LFM2.5-350M` | Transformers | 0.35B |
| `LiquidAI/LFM2.5-1.2B-Instruct` | Transformers | 1.2B |
| `LiquidAI/LFM2.5-2.6B` | Transformers | 2.7B |
| `LiquidAI/LFM2.5-8B-A1B` | Transformers | 8.5B total, ~1B active (MoE) |
| `LiquidAI/LFM2.5-VL-3B` | Transformers, text-only prompts | 3.1B |
| `<target>@sglang` for the four targets below | SGLang, no draft | as the target |
| `LiquidAI/LFM2.5-1.2B-Instruct+DSpark` | SGLang + `LFM2.5-1.2B-Instruct-DSpark` | 1.2B + 0.30B draft |
| `LiquidAI/LFM2.5-2.6B+DSpark` | SGLang + `LFM2.5-2.6B-DSpark` | 2.7B + 0.33B draft |
| `LiquidAI/LFM2.5-8B-A1B+DSpark` | SGLang + `LFM2.5-8B-A1B-DSpark` | 8.5B + 0.33B draft |
| `LiquidAI/LFM2.5-VL-3B+DSpark` | SGLang + `LFM2.5-VL-3B-DSpark` | 3.1B + 0.28B draft |

Every run pins its weights (and its draft's) to a commit. Each `+DSpark` run follows
its model card's SGLang recipe; `@sglang` is the same launch without the speculative
flags. Under greedy decoding a draft cannot change the output, so `lrb report` checks
each `+DSpark` run against its `@sglang` baseline item by item and fails if they
disagree ([ADR-0007](docs/adr/0007-dspark-speculative-decoding.md)).

## Grid'5000

```bash
ssh nancy
git clone https://github.com/NoeFlandre/benchmark-llms-landuse-relevance.git
cd benchmark-llms-landuse-relevance
usagepolicycheck -t
scripts/g5k_submit.sh 4:00
```

The node script checkpoints each model's result as it finishes and skips models already
done, so an overrun job is resumed rather than repeated. See
[docs/grid5000.md](docs/grid5000.md).

## Docker

The default image contains the Transformers runtime. Build the separate CUDA-backed
SGLang image with `--target sglang`:

```bash
docker build -t landuse-relevance-bench .
docker build --target sglang -t landuse-relevance-bench:sglang .
docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$PWD/results:/app/results" \
  landuse-relevance-bench run LiquidAI/LFM2.5-350M
```

For authenticated model access, pass `HF_TOKEN` at container runtime (for example,
`-e HF_TOKEN`); never put credentials in a Docker build argument.

## Development

```bash
make check     # every CI gate: ruff → ty → unit+property (coverage floor) → acceptance →
               # architecture → CRAP → mutation → smoke → docs → lockfile → pip-audit
make mutation  # mutation testing over the domain alone
make docker    # reproducible runtime image
```

Three layers — `cli → adapters → domain` — enforced by `import-linter` and by a test
that fails the build if a domain module reaches for the filesystem or a model runtime.
Everything above the unit level substitutes a scripted generator through the one-method
`TextGenerator` protocol, which is why the acceptance suite runs in seconds without a
GPU. See [docs/architecture.md](docs/architecture.md) and
[docs/debt.md](docs/debt.md) for the known weaknesses.

## Licence and source attribution

The software is MIT licensed. The benchmark sentences retain the licenses and reuse
terms of their original sources; `source_url` is carried on every row for attribution.
Check the upstream terms before reusing or redistributing source text. The software
license does not relicense those quotations.
