# benchmark-llms-landuse-relevance

Do small open-weight LLMs know when a sentence about a place says something a
satellite could see?

154 adjudicated sentences from Wikipedia articles and institutional websites, each
labelled `yes` if it carries land-use, land-cover, or geographic-environment signal —
vegetation, water, terrain, buildings, roads, mining, managed land — and `no` if it
only concerns history, administration, people, or events. One prompt, one token of
expected output, five Liquid AI LFM2.5 models — four of them also run with a DSpark
speculative-decoding draft — scored and timed end to end on a Grid'5000 GPU.

- **Code:** this repository
- **Results:** [NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance)
- **Docs:** `make docs`, or [the published site](https://noeflandre.github.io/benchmark-llms-landuse-relevance/)

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

## What gets measured

Positive class is `yes`. Each run reports accuracy, precision, recall, F1, balanced
accuracy, Matthews correlation, and `unparsed_rate`, and alongside them how fast the
answers came: wall time, sentences/s, per-sentence latency (mean, p50, p95), generated
tokens and output tokens/s, and for speculative runs the mean accept length and draft
accept rate ([ADR-0006](docs/adr/0006-speed-measurement.md)).

Decoding is greedy, so a run replays exactly. The verdict is the last standalone
`yes`/`no` in the generation — two of these models open with an analysis preamble that
restates the prompt's own rubric, and reading from the front scores that restatement as
the answer. A generation that exhausted its token budget without stopping carries no
verdict at all, whatever words appear in it. Either failure is counted as an error
rather than dropped or coerced, and the raw text is kept, so another convention can be
recomputed from the published results without re-running the models. See
[ADR-0002](docs/adr/0002-unparsed-as-error.md) and
[ADR-0005](docs/adr/0005-generation-budget.md).

The published benchmark uses a 4096-token generation budget so models have enough room
to reach their yes/no verdict without turning the score into a test of output length.

Every result file pins the model revision, the prompt sha256, the benchmark sha256, the
decoding settings, the seed, and the source commit.

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

## Licence

MIT. The benchmark sentences are quoted from their cited sources; `source_url` is
carried on every row.
