# benchmark-llms-landuse-relevance

Do small open-weight LLMs know when a sentence about a place says something a
satellite could see?

The active benchmark is an 85-language golden human set with 300 aligned adjudicated
items per language. Each item is labelled `yes` if it carries land-use, land-cover, or
geographic-environment signal — vegetation, water, terrain, buildings, roads, mining,
managed land — and `no` if it only concerns history, administration, people, or events.
One English prompt, one token of expected output, and the open-weight models listed in
`domain/roster.py`, scored end to end on Grid'5000 GPUs.

- **Code:** this repository
- **Results:** [NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance)
- **Docs:** `make docs`, or [the published site](https://noeflandre.github.io/benchmark-llms-landuse-relevance/)

## Quick start

```bash
uv sync --extra inference --extra scoring --extra publish
uv run lrb models                      # the roster
uv run lrb languages                   # active language inventory
uv run lrb run LiquidAI/LFM2.5-350M    # one model, all languages
uv run lrb run LiquidAI/LFM2.5-350M --language en,fr
uv run lrb run-all                     # every model
uv run lrb scorers                     # non-generative scoring roster
uv run lrb score Alibaba-NLP/gte-multilingual-reranker-base --out benchmark-runs/gte
uv run lrb score convaiinnovations/laya-multilingual --out benchmark-runs/laya
uv run lrb report                      # detailed and aggregate leaderboards
uv run lrb publish NoeFlandre/benchmark-llms-landuse-relevance
```

Without a GPU, `lrb report` and `lrb models` still work — `torch` is imported only
when a model is actually loaded.

## What gets measured

Positive class is `yes`. Generative runs report accuracy, precision, recall, F1,
balanced accuracy, Matthews correlation, and `unparsed_rate`. Scoring runs also keep
the model's normalised relevance score and native score per item, sweep thresholds, and
report the best MCC, F1, balanced accuracy, precision, recall, and ROC-AUC in
`scoring_summary.csv`.

Scoring runs record inference throughput in items per second and peak allocated CUDA
VRAM when a CUDA device is available. These values appear in the detailed
`leaderboard.csv` and the scoring summary.

The adapters preserve each checkpoint's intended interface: the Qwen rerankers score
their yes/no continuation; GTE receives a prompt/sentence pair and returns its
sequence-classification relevance logit; mxbai receives its official binary
query/document turn; Laya receives four JSON sentence fields with one typed `noul`
question per field; `LFM2.5-2.6B@logprob` reads `P(yes)` against `P(no)` from one
forward pass over the generative turn with an empty think block appended; the NLI
models, GLiClass and GLiNER2 score the hypothesis in `data/prompt_zeroshot.txt`. Laya
uses the checkpoint's 1,024-token context and SDK-selected runtime dtype; other scoring
sequence lengths are recorded per run.

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

The active data inventory is `data/translations/manifest.json`. Reports read only
language-nested checkpoints and produce detailed per-language rows, model-level macro
aggregates, `threshold_sweep.csv`, and `scoring_summary.csv`. The published Hub folder
also contains `data/train.csv`, a single viewer-friendly multilingual split.

## Models

| model | parameters |
|---|---|
| `LiquidAI/LFM2.5-350M` | 0.35B |
| `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2B |
| `LiquidAI/LFM2.5-2.6B` | 2.7B |
| `LiquidAI/LFM2.5-8B-A1B` | 8.5B total, ~1B active (MoE) |
| `HuggingFaceTB/SmolLM3-3B` | ~3B |
| `allenai/OLMo-2-1124-7B-Instruct` | ~7B |
| `ibm-granite/granite-3.3-2b-instruct` | ~2B |
| `tiiuae/Falcon3-1B-Instruct` | ~1B |
| `tiiuae/Falcon3-3B-Instruct` | ~3B |
| `tiiuae/Falcon3-7B-Instruct` | ~7B |
| `Qwen/Qwen3-0.6B` | ~0.6B |
| `Qwen/Qwen3-1.7B` | ~1.7B |
| `Qwen/Qwen3-4B` | ~4B |
| `Qwen/Qwen3-8B` | ~8B |
| `Qwen/Qwen3-4B-Instruct-2507` | ~4B |
| `allenai/Olmo-3-7B-Instruct` | ~7B |
| `google/gemma-4-E2B-it` | ~2B |
| `google/gemma-4-E4B-it` | ~4B |
| `unsloth/Qwen3.8-27B-GGUF@UD-IQ2_XXS` | 27B at a ~2-bit GGUF quant (7.3 GB), llama.cpp |

### Scoring models

| model | parameters | scoring rule |
|---|---:|---|
| `Alibaba-NLP/gte-multilingual-reranker-base` | ~0.306B | sigmoid sequence-classification relevance logit |
| `mixedbread-ai/mxbai-rerank-base-v2` | ~0.5B | official binary 1/0 logit-difference normalisation |
| `convaiinnovations/laya-multilingual` | ~0.322B | typed `noul` yes probability |
| `Qwen/Qwen3-Reranker-0.6B` | ~0.596B | yes/no next-token argmax |
| `Qwen/Qwen3-Reranker-4B` | ~4.022B | yes/no next-token argmax |
| `LiquidAI/LFM2.5-2.6B@logprob` | ~2.7B | first-token yes/no log-probs, one forward pass |
| `knowledgator/gliclass-multilang-mini` | ~0.284B | GLiClass label probability |
| `MoritzLaurer/bge-m3-zeroshot-v2.0` | ~0.568B | NLI entailment probability |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | ~0.279B | NLI entailment probability |
| `BalaRajesh1/mmbert-small-nli` | ~0.141B | NLI entailment probability |
| `fastino/gliner2.5-multi-v1` | ~0.287B | GLiNER2 label confidence |

`LiquidAI/LFM2.5-2.6B@logprob` sends the generative prompt and chat turn, closes the
template's opening `<think>` in the input, and reads P(yes) vs P(no) at the next
position from a single forward pass — no token is generated. The NLI, GLiClass and
GLiNER2 models use their own zero-shot APIs with the sentence as input and one
hypothesis taken from the LLM prompt (`data/prompt_zeroshot.txt`). The GGUF quant runs
through llama.cpp with the generative prompt, template, greedy decoding and budget, and
records its quant label in `quantization` rather than `dtype`.

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
