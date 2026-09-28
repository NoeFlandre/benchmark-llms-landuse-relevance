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

## Data

The active inventory is [`data/translations/manifest.json`](data/translations/manifest.json):
85 language configurations with 300 aligned, adjudicated rows each. The Hub release
provides the same records as a single viewer-friendly `data/train.csv` split.

## Outputs

Each completed model-language pair is checkpointed as
`results/<language>/<run-name>.json`. `lrb report` derives per-language
`leaderboard.csv`, model-level macro `aggregates.csv`, `threshold_sweep.csv`, and
`scoring_summary.csv` from those predictions. Earlier single-language files are kept
under `results/archive/` and are excluded from active reports.

Both locations can be moved without passing flags to every command. `LRB_DATA_DIR`
(default `data`) sets where `translations/`, `prompt.txt` and `prompt_reranker.txt` are
read from. `LRB_RESULTS_DIR` (default `results`) sets where runs are written and read.
Explicit `--data-root`, `--prompt` and `--out`/`--results-dir` options still win.

## Models

| model | parameters |
|---|---|
| `LiquidAI/LFM2.5-350M` | 0.35B |
| `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2B |
| `LiquidAI/LFM2.5-2.6B` | 2.7B |
| `LiquidAI/LFM2.5-8B-A1B` | 8.5B total, ~1B active (MoE) |
| `LiquidAI/LFM2.5-VL-3B` | 3.1B; text-only benchmark prompts |
| `LiquidAI/LFM2.5-1.2B-Instruct@sglang` | 1.2B; SGLang baseline |
| `LiquidAI/LFM2.5-1.2B-Instruct+DSpark` | 1.2B + 0.30B draft |
| `LiquidAI/LFM2.5-2.6B@sglang` | 2.7B; SGLang baseline |
| `LiquidAI/LFM2.5-2.6B+DSpark` | 2.7B + 0.33B draft |
| `LiquidAI/LFM2.5-8B-A1B@sglang` | 8.5B total; SGLang baseline |
| `LiquidAI/LFM2.5-8B-A1B+DSpark` | 8.5B total + 0.33B draft |
| `LiquidAI/LFM2.5-VL-3B@sglang` | 3.1B; SGLang baseline |
| `LiquidAI/LFM2.5-VL-3B+DSpark` | 3.1B + 0.28B draft |
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
| `LiquidAI/LFM2.5-Encoder-350M` | ~0.355B | yes/no masked-token logits (15 languages) |

`LiquidAI/LFM2.5-2.6B@logprob` sends the generative prompt and chat turn, closes the
template's opening `<think>` in the input, and reads P(yes) vs P(no) at the next
position from a single forward pass — no token is generated. The NLI, GLiClass and
GLiNER2 models use their own zero-shot APIs with the sentence as input and one
hypothesis taken from the LLM prompt (`data/prompt_zeroshot.txt`). The GGUF quant runs
through llama.cpp with the generative prompt, template, greedy decoding and budget, and
records its quant label in `quantization` rather than `dtype`.

`LiquidAI/LFM2.5-Encoder-350M` is scored separately as a bidirectional masked-LM: its
single `[MASK]` position ranks the `yes` and `no` tokens. Results cover only the model
card's 15 supported languages.

For each `+DSpark` run, `lrb report` checks identical generated text and verdicts
against the same-target `@sglang` baseline under greedy decoding. The card and
`leaderboard.csv` also include recorded speed and confidence summaries.

## Grid'5000

```bash
ssh nancy
git clone https://github.com/NoeFlandre/benchmark-llms-landuse-relevance.git
cd benchmark-llms-landuse-relevance
usagepolicycheck -t
 scripts/g5k_submit.sh
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

## Docker

The default image uses Transformers. Build it with the current commit recorded in run
metadata, then keep model weights and results in Docker-managed volumes:

```bash
docker build --build-arg LRB_SOURCE_COMMIT="$(git rev-parse HEAD)" \
  --target transformers -t landuse-relevance-bench:transformers .
docker volume create lrb-results
docker volume create lrb-hf-cache
docker run --rm --gpus all \
  -v lrb-results:/app/results \
  -v lrb-hf-cache:/cache/huggingface \
  landuse-relevance-bench:transformers run LiquidAI/LFM2.5-1.2B-Instruct --language en
docker run --rm -v lrb-results:/app/results \
  landuse-relevance-bench:transformers report --results-dir /app/results
```

For the separate CUDA/SGLang runtime, build `--target sglang` and select an `@sglang`
model id such as `LiquidAI/LFM2.5-2.6B@sglang`. SGLang needs an NVIDIA driver and the
NVIDIA Container Toolkit on the host. To publish, make `HF_TOKEN` available in the
shell and pass it at runtime with `-e HF_TOKEN`; never pass credentials as build args:

```bash
docker run --rm -e HF_TOKEN \
  -v lrb-results:/app/results \
  -v lrb-hf-cache:/cache/huggingface \
  landuse-relevance-bench:transformers publish NoeFlandre/benchmark-llms-landuse-relevance
```

## Citation

Please cite the software release described in [`CITATION.cff`](CITATION.cff).

## Licence

MIT. The benchmark sentences are quoted from their cited sources; `source_url` is
carried on every row.
