# benchmark-llms-landuse-relevance

Do small open-weight LLMs know when a sentence about a place describes something that a
satellite can see?

The active benchmark is an 85-language golden human set. It has 300 aligned adjudicated
items for each language. An item has the label `yes` if it has land-use, land-cover, or
geographic-environment signal. Examples are vegetation, water, terrain, buildings,
roads, mining, and managed land. An item has the label `no` if it is only about
history, administration, people, or events.

The benchmark uses one English prompt and one token of expected output. It tests the
open-weight models in `domain/roster.py`. It scores them end to end on Grid'5000 GPUs.

- **Code:** this repository
- **Results:** [NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance)
- **Documentation:** run `make docs`, or read [the published site](https://noeflandre.github.io/benchmark-llms-landuse-relevance/)
- **Glossary:** [docs/glossary.md](docs/glossary.md)

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

You do not need a GPU for `lrb report` and `lrb models`. The code imports `torch` only
when it loads a model.

### Command-line flags

Run `lrb <command> --help` for the complete list. These flags are not in the quick
start.

| Flag | Commands | Effect |
| --- | --- | --- |
| `--only REGEX` | `run-all` | Run only the roster run names that match the regular expression. |
| `--runtime NAME` | `run-all` | Restrict the roster to `transformers` or `sglang`. Repeat the flag to select both. |
| `--skip-existing` / `--no-skip-existing` | `run-all` | Skip a model-language pair that already has a stored result (default), or run it again. |
| `--keep-going` | `run-all` | Continue after a failed run. The command prints each failure to standard error and exits with status 1 at the end. Without this flag, the first failure stops the command. |
| `--throughput` | `run`, `run-all` | Use the SGLang multi-request throughput mode. |
| `--json` | `models` | Print the roster as JSON. |

```bash
uv run lrb run-all --runtime transformers --only 'LFM2' --keep-going
uv run lrb models --json
```

`run-all` fails with a usage error when `--only` and `--runtime` match no rostered run.

## What the benchmark measures

The positive class is `yes`. A generative run reports accuracy, precision, recall, F1,
balanced accuracy, Matthews correlation, and `unparsed_rate`.

A scoring run keeps the normalised relevance score and the native score of the model for
each item. It sweeps thresholds. It reports the best MCC, F1, balanced accuracy,
precision, recall, and ROC-AUC in `scoring_summary.csv`.

A scoring run records the inference throughput in items per second. It also records the
peak allocated CUDA VRAM when a CUDA device is available. These values are in the
detailed `leaderboard.csv` and in the scoring summary.

Each adapter keeps the interface that the checkpoint was made for:

- The Qwen rerankers score their yes/no continuation.
- GTE receives a prompt/sentence pair. It returns its sequence-classification relevance
  logit.
- mxbai receives its official binary query/document turn.
- Laya receives four JSON sentence fields. It receives one typed `noul` question for
  each field.
- `LFM2.5-2.6B@logprob` reads `P(yes)` against `P(no)` from one forward pass. The
  forward pass uses the generative turn with an empty think block at the end.
- The NLI models, GLiClass, and GLiNER2 score the hypothesis in `data/prompt_zeroshot.txt`.

Laya uses the context of 1,024 tokens of the checkpoint and the runtime dtype that the
SDK selects. The run records the other scoring sequence lengths.

Decoding is greedy, so a run replays exactly. The verdict is the last standalone
`yes`/`no` in the generation. Two of these models start with an analysis preamble. The
preamble repeats the rubric of the prompt. If the parser reads from the start, it scores
this repeated text as the answer.

A generation that used the complete token budget without a stop has no verdict. This is
true for all the words in the generation. The benchmark counts each of these two failures
as an error. It does not drop the item and it does not force a label. It keeps the raw
text. Therefore, you can recompute another convention from the published results. You
do not need to run the models again. Refer to
[ADR-0002](docs/adr/0002-unparsed-as-error.md) and
[ADR-0005](docs/adr/0005-generation-budget.md).

The published benchmark uses a generation budget of 4096 tokens. This gives the models
enough room to reach their yes/no verdict. The score then does not test the length of
the output.

Each result file pins the model revision, the prompt sha256, the benchmark sha256, the
decoding settings, the seed, and the source commit.

## Data

The active inventory is [`data/translations/manifest.json`](data/translations/manifest.json).
It has 85 language configurations. Each configuration has 300 aligned adjudicated rows.
The Hub release gives the same records as one `data/train.csv` split for the viewer.

## Outputs

The benchmark saves each completed model-language pair as a checkpoint in
`results/<language>/<run-name>.json`. The command `lrb report` makes these files from
the predictions: the `leaderboard.csv` of each language, the model-level macro
`aggregates.csv`, `threshold_sweep.csv`, and `scoring_summary.csv`. The earlier
single-language files are in `results/archive/`. The active reports do not include
them.

You can move both locations without flags on each command. `LRB_DATA_DIR` (default
`data`) sets the directory from which the code reads `translations/`, `prompt.txt`, and
`prompt_reranker.txt`. `LRB_RESULTS_DIR` (default `results`) sets the directory to which
the code writes runs and from which it reads them. The explicit options `--data-root`,
`--prompt`, and `--out`/`--results-dir` have priority.

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

`LiquidAI/LFM2.5-2.6B@logprob` sends the generative prompt and the chat turn. It closes
the opening `<think>` of the template in the input. It reads P(yes) against P(no) at the
next position from one forward pass. The model does not generate a token.

The NLI, GLiClass, and GLiNER2 models use their own zero-shot APIs. The input is the
sentence. They use one hypothesis from the LLM prompt (`data/prompt_zeroshot.txt`).

The GGUF quant runs through llama.cpp. It uses the generative prompt, the template,
greedy decoding, and the budget. It records its quant label in `quantization`, not in
`dtype`.

`LiquidAI/LFM2.5-Encoder-350M` is a bidirectional masked-LM. The benchmark scores it
separately. Its single `[MASK]` position ranks the `yes` and `no` tokens. The results
cover only the 15 languages that the model card supports.

For each `+DSpark` run, `lrb report` compares the generated text and the verdicts with
the `@sglang` baseline of the same target. The comparison must show identical text and
verdicts under greedy decoding. The card and `leaderboard.csv` also include the recorded
speed and confidence summaries.

## Grid'5000

```bash
ssh nancy
git clone https://github.com/NoeFlandre/benchmark-llms-landuse-relevance.git
cd benchmark-llms-landuse-relevance
usagepolicycheck -t
 scripts/g5k_submit.sh
```

The node script saves the result of each model as a checkpoint when the model is
complete. It skips the models that are complete. Therefore, you resume a job that
exceeded its walltime. You do not repeat it. Refer to [docs/grid5000.md](docs/grid5000.md).

## Development

```bash
make check     # ruff → ty → unit → property → acceptance → architecture → CRAP
make mutation  # mutation testing over the domain
make docker    # reproducible runtime image
```

The code has three layers: `cli → adapters → domain`. `import-linter` enforces them. A
test fails the build if a domain module uses the filesystem or a model runtime.

All tests above the unit level replace the generator with a scripted generator. They use
the `TextGenerator` protocol, which has one method. For this reason, the acceptance
suite runs in seconds without a GPU. Refer to [docs/architecture.md](docs/architecture.md)
and [docs/debt.md](docs/debt.md) for the known weaknesses.

## Docker

The default image uses Transformers. Build it with the current commit in the run
metadata. Then keep the model weights and the results in Docker-managed volumes:

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

For the separate CUDA/SGLang runtime, build `--target sglang`. Select an `@sglang` model
id, for example `LiquidAI/LFM2.5-2.6B@sglang`. SGLang needs an NVIDIA driver and the
NVIDIA Container Toolkit on the host.

WARNING: Do not pass credentials as build arguments.

To publish, make `HF_TOKEN` available in the shell. Pass it at runtime with `-e HF_TOKEN`:

```bash
docker run --rm -e HF_TOKEN \
  -v lrb-results:/app/results \
  -v lrb-hf-cache:/cache/huggingface \
  landuse-relevance-bench:transformers publish NoeFlandre/benchmark-llms-landuse-relevance
```

## Citation

Cite the software release that [`CITATION.cff`](CITATION.cff) describes.

## Licence

MIT. The benchmark sentences are quotes from the sources that they cite. Each row has
its `source_url`.
