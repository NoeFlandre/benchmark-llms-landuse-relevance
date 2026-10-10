---
license: mit
configs:
- config_name: default
  data_files:
  - split: train
    path: data/viewer.csv
task_categories:
- text-classification
tags:
- land-use
- land-cover
- remote-sensing
- llm-benchmark
---

# Land-use relevance benchmark

`benchmark.csv` · 2 languages x 4 items/language ·
8 items · binary `yes`/`no` labels.

[Code](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance)

## Task and prompt

Does a sentence describe a place's land or environment in ways visible to satellites?

English prompt · greedy decoding · seed 0 · `max_new_tokens=8` ·
`bfloat16` · batch 16.
`unsloth/Big-GGUF@IQ2` runs the `IQ2` GGUF quant through llama.cpp (same prompt, template, greedy decoding and budget).

### Prompt text

Replace `{}` with the target sentence.

```text
Classify the sentence.

TARGET SENTENCE: {}
```

## Aggregate scores

Per-model macro averages across languages. Per-language 95% intervals and paired tests:
[`leaderboard.csv`](leaderboard.csv); full macro metrics: [`aggregates.csv`](aggregates.csv).
Bold = best; underline = second best in each metric column.

| model_id | language_count | accuracy_macro | balanced_accuracy_macro | f1_macro | precision_macro | recall_macro | matthews_corrcoef_macro |
|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-2.6B | 2 | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** | **0.0** |
| LiquidAI/LFM2.5-VL-3B+DSpark-throughput-b16 | 2 | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** | **0.0** |
| LiquidAI/LFM2.5-VL-3B@sglang-throughput-b16 | 2 | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** | **0.0** |
| unsloth/Big-GGUF@IQ2 | 2 | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** | **0.0** |

## Scoring models

Scores are normalized to [0, 1]. Best thresholds are selected on this benchmark (an upper
bound); ROC-AUC needs no threshold. Full sweep: [`threshold_sweep.csv`](threshold_sweep.csv).

### Scoring setup

| model | handling | relevance score / decision rule | sequence length (tokens) | dtype / batch / seed | revision |
|---|---|---|---|---|---|
| LiquidAI/LFM2.5-2.6B@logprob | causal LM, no decoding: LLM prompt + chat turn, empty think block | first-token P(yes) vs P(no); argmax over the first-token yes/no log-probabilities | 8192 | bfloat16; batch 16; seed 0 | abc123 |
| LiquidAI/LFM2.5-Encoder-350M | bidirectional masked-LM encoder: task prompt + one mask; 15 supported languages | yes/no masked-token logits; argmax over the masked-token yes/no logits | 8192 | bfloat16; batch 16; seed 0 | abc123 |

### Scoring prompts

`LiquidAI/LFM2.5-Encoder-350M`:

```text
<Instruct>: judge it
<Query>: is it?
<Document>: {}
```

`LiquidAI/LFM2.5-2.6B@logprob` uses the task prompt above.

### Best thresholded scoring metrics

| model | languages | MCC @ threshold | F1 @ threshold | balanced accuracy @ threshold | precision @ threshold | recall @ threshold | ROC-AUC | items/s | peak VRAM (GiB) |
|---|---|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-2.6B@logprob | 2 | **1 @ 0.3** | **1 @ 0.3** | **1 @ 0.3** | **1 @ 0.3** | **1 @ 0** | **1** | **3.00** | **1.00** |
| LiquidAI/LFM2.5-Encoder-350M | 2 | <u>0.5774 @ 0.3</u> | <u>0.8 @ 0.3</u> | <u>0.75 @ 0.3</u> | **1 @ 0.8** | **1 @ 0** | <u>0.75</u> | <u>2.00</u> | <u>2.00</u> |

**Observed output behavior:** In these runs, LFM2.5-Encoder-350M predicted `yes` for 2 of 4 items per language across 2 languages.

### LFM2.5-2.6B: log-probabilities vs generation

Same model, prompt and languages; log-probs call yes when P(yes) > P(no).

| method | F1 | MCC | unparsed | ROC-AUC | GPU hours | ms/item |
|---|---|---|---|---|---|---|
| generation + parsing | 0.5 | 0.0 | 0.0 | n/a | 2.00 | 900000.0 |
| yes/no log-probs | 1.0 | 1.0 | 0.0 | 1 | 0.02 | 9000.0 |
| same GPU (NVIDIA A100; en, fr): generation vs log-probs | | | | | 1400 s vs 72.0 s (19x) | |

## Runtime performance

Timings are generation wall seconds summed across language runs; latency and throughput
are recomputed from prediction telemetry. Different devices and runtimes are not directly
comparable. Full per-language measurements are in `leaderboard.csv`.

| model_id | runtime | device | generation_mode | batch_size | language_count | cumulative_wall_seconds | sentences_per_second | latency_mean_seconds | latency_p50_seconds | latency_p95_seconds | generated_tokens | output_tokens_per_second | mean_accept_length | draft_accept_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-2.6B | transformers |  | static-batched | 16 | 2 | 7200.0 | 0.001 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-2.6B@logprob | transformers | NVIDIA A100 | static-batched | 16 | 2 | 72.0 | 0.111 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-Encoder-350M | transformers |  | static-batched | 16 | 2 | 2.0 | 4.0 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-VL-3B+DSpark-throughput-b16 | sglang | NVIDIA A100 | static-batched | 16 | 2 | 2.0 | 4.0 | 0.25 | 0.25 | 0.25 | 32 | 16.0 |  |  |
| LiquidAI/LFM2.5-VL-3B@sglang-throughput-b16 | sglang | NVIDIA A100 | static-batched | 16 | 2 | 2.0 | 4.0 | 0.5 | 0.5 | 0.5 | 32 | 16.0 |  |  |
| unsloth/Big-GGUF@IQ2 | transformers |  | static-batched | 16 | 2 | 2.0 | 4.0 |  |  |  |  |  |  |  |

## DSpark speculative decoding

Greedy DSpark runs should match the same-target SGLang baseline across every language;
the check requires complete item coverage and identical generated text.

| target_model | sglang_baseline | dspark_run | baseline_output_tokens_per_second | dspark_output_tokens_per_second | speedup | identical_predictions | languages_compared | items_compared | verdict_differences | text_differences |
|---|---|---|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-VL-3B | LiquidAI/LFM2.5-VL-3B@sglang-throughput-b16 | LiquidAI/LFM2.5-VL-3B+DSpark-throughput-b16 | 16.0 | 16.0 | 1.00x | yes | 2 | 8 | 0 | 0 |