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

`benchmark.csv` · 1 language x 4 items/language ·
4 items · binary `yes`/`no` labels.

[Code](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance)

## Task and prompt

Does a sentence describe a place's land or environment in ways visible to satellites?

English prompt · greedy decoding · seed 0 · `max_new_tokens=8` ·
`bfloat16` · batch 16.

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
| gen/one | 1 | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** | **0.0** |
| gen/two | 1 | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** | **0.0** |

## Scoring models

Scores are normalized to [0, 1]. Best thresholds are selected on this benchmark (an upper
bound); ROC-AUC needs no threshold. Full sweep: [`threshold_sweep.csv`](threshold_sweep.csv).

### Scoring setup

| model | handling | relevance score / decision rule | sequence length (tokens) | dtype / batch / seed | revision |
|---|---|---|---|---|---|
| score/best | scoring adapter: recorded prompt + sentence | normalized yes score; argmax over the native yes/no scores | 8192 | bfloat16; batch 16; seed 0 | abc123 |
| score/second | scoring adapter: recorded prompt + sentence | normalized yes score; argmax over the native yes/no scores | 8192 | bfloat16; batch 16; seed 0 | abc123 |

### Scoring prompts

`score/best`, `score/second`:

```text
<Instruct>: judge it
<Query>: is it?
<Document>: {}
```

### Best thresholded scoring metrics

| model | languages | MCC @ threshold | F1 @ threshold | balanced accuracy @ threshold | precision @ threshold | recall @ threshold | ROC-AUC | items/s | peak VRAM (GiB) |
|---|---|---|---|---|---|---|---|---|---|
| score/best | 1 | **1 @ 0.3** | **1 @ 0.3** | **1 @ 0.3** | **1 @ 0.3** | **1 @ 0** | **1** | **3.00** | **1.00** |
| score/second | 1 | <u>0.5774 @ 0.3</u> | <u>0.8 @ 0.3</u> | <u>0.75 @ 0.3</u> | **1 @ 0.8** | **1 @ 0** | <u>0.75</u> | <u>2.00</u> | <u>2.00</u> |

## Runtime performance

Timings are generation wall seconds summed across language runs; latency and throughput
are recomputed from prediction telemetry. Different devices and runtimes are not directly
comparable. Full per-language measurements are in `leaderboard.csv`.

| model_id | runtime | device | generation_mode | batch_size | language_count | cumulative_wall_seconds | sentences_per_second | latency_mean_seconds | latency_p50_seconds | latency_p95_seconds | generated_tokens | output_tokens_per_second | mean_accept_length | draft_accept_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen/one | transformers |  | static-batched | 16 | 1 | 1.0 | 4.0 |  |  |  |  |  |  |  |
| gen/two | transformers |  | static-batched | 16 | 1 | 1.0 | 4.0 |  |  |  |  |  |  |  |
| score/best | transformers |  | static-batched | 16 | 1 | 1.0 | 4.0 |  |  |  |  |  |  |  |
| score/second | transformers |  | static-batched | 16 | 1 | 1.0 | 4.0 |  |  |  |  |  |  |  |