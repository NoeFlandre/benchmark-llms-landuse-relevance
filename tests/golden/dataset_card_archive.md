---
license: mit
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.csv
task_categories:
- text-classification
tags:
- land-use
- land-cover
- remote-sensing
- llm-benchmark
---

# Land-use relevance benchmark

`benchmark.csv` · 1 language x 154 items/language ·
154 items · binary `yes`/`no` labels.

[Code](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance)

Package version recorded in run metadata: `0.1.0`.

## Task and prompt

Does a sentence describe a place's land or environment in ways visible to satellites?

English prompt · greedy decoding · seed 0 · `max_new_tokens=4096` ·
`bfloat16` · batch 16.

### Prompt text

Replace `{}` with the target sentence.

```text
Classify whether the TARGET SENTENCE contains information about the target place that could help characterize its land use, land cover, or geographic environment from remote sensing, either directly or through observable proxies.

Return exactly one token: yes or no.

Answer yes for information about vegetation, agriculture, forests, water, soil or surface, terrain, buildings, settlements, infrastructure, transport networks, mining, managed land, or other human or natural features with a spatial or remotely detectable signature.

Answer no for information only about history, administration, people, events, demographics, economy, navigation, or activities with no meaningful land-use, land-cover, or remotely detectable implication.

Output only the lowercase token yes or no.

TARGET SENTENCE: {}
```

## Aggregate scores

Per-model macro averages across languages. Per-language 95% intervals and paired tests:
[`leaderboard.csv`](leaderboard.csv); full macro metrics: [`aggregates.csv`](aggregates.csv).
Bold = best; underline = second best in each metric column.

| model_id | language_count | accuracy_macro | balanced_accuracy_macro | f1_macro | precision_macro | recall_macro | matthews_corrcoef_macro |
|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-2.6B | 1 | **0.8571** | **0.8574** | **0.8608** | 0.8718 | 0.85 | **0.7144** |
| Qwen/Qwen3-4B-Instruct-2507 | 1 | <u>0.8506</u> | 0.8492 | <u>0.8606</u> | 0.8353 | <u>0.8875</u> | 0.7016 |
| Qwen/Qwen3-8B | 1 | <u>0.8506</u> | <u>0.8537</u> | 0.8435 | 0.9254 | 0.775 | <u>0.7129</u> |
| google/gemma-4-E4B-it | 1 | 0.8182 | 0.8189 | 0.8205 | 0.8421 | 0.8 | 0.6374 |
| LiquidAI/LFM2.5-8B-A1B | 1 | 0.7727 | 0.7757 | 0.7619 | 0.8358 | 0.7 | 0.5556 |
| Qwen/Qwen3-4B | 1 | 0.7727 | 0.7762 | 0.7586 | 0.8462 | 0.6875 | 0.5588 |
| LiquidAI/LFM2.5-350M | 1 | 0.5195 | 0.5 | 0.6838 | 0.5195 | **1.0** | 0.0 |
| Qwen/Qwen3-0.6B | 1 | 0.5195 | 0.5 | 0.6838 | 0.5195 | **1.0** | 0.0 |
| LiquidAI/LFM2.5-1.2B-Instruct | 1 | 0.7208 | 0.7267 | 0.6815 | 0.8364 | 0.575 | 0.4727 |
| google/gemma-4-E2B-it | 1 | 0.7013 | 0.71 | 0.629 | 0.8864 | 0.4875 | 0.4644 |
| ibm-granite/granite-3.3-2b-instruct | 1 | 0.6364 | 0.6449 | 0.5484 | 0.7727 | 0.425 | 0.3206 |
| tiiuae/Falcon3-3B-Instruct | 1 | 0.6558 | 0.6687 | 0.5047 | **1.0** | 0.3375 | 0.4435 |
| HuggingFaceTB/SmolLM3-3B | 1 | 0.6494 | 0.6625 | 0.4906 | **1.0** | 0.325 | 0.4335 |
| allenai/Olmo-3-7B-Instruct | 1 | 0.6299 | 0.6432 | 0.4571 | <u>0.96</u> | 0.3 | 0.3882 |
| tiiuae/Falcon3-7B-Instruct | 1 | 0.5974 | 0.6125 | 0.3673 | **1.0** | 0.225 | 0.3499 |
| tiiuae/Falcon3-1B-Instruct | 1 | 0.513 | 0.5307 | 0.1379 | 0.8571 | 0.075 | 0.1475 |
| allenai/OLMo-2-1124-7B-Instruct | 1 | 0.5 | 0.5188 | 0.0723 | **1.0** | 0.0375 | 0.1356 |
| Qwen/Qwen3-1.7B | 1 | 0.4935 | 0.5125 | 0.0488 | **1.0** | 0.025 | 0.1103 |

## Runtime performance

Timings are generation wall seconds summed across language runs; latency and throughput
are recomputed from prediction telemetry. Different devices and runtimes are not directly
comparable. Full per-language measurements are in `leaderboard.csv`.

| model_id | runtime | device | generation_mode | batch_size | language_count | cumulative_wall_seconds | sentences_per_second | latency_mean_seconds | latency_p50_seconds | latency_p95_seconds | generated_tokens | output_tokens_per_second | mean_accept_length | draft_accept_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM3-3B | transformers |  | static-batched | 16 | 1 | 19.39 | 7.944 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-1.2B-Instruct | transformers |  | static-batched | 16 | 1 | 2.21 | 69.715 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-2.6B | transformers |  | static-batched | 16 | 1 | 507.49 | 0.303 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-350M | transformers |  | static-batched | 16 | 1 | 1.58 | 97.222 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-8B-A1B | transformers |  | static-batched | 16 | 1 | 853.21 | 0.18 |  |  |  |  |  |  |  |
| Qwen/Qwen3-0.6B | transformers |  | static-batched | 16 | 1 | 2.91 | 52.848 |  |  |  |  |  |  |  |
| Qwen/Qwen3-1.7B | transformers |  | static-batched | 16 | 1 | 3.12 | 49.327 |  |  |  |  |  |  |  |
| Qwen/Qwen3-4B | transformers |  | static-batched | 16 | 1 | 5.47 | 28.174 |  |  |  |  |  |  |  |
| Qwen/Qwen3-4B-Instruct-2507 | transformers |  | static-batched | 16 | 1 | 5.91 | 26.053 |  |  |  |  |  |  |  |
| Qwen/Qwen3-8B | transformers |  | static-batched | 16 | 1 | 7.82 | 19.683 |  |  |  |  |  |  |  |
| allenai/OLMo-2-1124-7B-Instruct | transformers |  | static-batched | 16 | 1 | 7.73 | 19.912 |  |  |  |  |  |  |  |
| allenai/Olmo-3-7B-Instruct | transformers |  | static-batched | 16 | 1 | 8.6 | 17.907 |  |  |  |  |  |  |  |
| google/gemma-4-E2B-it | transformers |  | static-batched | 16 | 1 | 8.78 | 17.534 |  |  |  |  |  |  |  |
| google/gemma-4-E4B-it | transformers |  | static-batched | 16 | 1 | 5.98 | 25.744 |  |  |  |  |  |  |  |
| ibm-granite/granite-3.3-2b-instruct | transformers |  | static-batched | 16 | 1 | 4.97 | 30.98 |  |  |  |  |  |  |  |
| tiiuae/Falcon3-1B-Instruct | transformers |  | static-batched | 16 | 1 | 3.69 | 41.723 |  |  |  |  |  |  |  |
| tiiuae/Falcon3-3B-Instruct | transformers |  | static-batched | 16 | 1 | 3.64 | 42.296 |  |  |  |  |  |  |  |
| tiiuae/Falcon3-7B-Instruct | transformers |  | static-batched | 16 | 1 | 7.0 | 21.987 |  |  |  |  |  |  |  |