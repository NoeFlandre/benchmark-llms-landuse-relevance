---
license: mit
task_categories:
- text-classification
tags:
- land-use
- land-cover
- remote-sensing
- llm-benchmark
---

# Land-use relevance benchmark

`benchmark.csv`; 154 labelled sentences; greedy decoding;
`max_new_tokens=4096`; seed 0.
Scores are recomputed from the published predictions.

- Benchmark sha256: `ac185e51835eb626c932744009873382aa80027d26e93615dbde28d23af1d5fd`; prompt sha256: `2fb48569c8bbb73fcf584fb7549b34ddd5e4cc365b7a5c75be7429906e312097`
- Code: https://github.com/NoeFlandre/benchmark-llms-landuse-relevance

## Scores

| model_id | accuracy | balanced_accuracy | f1 | precision | recall | matthews_corrcoef | unparsed_rate | truncated |
|---|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-2.6B | 0.8571 | 0.8574 | 0.8608 | 0.8718 | 0.85 | 0.7144 | 0.0 | 0 |
| Qwen/Qwen3-4B-Instruct-2507 | 0.8506 | 0.8492 | 0.8606 | 0.8353 | 0.8875 | 0.7016 | 0.0 | 0 |
| Qwen/Qwen3-8B | 0.8506 | 0.8537 | 0.8435 | 0.9254 | 0.775 | 0.7129 | 0.0 | 0 |
| google/gemma-4-E4B-it | 0.8182 | 0.8189 | 0.8205 | 0.8421 | 0.8 | 0.6374 | 0.0 | 0 |
| LiquidAI/LFM2.5-8B-A1B | 0.7727 | 0.7757 | 0.7619 | 0.8358 | 0.7 | 0.5556 | 0.0 | 0 |
| Qwen/Qwen3-4B | 0.7727 | 0.7762 | 0.7586 | 0.8462 | 0.6875 | 0.5588 | 0.0 | 0 |
| LiquidAI/LFM2.5-350M | 0.5195 | 0.5 | 0.6838 | 0.5195 | 1.0 | 0.0 | 0.0 | 0 |
| Qwen/Qwen3-0.6B | 0.5195 | 0.5 | 0.6838 | 0.5195 | 1.0 | 0.0 | 0.0 | 0 |
| LiquidAI/LFM2.5-1.2B-Instruct | 0.7208 | 0.7267 | 0.6815 | 0.8364 | 0.575 | 0.4727 | 0.0 | 0 |
| google/gemma-4-E2B-it | 0.7013 | 0.71 | 0.629 | 0.8864 | 0.4875 | 0.4644 | 0.0 | 0 |
| ibm-granite/granite-3.3-2b-instruct | 0.6364 | 0.6449 | 0.5484 | 0.7727 | 0.425 | 0.3206 | 0.0 | 0 |
| tiiuae/Falcon3-3B-Instruct | 0.6558 | 0.6687 | 0.5047 | 1.0 | 0.3375 | 0.4435 | 0.0 | 0 |
| HuggingFaceTB/SmolLM3-3B | 0.6494 | 0.6625 | 0.4906 | 1.0 | 0.325 | 0.4335 | 0.0 | 0 |
| allenai/Olmo-3-7B-Instruct | 0.6299 | 0.6432 | 0.4571 | 0.96 | 0.3 | 0.3882 | 0.0 | 0 |
| tiiuae/Falcon3-7B-Instruct | 0.5974 | 0.6125 | 0.3673 | 1.0 | 0.225 | 0.3499 | 0.0 | 0 |
| tiiuae/Falcon3-1B-Instruct | 0.513 | 0.5307 | 0.1379 | 0.8571 | 0.075 | 0.1475 | 0.0 | 0 |
| allenai/OLMo-2-1124-7B-Instruct | 0.5 | 0.5188 | 0.0723 | 1.0 | 0.0375 | 0.1356 | 0.0 | 0 |
| Qwen/Qwen3-1.7B | 0.4935 | 0.5125 | 0.0488 | 1.0 | 0.025 | 0.1103 | 0.0 | 0 |
