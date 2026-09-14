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

# Land-use relevance: small-LLM benchmark results

Predictions and scores for small open-weight LLMs asked to judge whether a sentence
about a place carries land-use, land-cover, or geographic-environment signal that
could be observed by remote sensing.

- Benchmark: `benchmark.csv` (154 labelled sentences)
- Benchmark sha256: `ac185e51835eb626c932744009873382aa80027d26e93615dbde28d23af1d5fd`
- Prompt sha256: `2fb48569c8bbb73fcf584fb7549b34ddd5e4cc365b7a5c75be7429906e312097`
- Decoding: greedy, `max_new_tokens=4096`, seed 0
- Code: https://github.com/NoeFlandre/benchmark-llms-landuse-relevance

## Leaderboard

| model_id | accuracy | balanced_accuracy | f1 | precision | recall | matthews_corrcoef | unparsed_rate | truncated |
|---|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-2.6B | 0.8571 | 0.8574 | 0.8608 | 0.8718 | 0.85 | 0.7144 | 0.0 | 0 |
| LiquidAI/LFM2.5-8B-A1B | 0.7727 | 0.7757 | 0.7619 | 0.8358 | 0.7 | 0.5556 | 0.0 | 0 |
| LiquidAI/LFM2.5-350M | 0.5195 | 0.5 | 0.6838 | 0.5195 | 1.0 | 0.0 | 0.0 | 0 |
| LiquidAI/LFM2.5-1.2B-Instruct | 0.7208 | 0.7267 | 0.6815 | 0.8364 | 0.575 | 0.4727 | 0.0 | 0 |

## Companion configurations

The same benchmark and prompt, run under different settings:

- `strict-8-tokens/`

## Files

- `<namespace>__<model>.json` — one file per model: run metadata, every raw generation,
  the parsed verdict, and the scores computed from exactly those predictions.
- `leaderboard.csv` — the table above, with columns model_id, n_items, accuracy, balanced_accuracy, f1, precision, recall, matthews_corrcoef, unparsed_rate, truncated, duration_seconds, model_revision.

Generations the model did not express as `yes`/`no` are counted as errors, never dropped.
The verdict is the last standalone `yes`/`no` in a generation that stopped on its own; a
generation that exhausted its token budget carries no verdict at all, and `truncated`
counts those.
