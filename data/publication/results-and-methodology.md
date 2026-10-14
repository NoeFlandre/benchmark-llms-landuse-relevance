# Results, methods, and provenance

This document records the full task prompt, native decision rules, scorer revisions, tuned-score protocol, d1-3B precision caveat, and runtime observations. The concise all-model summary is in [`../README.md`](../README.md). <!-- LRB:PACKAGE_VERSION_NOTE --> For machine-readable values, use [`../aggregates.csv`](../aggregates.csv) for native macro metrics, [`../leaderboard.csv`](../leaderboard.csv) for per-language results and revisions, and [`../scoring_summary.csv`](../scoring_summary.csv) plus [`../threshold_sweep.csv`](../threshold_sweep.csv) for tuned scoring results. [`../SNAPSHOT_STATUS.md`](../SNAPSHOT_STATUS.md) records coverage and completion.

The 85 language editions translate the same 300 source sentences. Treat them as aligned
language results, not as 25,500 independent source examples; macro metrics average across
languages.

Current scoring runs store unrounded label scores in the JSON `label_scores` field; the
readable `raw_output` string remains rounded to six decimals. Legacy result files without
`label_scores` remain readable and use their saved text scores for threshold sweeps.

## Task and prompt

Does a sentence describe a place's land or environment in ways visible to satellites?

English prompt · greedy decoding · seed 0 · `max_new_tokens=4096` ·
`bfloat16` · batch varies by model.
`unsloth/Qwen3.8-27B-GGUF@UD-IQ2_XXS` runs the `UD-IQ2_XXS` GGUF quant through llama.cpp (same prompt, template, greedy decoding and budget).

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

## Scoring models

Scores are normalized to [0, 1]. Best thresholds are selected on this benchmark (an upper
bound); ROC-AUC needs no threshold. Full sweep: [`threshold_sweep.csv`](../threshold_sweep.csv).

### Scoring setup

| model | handling | relevance score / decision rule | sequence length (tokens) | dtype / batch / seed | revision |
|---|---|---|---|---|---|
| Alibaba-NLP/gte-multilingual-reranker-base | sequence classifier: prompt + sentence pair | sigmoid relevance logit; yes if score ≥ 0.5 | 8192 | bfloat16; batch 16; seed 0 | 8215cf04918ba6f7b6a62bb44238ce2953d8831c |
| BalaRajesh1/mmbert-small-nli | NLI zero-shot pipeline: sentence premise + hypothesis | entailment probability; yes if score ≥ 0.5 | 512 | bfloat16; batch 16; seed 0 | 2e7a7a1b86760ec00c5596f86e5b8bcb9ff8d9fb |
| LiquidAI/LFM2.5-2.6B@logprob | causal LM, no decoding: LLM prompt + chat turn, empty think block | first-token P(yes) vs P(no); argmax over the first-token yes/no log-probabilities | 8192 | bfloat16; batch 16; seed 0 | 654f9463ce32b05d0429d76fe1f580b27d4c1ac0 |
| LiquidAI/LFM2.5-Encoder-350M | bidirectional masked-LM encoder: task prompt + one mask; 15 supported languages | yes/no masked-token logits; argmax over the masked-token yes/no logits | 8192 | bfloat16; batch 16; seed 0 | b886781f7c6f10ca9b7096e21b83e30a073c2f39 |
| MoritzLaurer/bge-m3-zeroshot-v2.0 | NLI zero-shot pipeline: sentence premise + hypothesis | entailment probability; yes if score ≥ 0.5 | 512 | bfloat16; batch 16; seed 0 | 9abf1c8aaeb82a2447809c20753ed0b106b76652 |
| MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | NLI zero-shot pipeline: sentence premise + hypothesis | entailment probability; yes if score ≥ 0.5 | 512 | bfloat16; batch 16; seed 0 | b5113eb38ab63efdd7f280f8c144ea8b13f978ce |
| Qwen/Qwen3-Reranker-0.6B | causal-LM reranker: manual yes/no reranker turn | yes/no next-token probability; argmax over the native yes/no scores | model-defined | bfloat16; batch 16; seed 0 | e61197ed45024b0ed8a2d74b80b4d909f1255473 |
| Qwen/Qwen3-Reranker-4B | causal-LM reranker: manual yes/no reranker turn | yes/no next-token probability; argmax over the native yes/no scores | model-defined | bfloat16; batch 16; seed 0 | 22e683669bc0f0bd69640a1354a6d0aebcfeede5 |
| convaiinnovations/laya-multilingual | typed decision model: JSON state + 4 `noul` questions/call | Laya `noul` yes probability; yes if score ≥ 0.5 | 1024 | varies across runs (dtype is recorded per run); batch 16; seed 0 | b4a904d1a2a54c822b829e24291d4b8f280fe43e |
| fastino/gliner2.5-multi-v1 | GLiNER2 classify_text: sentence + hypothesis label | label confidence; yes if score ≥ 0.5 | model-defined | float32; batch 16; seed 0 | a221b77a8baf4a613b8f8652661d41fa10a5641e |
| knowledgator/gliclass-multilang-mini | GLiClass zero-shot: sentence + hypothesis label | label probability; yes if score ≥ 0.5 | 512 | bfloat16; batch 16; seed 0 | 0bd888b6c3ef9fca5f0a9d407bddfbbc7623486b |
| mixedbread-ai/mxbai-rerank-base-v2 | causal-LM reranker: official query/document turn | sigmoid(1-logit - 0-logit - 4.5); yes if score ≥ 0.5 | 8192 | bfloat16; batch 16; seed 0 | 3ea9d4dffa7d12a4f366be8e275c349de9fc9865 |
| LiquidAI/d1-3B | official AutoModel `system_one_batch`; one typed `noul` question per sentence | argmax of unrounded P(yes) and 1−P(yes); exact ties select no (yes iff P(yes) > 0.5) | 32768 | bfloat16; batch 16; seed 0 | 051bcc464b01b9f92942b364d9586b0ef5912432 |


`LiquidAI/d1-3B` uses the official `AutoModel`/`system_one_batch` interface at pinned model revision `051bcc464b01b9f92942b364d9586b0ef5912432`. The benchmark passes its English rubric as one typed `noul` question per sentence and applies argmax over the unrounded P(yes) and complementary P(no)=1−P(yes), with exact ties selecting no (so yes iff P(yes)>0.5). The 0.5 decision boundary was fixed; no threshold was tuned on gold labels. The model card lists 16 supported languages (Arabic, Chinese, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Thai, and Vietnamese); the other 69 languages in this full 85-language run are outside the model card's stated coverage. The run used code commit `53608400649507118903868ec571746f91d734ea`, source dataset `NoeFlandre/landuse-sentence-relevance-golden-human-set@d431a6c522b3c3a616e4f3d2814906c3011843e7` (manifest whole-set SHA-256 `7ab151eb5515a46f3a1c10b4f7df06bd53166db503790747f0f753cff43ee5f0`), and Transformers 5.17.0. Its fixed-threshold per-language results and raw probabilities are in the language JSON files.

### Scoring prompts

`Alibaba-NLP/gte-multilingual-reranker-base`, `Qwen/Qwen3-Reranker-0.6B`, `Qwen/Qwen3-Reranker-4B`, `convaiinnovations/laya-multilingual`, `mixedbread-ai/mxbai-rerank-base-v2`:

```text
<Instruct>: Judge whether the Document carries information about its target place that could help characterize land use, land cover, or the geographic environment from remote sensing, either directly or through observable proxies. Answer yes for vegetation, agriculture, forests, water, soil or surface, terrain, buildings, settlements, infrastructure, transport networks, mining, managed land, or other human or natural features with a spatial or remotely detectable signature. Answer no for information only about history, administration, people, events, demographics, economy, navigation, or activities with no meaningful land-use, land-cover, or remotely detectable implication.
<Query>: Does this sentence carry land-use, land-cover, or geographic-environment signal observable from remote sensing?
<Document>: {}
```

`BalaRajesh1/mmbert-small-nli`, `MoritzLaurer/bge-m3-zeroshot-v2.0`, `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, `fastino/gliner2.5-multi-v1`, `knowledgator/gliclass-multilang-mini`:

```text
Zero-shot classification input for NLI and label-matching models. The premise is the TARGET SENTENCE; the hypothesis / positive label is the line after HYPOTHESIS, derived from the LLM prompt.

TARGET SENTENCE: {}

HYPOTHESIS: This sentence contains information useful for inferring the land use or land cover of the associated geographic place.
```

`LiquidAI/LFM2.5-Encoder-350M`:

```text
Classify whether the target sentence contains information about its target place that
could help characterize land use, land cover, or geographic environment from features
observable by remote sensing.

Fill the blank with yes or no.

Answer yes for information about vegetation, agriculture, forests, water, soil or
surface, terrain, buildings, settlements, infrastructure, transport networks, mining,
managed land, or other human or natural features with a spatial or remotely detectable
signature.

Answer no for information only about history, administration, people, events,
demographics, economy, navigation, or activities with no meaningful land-use,
land-cover, or remotely detectable implication.

TARGET SENTENCE: {}
ANSWER: [MASK]
```

`LiquidAI/LFM2.5-2.6B@logprob` uses the task prompt above.

### Best thresholded scoring metrics

The d1-3B verdicts came from the scorer's unrounded two-label argmax. The boundary is 0.5, and exact P(no)=P(yes) ties resolve to `no` (equivalent to `yes` iff P(yes)>0.5). The published six-decimal scores show 861 records as `no=0.500000 yes=0.500000 native=0.500000`; all 861 saved predictions are `no`. Replaying the stored scores with `P(yes)>0.5` reproduces all 25,500 saved predictions; using the inclusive `P(yes)≥0.5` rule would flip those 861. The original full-precision per-item probabilities were not stored or logged, so the preserved files cannot distinguish exact 0.5 values from values below 0.5 that rounded to 0.500000. Saved labels and metrics are unchanged. No threshold was tuned on gold labels, so d1-3B is intentionally absent from the threshold sweep and best-threshold summary. See the [tie audit](../results/liquidai-d1-3b-20261013T1022Z/tie-audit.json).

| model | languages | MCC @ threshold | F1 @ threshold | balanced accuracy @ threshold | precision @ threshold | recall @ threshold | ROC-AUC | items/s | peak VRAM (GiB) |
|---|---|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-2.6B@logprob | 85 | **0.3942 @ 0.9** | **0.7307 @ 0.8** | **0.6884 @ 0.9** | 0.7109 @ 0.9 | **1 @ 0** | **0.7773** | 112.48 | 6.76 |
| Qwen/Qwen3-Reranker-4B | 85 | <u>0.2897 @ 0.03</u> | <u>0.723 @ 0.003</u> | <u>0.6435 @ 0.03</u> | <u>0.8288 @ 0.3</u> | **1 @ 0** | <u>0.7051</u> | n/a | n/a |
| MoritzLaurer/bge-m3-zeroshot-v2.0 | 85 | 0.2573 @ 0.1 | 0.7041 @ 0.03 | 0.6226 @ 0.1 | **0.8344 @ 0.6** | **1 @ 0** | 0.6651 | 112.57 | 1.25 |
| knowledgator/gliclass-multilang-mini | 85 | 0.1987 @ 0.2 | 0.7037 @ 0.03 | 0.5976 @ 0.2 | 0.6383 @ 0.3 | **1 @ 0** | 0.6258 | 258.37 | n/a |
| Qwen/Qwen3-Reranker-0.6B | 85 | 0.187 @ 0.0003 | 0.7026 @ 1e-05 | 0.593 @ 0.0003 | 0.7863 @ 0.03 | **1 @ 0** | 0.6353 | n/a | n/a |
| mixedbread-ai/mxbai-rerank-base-v2 | 85 | 0.1416 @ 0.003 | 0.7007 @ 0.001 | 0.5608 @ 0.01 | 0.6345 @ 0.01 | **1 @ 0** | 0.6153 | 14.34 | 7.14 |
| LiquidAI/LFM2.5-Encoder-350M | 15 | 0.0912 @ 0.9 | 0.6967 @ 0.8 | 0.5171 @ 0.9 | 0.5424 @ 0.9 | **1 @ 0** | 0.5462 | 194.86 | 1.75 |
| convaiinnovations/laya-multilingual | 85 | 0.0383 @ 0.4 | 0.6959 @ 0.01 | 0.5056 @ 0.4 | 0.5361 @ 0.4 | **1 @ 0** | 0.4871 | 156.37 | 1.55 |
| Alibaba-NLP/gte-multilingual-reranker-base | 85 | 0.0593 @ 0.5 | 0.6957 @ 0 | 0.5202 @ 0.6 | 0.5665 @ 0.6 | **1 @ 0** | 0.5409 | 78.64 | 1.29 |
| BalaRajesh1/mmbert-small-nli | 85 | 0 @ 0 | 0.6957 @ 0 | 0.5 @ 0 | 0.5333 @ 0 | **1 @ 0** | 0.4992 | **373.45** | **0.42** |
| MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | 85 | 0.005 @ 0.9 | 0.6957 @ 0 | 0.5009 @ 0.9 | 0.5338 @ 0.9 | **1 @ 0** | 0.5284 | <u>260.91</u> | <u>0.74</u> |
| fastino/gliner2.5-multi-v1 | 85 | 0.1305 @ 0.9 | 0.6957 @ 0 | 0.5607 @ 0.9 | 0.5833 @ 0.9 | **1 @ 0** | 0.6075 | 61.97 | 1.18 |

**Observed output behavior:** In these runs, LFM2.5-Encoder-350M predicted `yes` for 298 to 300 of 300 items per language across 15 languages.

### LFM2.5-2.6B: log-probabilities vs generation

Same model, prompt and languages; log-probs call yes when P(yes) > P(no).

| method | F1 | MCC | unparsed | ROC-AUC | GPU hours | ms/item |
|---|---|---|---|---|---|---|
| generation + parsing | 0.8029 | 0.5352 | 0.0042 | n/a | 40.11 | 5663.1 |
| yes/no log-probs | 0.6965 | 0.0191 | 0.0 | 0.7773 | 0.06 | 9.0 |
| same GPU (NVIDIA A100-SXM4-40GB; en, fr, zh): generation vs log-probs | | | | | 2462 s vs 7.1 s (349x) | |

## Runtime performance

Timings are generation wall seconds summed across language runs; latency and throughput
are recomputed from prediction telemetry. Different devices and runtimes are not directly
comparable. Full per-language measurements are in `leaderboard.csv`. [Pinned run manifest and artifact hashes](../results/liquidai-d1-3b-20261013T1022Z/run-manifest.json).

Reproducibility probes show GPU sensitivity: on 15 overlapping LFM2.5-2.6B SGLang-throughput languages, A40 versus RTX 6000 Ada changed 474/4,500 verdicts; LFM2.5-8B-A1B changed 31/300 verdicts across GPU types. VL-3B SGLang throughput changed 158/25,500 verdicts between RTX A6000 and RTX 6000 Ada, while its same-GPU DSpark comparison changed 0/25,500. Throughput mode versus batch size 1 changed 2/300 VL-3B English verdicts. Transformers continuous batching fails on LFM2 with `Invalid group type: conv`. Compare runs only with the same runtime, mode and GPU model.

| model_id | runtime | device | generation_mode | batch_size | language_count | cumulative_wall_seconds | sentences_per_second | latency_mean_seconds | latency_p50_seconds | latency_p95_seconds | generated_tokens | output_tokens_per_second | mean_accept_length | draft_accept_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alibaba-NLP/gte-multilingual-reranker-base | transformers |  | static-batched | 16 | 85 | 326.72 | 78.049 |  |  |  |  |  |  |  |
| BalaRajesh1/mmbert-small-nli | transformers | Quadro RTX 6000 | static-batched | 16 | 85 | 71.8 | 355.143 |  |  |  |  |  |  |  |
| HuggingFaceTB/SmolLM3-3B | transformers |  | static-batched | 16 | 85 | 4506.36 | 5.659 |  |  |  |  |  |  |  |
| LiquidAI/d1-3B | transformers | NVIDIA RTX PRO 6000 Blackwell Server Edition | static-batched | 16 | 85 | 121.21 | 210.373 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-1.2B-Instruct | transformers |  | static-batched | 16 | 85 | 771.63 | 33.047 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-1.2B-Instruct+DSpark-throughput-b16 | sglang | NVIDIA L40S | sglang-throughput | 16 | 85 | 138.34 | 184.328 | 0.0783 | 0.0621 | 0.0997 | 51000 | 368.66 | 2.0 | 0.1111 |
| LiquidAI/LFM2.5-1.2B-Instruct@sglang-throughput-b16 | sglang | varies | sglang-throughput | 16 | 85 | 116.89 | 218.156 | 0.0651 | 0.0539 | 0.0884 | 51000 | 436.31 |  |  |
| LiquidAI/LFM2.5-2.6B | transformers |  | static-batched | 16 | 85 | 144409.53 | 0.177 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-2.6B+DSpark-throughput-b16 | sglang | varies | sglang-throughput | 16 | 85 | 17660.83 | 1.444 | 4.3867 | 3.3135 | 11.0612 | 28688903 | 1624.44 | 3.965 | 0.3291 |
| LiquidAI/LFM2.5-2.6B@logprob | transformers | NVIDIA A100-SXM4-40GB | static-batched | 16 | 85 | 230.01 | 110.864 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-2.6B@sglang-throughput-b16 | sglang | varies | sglang-throughput | 16 | 85 | 38332.09 | 0.665 | 9.1619 | 7.0569 | 24.1868 | 28575570 | 745.47 |  |  |
| LiquidAI/LFM2.5-350M | transformers |  | static-batched | 16 | 85 | 353.71 | 72.092 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-8B-A1B | transformers |  | static-batched | 16 | 85 | 118305.81 | 0.216 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-8B-A1B+DSpark-throughput-b16 | sglang | varies | sglang-throughput | 16 | 85 | 11027.38 | 2.312 | 3.7921 | 3.365 | 7.2036 | 10491590 | 951.41 | 4.093 | 0.3425 |
| LiquidAI/LFM2.5-8B-A1B@sglang-throughput-b16 | sglang | varies | sglang-throughput | 16 | 85 | 16132.6 | 1.581 | 5.4952 | 4.8097 | 10.2668 | 10543669 | 653.56 |  |  |
| LiquidAI/LFM2.5-Encoder-350M | transformers | NVIDIA L4 | static-batched | 16 | 15 | 24.88 | 180.883 |  |  |  |  |  |  |  |
| LiquidAI/LFM2.5-VL-3B | transformers | NVIDIA L40S | static-batched | 16 | 85 | 248.13 | 102.769 | 0.1539 | 0.1479 | 0.1879 | 51000 | 205.54 |  |  |
| LiquidAI/LFM2.5-VL-3B+DSpark-throughput-b16 | sglang | NVIDIA RTX 6000 Ada Generation | sglang-throughput | 16 | 85 | 335.76 | 75.948 | 0.2011 | 0.1995 | 0.239 | 51000 | 151.9 | 2.0 | 0.3362 |
| LiquidAI/LFM2.5-VL-3B@sglang-throughput-b16 | sglang | NVIDIA RTX 6000 Ada Generation | sglang-throughput | 16 | 85 | 291.37 | 87.516 | 0.1737 | 0.1731 | 0.2091 | 51000 | 175.03 |  |  |
| MoritzLaurer/bge-m3-zeroshot-v2.0 | transformers | Quadro RTX 6000 | static-batched | 16 | 85 | 228.53 | 111.583 |  |  |  |  |  |  |  |
| MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | transformers | Quadro RTX 6000 | static-batched | 16 | 85 | 98.72 | 258.312 |  |  |  |  |  |  |  |
| Qwen/Qwen3-0.6B | transformers |  | static-batched | 16 | 85 | 280.58 | 90.883 |  |  |  |  |  |  |  |
| Qwen/Qwen3-1.7B | transformers |  | static-batched | 16 | 85 | 537.17 | 47.471 |  |  |  |  |  |  |  |
| Qwen/Qwen3-4B | transformers |  | static-batched | 16 | 85 | 908.71 | 28.062 |  |  |  |  |  |  |  |
| Qwen/Qwen3-4B-Instruct-2507 | transformers |  | static-batched | 16 | 85 | 829.5 | 30.741 |  |  |  |  |  |  |  |
| Qwen/Qwen3-8B | transformers |  | static-batched | 16 | 85 | 890.93 | 28.622 |  |  |  |  |  |  |  |
| Qwen/Qwen3-Reranker-0.6B | transformers |  | static-batched | 16 | 85 | 177.81 | 143.41 |  |  |  |  |  |  |  |
| Qwen/Qwen3-Reranker-4B | transformers |  | static-batched | 16 | 85 | 662.51 | 38.49 |  |  |  |  |  |  |  |
| Qwen/Qwen3.5-0.8B | transformers |  | static-batched | 16 | 85 | 321.1 | 79.414 |  |  |  |  |  |  |  |
| Qwen/Qwen3.5-2B | transformers |  | static-batched | 16 | 85 | 377.11 | 67.62 |  |  |  |  |  |  |  |
| Qwen/Qwen3.5-4B | transformers |  | static-batched | 16 | 85 | 1096.92 | 23.247 |  |  |  |  |  |  |  |
| Qwen/Qwen3.5-9B | transformers |  | static-batched | 16 | 85 | 1614.21 | 15.797 |  |  |  |  |  |  |  |
| allenai/OLMo-2-1124-7B-Instruct | transformers |  | static-batched | 16 | 85 | 915.11 | 27.866 |  |  |  |  |  |  |  |
| allenai/Olmo-3-7B-Instruct | transformers |  | static-batched | 16 | 85 | 1067.17 | 23.895 |  |  |  |  |  |  |  |
| convaiinnovations/laya-multilingual | transformers |  | static-batched | 16 | 85 | 180.29 | 141.44 |  |  |  |  |  |  |  |
| fastino/gliner2.5-multi-v1 | transformers | NVIDIA A100-SXM4-40GB | static-batched | 16 | 85 | 411.7 | 61.939 |  |  |  |  |  |  |  |
| google/gemma-4-E2B-it | transformers |  | static-batched | 16 | 85 | 430.29 | 59.262 |  |  |  |  |  |  |  |
| google/gemma-4-E4B-it | transformers |  | static-batched | 16 | 85 | 817.29 | 31.201 |  |  |  |  |  |  |  |
| ibm-granite/granite-3.3-2b-instruct | transformers |  | static-batched | 16 | 85 | 547.74 | 46.555 |  |  |  |  |  |  |  |
| ibm-granite/granite-4.1-3b | transformers |  | static-batched | 16 | 85 | 437.63 | 58.269 |  |  |  |  |  |  |  |
| knowledgator/gliclass-multilang-mini | transformers | Quadro RTX 6000 | static-batched | 16 | 85 | 100.12 | 254.699 |  |  |  |  |  |  |  |
| microsoft/Phi-4-mini-instruct | transformers |  | static-batched | 16 | 85 | 563.83 | 45.226 |  |  |  |  |  |  |  |
| mistralai/Ministral-3-3B-Instruct-2512-BF16 | transformers |  | static-batched | 16 | 85 | 1707.26 | 14.936 |  |  |  |  |  |  |  |
| mistralai/Ministral-3-8B-Instruct-2512-BF16 | transformers |  | static-batched | 16 | 85 | 1455.05 | 17.525 |  |  |  |  |  |  |  |
| mixedbread-ai/mxbai-rerank-base-v2 | transformers |  | static-batched | 16 | 85 | 1880.78 | 13.558 |  |  |  |  |  |  |  |
| swiss-ai/Apertus-8B-Instruct-2509 | transformers |  | static-batched | 16 | 85 | 1300.16 | 19.613 |  |  |  |  |  |  |  |
| tiiuae/Falcon-H1-3B-Instruct | transformers |  | static-batched | 1 | 85 | 7697.51 | 3.313 |  |  |  |  |  |  |  |
| tiiuae/Falcon3-1B-Instruct | transformers |  | static-batched | 16 | 85 | 278.09 | 91.696 |  |  |  |  |  |  |  |
| tiiuae/Falcon3-3B-Instruct | transformers |  | static-batched | 16 | 85 | 427.21 | 59.69 |  |  |  |  |  |  |  |
| tiiuae/Falcon3-7B-Instruct | transformers |  | static-batched | 16 | 85 | 910.7 | 28.0 |  |  |  |  |  |  |  |
| unsloth/Qwen3.8-27B-GGUF@UD-IQ2_XXS | transformers | NVIDIA A100-SXM4-40GB | static-batched | 16 | 85 | 6530.99 | 3.904 |  |  |  |  |  |  |  |
| utter-project/EuroLLM-9B-Instruct-2512 | transformers |  | static-batched | 16 | 85 | 673.97 | 37.835 |  |  |  |  |  |  |  |

## DSpark speculative decoding

Greedy DSpark runs should match the same-target SGLang baseline across every language;
the check requires complete item coverage and identical generated text.

| target_model | sglang_baseline | dspark_run | baseline_output_tokens_per_second | dspark_output_tokens_per_second | speedup | identical_predictions | languages_compared | items_compared | verdict_differences | text_differences |
|---|---|---|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2.5-1.2B-Instruct | LiquidAI/LFM2.5-1.2B-Instruct@sglang-throughput-b16 | LiquidAI/LFM2.5-1.2B-Instruct+DSpark-throughput-b16 | 436.31 | 368.66 | 0.84x | yes | 85 | 25500 | 0 | 0 |
| LiquidAI/LFM2.5-2.6B | LiquidAI/LFM2.5-2.6B@sglang-throughput-b16 | LiquidAI/LFM2.5-2.6B+DSpark-throughput-b16 | 745.47 | 1624.44 | 2.18x | no | 85 | 25500 | 3004 | 24962 |
| LiquidAI/LFM2.5-8B-A1B | LiquidAI/LFM2.5-8B-A1B@sglang-throughput-b16 | LiquidAI/LFM2.5-8B-A1B+DSpark-throughput-b16 | 653.56 | 951.41 | 1.46x | no | 85 | 25500 | 4641 | 25029 |
| LiquidAI/LFM2.5-VL-3B | LiquidAI/LFM2.5-VL-3B@sglang-throughput-b16 | LiquidAI/LFM2.5-VL-3B+DSpark-throughput-b16 | 175.03 | 151.9 | 0.87x | yes | 85 | 25500 | 0 | 0 |

## Snapshot and provenance

<!-- LRB:SNAPSHOT_SUMMARY -->

Scorer model revisions and their native rules are listed in the scoring setup table above. Per-language rows preserve model revisions for all runs. The d1-3B [immutable run manifest](../results/liquidai-d1-3b-20261013T1022Z/run-manifest.json) records code revision `53608400649507118903868ec571746f91d734ea`, model revision `051bcc464b01b9f92942b364d9586b0ef5912432`, source dataset revision `d431a6c522b3c3a616e4f3d2814906c3011843e7`, source-set SHA-256 `7ab151eb5515a46f3a1c10b4f7df06bd53166db503790747f0f753cff43ee5f0`, and Transformers 5.17.0. Its per-language JSON outputs retain the saved predictions and scores.

<!-- LRB:CURRENT_ADDITIONS -->

The card metadata declares the MIT license. The benchmark implementation is linked at [GitHub](https://github.com/NoeFlandre/benchmark-llms-landuse-relevance). The d1-3B language-support statement refers to the [official model card](https://huggingface.co/LiquidAI/d1-3B). Cite the dataset and code repository when using these results.
