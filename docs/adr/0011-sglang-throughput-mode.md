# ADR-0011 — SGLang throughput mode runs the multilingual sweep; batch size 1 stays the latency baseline

**Status:** accepted · 2026-09-27
**Refines:** [ADR-0009](0009-speed-measurement.md), [ADR-0010](0010-dspark-speculative-decoding.md)

## Context

SGLang can serve many requests at once, and `--throughput` (`LRB_THROUGHPUT=1` on
Grid'5000, its default there) submits the prompts as concurrent requests. That raises
items per second but changes what the speed columns mean. DSpark's published gain is a
latency gain measured one request at a time: a draft helps most when the target is
memory-bound, and batching moves it towards compute-bound, shrinking or hiding the
speed-up.

At batch size 1 the eight SGLang runs cannot cover 85 languages: LFM2.5-2.6B alone
took 966 s for English on an H100 NVL, about 23 GPU-hours for the sweep.

Measured on English with LFM2.5-VL-3B@sglang on one L40S (issue #78):

| mode | seconds | sentences/s | verdicts differing from bs=1 |
|---|---:|---:|---:|
| batch size 1 | 14.45 | 20.8 | — |
| throughput, 16 concurrent | 2.17 | 138.1 | 2 / 300 |

Concurrency is not batch-invariant: greedy decoding over a different batch changes
floating-point reductions, so a few generations diverge. The same holds across GPU
types (see ADR-0010): runs are only byte-comparable on the same mode and GPU model.

Cross-GPU reruns measured 31/300 verdict changes for LFM2.5-8B-A1B. The VL-3B
SGLang-throughput rerun changed 158/25,500 verdicts between RTX A6000 and RTX 6000
Ada, while the same-Ada SGLang/DSpark pair matched all 25,500 items. For the 15
overlapping LFM2.5-2.6B throughput languages, A40 versus RTX 6000 Ada changed
474/4,500 verdicts. Throughput mode versus batch size 1 changed 2/300 English VL-3B
verdicts. Transformers continuous batching currently fails on LFM2 with
`Invalid group type: conv`.

## Decision

- The roster keeps every SGLang run, with and without the draft, at `batch_size=1`.
  English runs in that mode are the published latency rows and DSpark's speed claim.
- The 85-language SGLang and DSpark sweep runs in throughput mode. Its runs are named
  `<run>-throughput-b<N>` and are never mixed with the batch-size-1 rows.
- DSpark's lossless check compares a `+DSpark` run with its `@sglang` baseline in the
  same mode on the same GPU model; any difference there is a bug, not a result.
- Throughput mode stays opt-in for `lrb run`; the Grid'5000 node script enables it.

## Consequences

- Accuracy for the SGLang variants across 85 languages is reported from throughput
  runs; it may differ from batch size 1 on a small fraction of items (2/300 above).
- Throughput speed columns answer a capacity question, not DSpark's latency claim.
- Reproducing a throughput run exactly needs the same concurrency and GPU model, both
  recorded in the result metadata.
