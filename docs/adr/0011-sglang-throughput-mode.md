# ADR-0011 - SGLang throughput mode runs the multilingual sweep; batch size 1 stays the latency baseline

**Status:** accepted · 2026-09-27
**Refines:** [ADR-0009](0009-speed-measurement.md), [ADR-0010](0010-dspark-speculative-decoding.md)

## Context

SGLang can serve many requests at the same time. The option `--throughput`
(`LRB_THROUGHPUT=1` on Grid'5000, and it is the default there) submits the prompts as
concurrent requests. This increases the items per second. But it changes the meaning of
the speed columns.

The published gain of DSpark is a latency gain. It is measured one request at a time. A
draft helps most when the target is memory-bound. Batching moves the target toward
compute-bound. This reduces the speed-up or hides it.

At batch size 1, the eight SGLang runs cannot cover 85 languages. LFM2.5-2.6B alone took
966 s for English on an H100 NVL. This is about 23 GPU-hours for the sweep.

The measurement on English with LFM2.5-VL-3B@sglang on one L40S (issue #78):

| mode | seconds | sentences/s | verdicts differing from bs=1 |
|---|---:|---:|---:|
| batch size 1 | 14.45 | 20.8 | - |
| throughput, 16 concurrent | 2.17 | 138.1 | 2 / 300 |

Concurrency is not batch-invariant. Greedy decoding over a different batch changes the
floating-point reductions. Therefore, a few generations diverge. The same is true across
GPU types (refer to ADR-0010). Runs are byte-comparable only on the same mode and the same
GPU model.

The cross-GPU reruns gave these results:

- LFM2.5-8B-A1B: 31/300 verdict changes.
- VL-3B SGLang-throughput rerun: 158/25,500 verdict changes between RTX A6000 and RTX 6000
  Ada. The SGLang/DSpark pair on the same Ada GPU matched all 25,500 items.
- LFM2.5-2.6B, for the 15 overlapping throughput languages: 474/4,500 verdict changes
  between A40 and RTX 6000 Ada.
- Throughput mode compared with batch size 1: 2/300 verdict changes for English VL-3B.

Transformers continuous batching currently fails on LFM2 with the error
`Invalid group type: conv`.

## Decision

- The roster keeps every SGLang run, with and without the draft, at `batch_size=1`. The
  English runs in that mode are the published latency rows and the speed claim of DSpark.
- The 85-language sweep of SGLang and DSpark runs in throughput mode. The names of its
  runs are `<run>-throughput-b<N>`. The benchmark never mixes them with the rows of
  batch size 1.
- The lossless check of DSpark compares a `+DSpark` run with its `@sglang` baseline in
  the same mode on the same GPU model. Any difference there is a bug. It is not a result.
- Throughput mode stays optional for `lrb run`. The node script of Grid'5000 enables it.

## Consequences

- The accuracy of the SGLang variants across 85 languages comes from throughput runs. It
  can be different from batch size 1 on a small fraction of items (2/300 above).
- The throughput speed columns answer a question about capacity. They do not answer the
  latency claim of DSpark.
- To reproduce a throughput run exactly, use the same concurrency and the same GPU model.
  The result metadata records both.
