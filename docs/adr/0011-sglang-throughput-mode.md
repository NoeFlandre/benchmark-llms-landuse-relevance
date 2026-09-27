# ADR-0011 — SGLang throughput mode is opt-in; batch size 1 stays the latency baseline

**Status:** accepted · 2026-09-27
**Refines:** [ADR-0009](0009-speed-measurement.md), [ADR-0010](0010-dspark-speculative-decoding.md)

## Context

SGLang can serve many requests at once, and `--throughput` (`LRB_THROUGHPUT=1` on
Grid'5000) submits the prompts as concurrent requests. That raises items per second
but changes what the speed columns mean. DSpark's published gain is a latency gain
measured one request at a time: a draft helps most when the target is memory-bound,
and batching moves it towards compute-bound, shrinking or hiding the speed-up.

## Decision

- The roster pins every SGLang run, with and without the draft, to `batch_size=1`.
  These are the published latency rows and the like-for-like DSpark comparison.
- Throughput mode is opt-in, per invocation, and never the default.
- A throughput run is reported only if its predictions match the batch-size-1 run of
  the same model and language; a mismatch is a bug in the run, not a result.

## Consequences

- The `@sglang` / `+DSpark` speed comparison stays the one DSpark's claim is about.
- Throughput numbers answer a separate capacity question and are labelled as such.
- Concurrency must not change a score; the predictions are compared before reporting.
