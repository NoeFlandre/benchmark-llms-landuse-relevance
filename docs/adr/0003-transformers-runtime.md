# ADR-0003 - Hugging Face Transformers, not vLLM

**Status:** accepted · 2026-09-13

## Context

The complete benchmark is 85 aligned language splits of 300 rows each. vLLM is the faster
serving runtime. Transformers is the reference implementation. New architectures appear
first in Transformers.

## Decision

Run through `transformers`. Use a batched, left-padded greedy decode. Put it behind the
`TextGenerator` protocol.

## Consequences

- The workload is small. The throughput advantage of vLLM gives no benefit. It costs a
  server process, a compatibility matrix, and a slower cold start on a reserved node.
- `transformers` supports architectures such as LFM2.5 on the day of release. vLLM
  support can be late. A benchmark that cannot load the model measures nothing.
- To use a vLLM engine later, add a new class that implements `TextGenerator`. Nothing
  in the domain changes.
- GGUF quants and the GLiClass and GLiNER2 SDKs are the exceptions. Refer to
  [ADR-0007](0007-additional-runtimes.md).
