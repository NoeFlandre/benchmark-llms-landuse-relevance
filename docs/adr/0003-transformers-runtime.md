# ADR-0003 — Hugging Face Transformers rather than vLLM

**Status:** accepted · 2026-09-13

## Context

The complete benchmark is 85 aligned language splits of 300 rows. vLLM is the faster
serving runtime; Transformers is the reference implementation that new architectures
land in first.

## Decision

Run through `transformers` with a batched, left-padded greedy decode, behind the
`TextGenerator` protocol.

## Consequences

- The workload is tiny, so vLLM's throughput advantage buys nothing while costing a
  server process, a compatibility matrix, and a slower cold start on a reserved node.
- Architectures such as LFM2.5 are supported by `transformers` on release day; vLLM
  support can lag, and a benchmark that cannot load the model measures nothing.
- Swapping in a vLLM engine later is a new class implementing `TextGenerator`; nothing
  in the domain changes.
- GGUF quants and the GLiClass/GLiNER2 SDKs are the exceptions — see
  [ADR-0007](0007-additional-runtimes.md).
