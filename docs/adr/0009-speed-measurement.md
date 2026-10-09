# ADR-0009 - Measure the speed with the scores

**Status:** accepted · 2026-09-25

## Context

The scores show which model is right. They do not show what it costs to get the answer.
The benchmark tests speculative decoding (ADR-0010) only for its speed. The reasoning-first
models produce outputs that are about two orders of magnitude longer than the outputs of
the models that answer with one token.

## Decision

Every run records these values for each prediction:

- the wall time of the generator call that gave the answer (`latency_seconds`)
- the tokens that the model produced (`generated_tokens`)
- for speculative decoding: the target verification passes, and the draft tokens that
  the target accepted and the draft tokens that the draft proposed

The code derives the run-level figures from these values only:

| figure | definition |
|---|---|
| `wall_seconds` | generation wall time for the complete benchmark; model loading is excluded |
| `sentences_per_second` | items / wall time |
| `latency_{mean,p50,p95}_seconds` | over the latencies of the predictions; p-values by linear interpolation |
| `generated_tokens` | sum over the predictions |
| `output_tokens_per_second` | generated tokens / wall time |
| `mean_accept_length` | generated tokens / verification passes, pooled over the run |
| `draft_accept_rate` | accepted draft tokens / proposed draft tokens, pooled |

The code reports a figure only when every prediction recorded its input. A partial sum
makes the run look slower or faster than it is, without notice.

## Consequences

- The code recomputes the speed. It never stores it as the authoritative value.
  `RunResult.speed` is derived from the predictions. The `speed` block in a result file
  is a copy for convenience. The reader ignores it. Older result files still load and
  report wall-time throughput. Their latency and token figures are blank.
- The latency is for each generator call. With batching, all sentences in a batch share
  the latency of the batch. The SGLang runs use batch size 1. Therefore, their latency is
  for each sentence. This matches the measurement setting of the DSpark card.
- Transformers token counts stop at the first stop token. The shorter rows in a batch
  have padding after it. SGLang reports `completion_tokens` itself.
- The throughput depends on the GPU. Each result file records its settings. Compare the
  speed only between runs from the same job on the same node.
