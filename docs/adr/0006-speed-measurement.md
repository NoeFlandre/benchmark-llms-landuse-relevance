# ADR-0006 — Measuring speed alongside the scores

**Status:** accepted · 2026-09-25

## Context

The scores say which model is right; they say nothing about what it costs to get
the answer. Speculative decoding (ADR-0007) is only worth benchmarking for its
speed, and the reasoning-first models differ from the one-token answerers by two
orders of magnitude in output length.

## Decision

Every run records, per prediction, the wall time of the generator call that answered
it (`latency_seconds`), the tokens the model produced (`generated_tokens`), and, under
speculative decoding, the target verification passes and the draft tokens accepted
and proposed. The run-level figures are derived from those alone:

| figure | definition |
|---|---|
| `wall_seconds` | generation wall time for the whole benchmark; model loading excluded |
| `sentences_per_second` | items / wall time |
| `latency_{mean,p50,p95}_seconds` | over the per-prediction latencies; p-values by linear interpolation |
| `generated_tokens` | sum over predictions |
| `output_tokens_per_second` | generated tokens / wall time |
| `mean_accept_length` | generated tokens / verification passes, pooled over the run |
| `draft_accept_rate` | accepted draft tokens / proposed draft tokens, pooled |

A figure is reported only when every prediction recorded its input; a partial sum
would silently understate the run.

## Consequences

- Speed is recomputed, never stored authoritatively: `RunResult.speed` is derived
  from the predictions, and the `speed` block written to a result file is a
  convenience copy that is ignored on read. Older result files still load and report
  wall-time throughput, with latency and token figures left blank.
- Latency is per generator call. With batching, every sentence in a batch shares the
  batch's latency; the SGLang runs use batch size 1 so their latency is per sentence,
  matching the DSpark card's measurement setting.
- Transformers token counts stop at the first stop token, since shorter rows in a
  batch are padded past it. SGLang reports `completion_tokens` itself.
- Throughput depends on the GPU. Each result file records its settings, and speed
  should only be compared between runs from the same job on the same node.
