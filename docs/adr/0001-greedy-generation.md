# ADR-0001 - Greedy generation, not logit scoring

**Status:** accepted · 2026-09-13

## Context

The prompt asks each model for exactly one lowercase verdict: `yes` or `no`. There are
two ways to get it. One way is to compare the logits of the candidate continuations. The
other way is to generate text greedily and parse the result.

## Decision

Generate greedily (`do_sample=False`, `max_new_tokens=4096`). Parse the text.

## Consequences

- The benchmark measures what a caller really gets. This includes failures to follow
  instructions. Logit scoring hides these failures. A model that would write a preamble
  still gets a clean verdict.
- The results are comparable across tokenizers and chat templates. "The logit of `yes`"
  is not comparable across them.
- Greedy decoding keeps a run replayable. The same weights revision and the same prompt
  digest give the same output.
- Cost: a model can refuse to answer or can use the complete generation budget. The
  benchmark shows this on purpose. Refer to ADR-0002 and ADR-0005.
