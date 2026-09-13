# ADR-0001 — Greedy generation rather than logit scoring

**Status:** accepted · 2026-09-13

## Context

The prompt asks each model for exactly one token, `yes` or `no`. Two ways to obtain
a verdict: compare the logits of the two candidate continuations, or generate a few
tokens greedily and parse the text.

## Decision

Generate greedily (`do_sample=False`, `max_new_tokens=8`) and parse the text.

## Consequences

- What we measure is what a caller would actually get, including instruction-following
  failures. Logit scoring hides those: a model that would have emitted a preamble still
  scores a clean verdict.
- Comparable across tokenizers and chat templates, where "the logit of `yes`" is not.
- Greedy decoding keeps a run replayable: same weights revision, same prompt digest,
  same output.
- Cost: a model can refuse to answer. That is deliberately visible — see ADR-0002.
