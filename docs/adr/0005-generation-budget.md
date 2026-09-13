# ADR-0005 — Reading the verdict from a model that reasons first

**Status:** accepted · 2026-09-13
**Refines:** [ADR-0001](0001-greedy-generation.md), [ADR-0002](0002-unparsed-as-error.md)

## Context

The prompt asks for exactly one token, so the first full run used
`max_new_tokens=8`. Two of the four models then scored 0.000 accuracy with a 100%
unparsed rate, because they open every answer with `1.  **Analyze the Request:**` and
never reach a verdict inside eight tokens.

Raising the budget to 256 made the failure worse rather than better: `LFM2.5-2.6B`
appeared to score 0.519 by answering `yes` to everything. Its raw generations show
what actually happened — it restates the prompt's own rubric,

> Criteria for "yes": vegetation, agriculture, forests, water, …

and the parser, reading from the front, scored that restatement as the verdict. The
number was an artefact of the harness, not a measurement of the model.

## Decision

Three changes, together:

1. **The verdict is the last standalone `yes`/`no`, not the first.** A direct answer is
   one token either way; a reasoning trace mentions both options before committing.
2. **A generation that used its whole budget without stopping carries no verdict.** The
   runtime reports truncation, and a truncated generation is scored unparsed however
   many verdict words appear in it.
3. **The default budget is 1024 tokens**, enough for these models to finish. The strict
   eight-token run is kept and published under `strict-8-tokens/`.

## Consequences

- The two compliant models score identically at every budget — they stop after the
  verdict — so the larger budget costs nothing in comparability while rescuing the
  models that preamble.
- `unparsed_rate` now separates two failures that were previously conflated: a model
  that never commits, and one that was cut off. The stored `truncated` flag says which.
- Reading from the end still has a failure mode: a conclusion followed by a caveat that
  names the other verdict would be misread. Raw generations are stored so this is
  auditable, and it is why the strict run is published rather than discarded.
- The strict run remains the honest answer to "does this model obey the output
  contract?" — a different and also useful question from "does it know the answer?".
