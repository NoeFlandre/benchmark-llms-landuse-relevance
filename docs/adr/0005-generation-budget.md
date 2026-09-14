# ADR-0005 — Reading the verdict from a model that reasons first

**Status:** superseded for publication · 2026-09-14
**Refines:** [ADR-0001](0001-greedy-generation.md), [ADR-0002](0002-unparsed-as-error.md)

The strict eight-token diagnostic described here was removed from the public release on
2026-09-14 because it measured output-budget compliance rather than land-use relevance.
The 4096-token configuration is the sole published benchmark.

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
3. **The budget is large enough for these models to finish** — 1024 by default, 4096 for
   the published run. At 1024 the 2.6B still truncated on a third of the benchmark; at
   4096 it truncates on none of it, and its accuracy settles at 0.857. The eight-token
   run was retained only as an exploratory diagnostic.

## Consequences

- The two compliant models score identically at every budget — they stop after the
  verdict — so the larger budget costs nothing in comparability while rescuing the
  models that preamble.
- `unparsed_rate` now separates two failures that were previously conflated: a model
  that never commits, and one that was cut off. The stored `truncated` flag says which.
- Reading from the end still has a failure mode: a conclusion followed by a caveat that
  names the other verdict would be misread. Raw generations are stored so this remains
  auditable.
- The exploratory strict run remains in Git history for provenance but is not part of the
  public benchmark or leaderboard.
