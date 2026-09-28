# ADR-0005 — Allow enough output budget for a final verdict

**Status:** accepted · 2026-09-20
**Refines:** [ADR-0001](0001-greedy-generation.md), [ADR-0002](0002-unparsed-as-error.md)

## Context

The prompt requests one lowercase verdict, but some instruction-tuned models produce a
reasoning preamble before committing. A short generation budget would measure whether a
model can answer within that budget rather than whether it can classify the sentence.

## Decision

Three rules apply together:

1. **The verdict is the last standalone `yes`/`no`.** A reasoning trace can mention
   both options before committing.
2. **A generation that uses its whole budget without stopping carries no verdict.** The
   runtime reports truncation, and a truncated generation is scored unparsed.
3. **The active default is 4096 tokens.** The runtime records truncation explicitly and
   retains the raw generation for audit.

## Consequences

- A larger budget gives every language split the same opportunity to reach a verdict and
  keeps comparisons focused on classification.
- `unparsed_rate` now separates two failures that were previously conflated: a model
  that never commits, and one that was cut off. The stored `truncated` flag says which.
- Reading from the end still has a failure mode: a conclusion followed by a caveat that
  names the other verdict would be misread. Raw generations are stored so this remains
  auditable.
- The budget is part of run metadata, so changing it requires a distinguishable result
  set and cannot silently mix with the active leaderboard.
