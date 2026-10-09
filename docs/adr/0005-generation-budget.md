# ADR-0005 - Allow enough output budget for a final verdict

**Status:** accepted · 2026-09-20
**Refines:** [ADR-0001](0001-greedy-generation.md), [ADR-0002](0002-unparsed-as-error.md)

## Context

The prompt requests one lowercase verdict. But some instruction-tuned models write a
reasoning preamble before they commit to an answer. A short generation budget then
measures if a model can answer within that budget. It does not measure if the model can
classify the sentence.

## Decision

Three rules apply together:

1. **The verdict is the last standalone `yes`/`no`.** A reasoning trace can mention both
   options before it commits.
2. **A generation that uses its complete budget without a stop has no verdict.** The
   runtime reports the truncation. The benchmark scores a truncated generation as
   unparsed.
3. **The active default is 4096 tokens.** The runtime records the truncation explicitly.
   It keeps the raw generation for audit.

## Consequences

- A larger budget gives each language split the same opportunity to reach a verdict. The
  comparisons then stay focused on classification.
- `unparsed_rate` now separates two failures that were mixed before: a model that never
  commits, and a model that was cut off. The stored `truncated` flag shows which failure
  occurred.
- Reading from the end still has a failure mode. If a conclusion is followed by a caveat
  that names the other verdict, the parser reads a wrong verdict. The benchmark stores
  the raw generations. Therefore, you can still audit this case.
- The budget is part of the run metadata. A change to the budget needs a result set that
  is distinguishable. It cannot mix with the active leaderboard without notice.
