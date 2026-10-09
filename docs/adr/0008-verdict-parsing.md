# ADR-0008 - Prefer explicit verdicts when parsing generations

**Status:** accepted · 2026-09-26

## Context

Some generations start with a direct `yes` or `no` answer. Then they continue with text
that mentions the other label. If the parser takes the final token, it can reverse an
answer that is clearly stated. Other models reason first and repeat the rubric. They put
their answer at the end.

## Decision

Parse in this order:

1. An exact standalone `yes` or `no`. Surrounding whitespace and punctuation are
   permitted.
2. A standalone `yes` or `no` at the start of the response.
3. The final standalone `yes` or `no` anywhere in the response. This keeps the fallback
   for outputs that reason first.

Each prediction stores `parse_mode` (`exact`, `leading`, or `last`). Unparsable outputs
and truncated outputs have no mode. The raw generations stay unchanged.

To recompute the saved results without a model load, run:
`python scripts/reparse_results.py results`

## Consequences

- A leading answer has priority over later mentions of the other label.
- Outputs that reason first keep their former last-token interpretation.
- When this rule changes, regenerate the historical results and their summaries.
