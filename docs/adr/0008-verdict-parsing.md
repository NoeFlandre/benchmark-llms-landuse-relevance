# ADR-0008 — Prefer explicit verdicts when parsing generations

**Status:** accepted · 2026-09-26

## Context

Some generations begin with a direct `yes` or `no` answer and continue with prose that
mentions the other label. Taking the final token can then reverse a clearly stated
answer. Other models reason first, restate the rubric, and put their answer at the end.

## Decision

Parse in this order:

1. An exact standalone `yes` or `no`, allowing surrounding whitespace and punctuation.
2. A standalone `yes` or `no` at the start of the response.
3. The final standalone `yes` or `no` anywhere in the response, preserving the fallback
   for reasoning-first outputs.

Each prediction stores `parse_mode` (`exact`, `leading`, or `last`); unparsable and
truncated outputs have no mode. Raw generations remain unchanged. Saved results can be
recomputed with `python scripts/reparse_results.py results` without loading models.

## Consequences

- A leading answer takes precedence over later mentions of the other label.
- Reasoning-first outputs retain their former last-token interpretation.
- Historical results and their summaries must be regenerated when this rule changes.
