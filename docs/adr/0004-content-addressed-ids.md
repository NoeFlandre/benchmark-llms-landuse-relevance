# ADR-0004 — Items are identified by the hash of their sentence

**Status:** accepted · 2026-09-13

## Context

Results from different models, produced on different days, must be joinable. A row
index breaks the moment the CSV is re-sorted or extended.

## Decision

`item_id = sha256(sentence)[:16]`. The loader rejects duplicate sentences so ids stay
unique.

## Consequences

- Per-item comparison across models and across benchmark revisions is a dictionary join.
- Editing a sentence produces a new item rather than silently changing an old one's
  meaning — which is the honest outcome, since the old predictions no longer apply.
- Cost: an id reveals nothing about position, and a typo fix orphans previous
  predictions for that sentence. Both are acceptable for a benchmark of this size.
