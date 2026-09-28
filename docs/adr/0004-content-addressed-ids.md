# ADR-0004 — Items keep source identity across language splits

**Status:** accepted · 2026-09-13

## Context

Results from different models and language splits must be joinable. A row index breaks
the moment a CSV is re-sorted or extended, and a translated sentence is not a stable
cross-language key.

## Decision

Each language-neutral source row receives a stable `source_item_id`, including a
deterministic occurrence number for duplicate neutral records. A language-aware row
then gets `item_id = sha256(source_item_id + "\\0" + language)[:16]`. The loader rejects
duplicate rows within a language and validates every split against the manifest's source
ID sequence.

## Consequences

- Per-item comparison across models is a dictionary join on `(source_item_id, language)`;
  source IDs provide the cross-language join.
- A changed translation produces a new benchmark revision rather than silently changing
  the meaning of an existing prediction.
- Cost: an id reveals nothing about position, and a source-data correction requires a
  new manifest digest. Both are explicit and auditable.
