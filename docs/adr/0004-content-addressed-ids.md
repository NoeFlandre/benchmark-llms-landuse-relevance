# ADR-0004 - Items keep the source identity across language splits

**Status:** accepted · 2026-09-13

## Context

The results of different models and language splits must be joinable. A row index breaks
when someone sorts or extends a CSV file. A translated sentence is not a stable
cross-language key.

## Decision

Each language-neutral source row gets a stable `source_item_id`. This includes a
deterministic occurrence number for duplicate neutral records. Then each language-aware
row gets `item_id = sha256(source_item_id + "\\0" + language)[:16]`.

The loader rejects duplicate rows in a language. It validates every split against the
sequence of source ids in the manifest.

## Consequences

- A comparison of items across models is a dictionary join on
  `(source_item_id, language)`. The source ids give the cross-language join.
- A changed translation gives a new benchmark revision. It does not change, without
  notice, the meaning of an existing prediction.
- Cost: an id does not show the position. A correction of the source data needs a new
  manifest digest. Both are explicit and auditable.
