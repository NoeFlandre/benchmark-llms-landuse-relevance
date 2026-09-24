# Multilingual Benchmark Design

**Status:** Approved on 2026-09-20

## Goal

Make the multilingual golden human set the only active benchmark: 85 language
configurations, 300 aligned items per language, language-aware item identity,
language-aware results and leaderboards, resumable Grid'5000 execution, and an
explicit archive boundary for all historical single-language artifacts.

## Constraints and decisions

- The active benchmark contains all 85 Dataset Viewer configurations, each with
  its `train` split and 300 rows.
- Every model-language pair is a separate deterministic greedy run. There are no
  repeated greedy runs or standard deviations from repeated identical decoding.
- The prompt remains one English template for every language. Only the sentence
  changes; the expected verdict remains the lowercase `yes` or `no` token.
- A verdict in another language is unparsed and therefore an error. Full
  `raw_output` remains in the result so alternative parsing can be recomputed
  later without rerunning models.
- Historical single-language data, results, documentation, and card content are
  archive-only. Active loaders, globs, reports, leaderboards, documentation, and
  publication must not read, link to, or describe them.
- Active commands reject legacy result records that lack a language. They are not
  silently upgraded or merged into multilingual output.

## Identity and dataset architecture

The upstream rows have no source-item ID, so identity is created once during
vendoring rather than guessed from translated text.

1. Build a canonical source key from the language-neutral fields: `label`,
   `polygon_name`, `h3_cell`, `latitude`, `longitude`, `source`, `region`, and
   `source_url`. Nulls, numeric values, and strings are normalized before hashing.
2. Add a deterministic occurrence number when an otherwise identical source key
   occurs more than once in the canonical English configuration.
3. Define `source_item_id` as the first 16 hexadecimal characters of the SHA-256
   digest of that canonical key plus occurrence number.
4. Store the generated mapping in `data/translations/manifest.json`. Every
   language file is validated against the manifest for the exact 300 source IDs,
   source metadata, labels, and row count; only `sentence` may vary by language.
5. Define `item_id` as the first 16 hexadecimal characters of the SHA-256 digest
   of `source_item_id + "\\0" + language`.

`BenchmarkItem` therefore carries `source_item_id`, `language`, and `item_id`.
The content hash no longer uses translated sentence text, so translations remain
joinable while identical translated text cannot collide across languages.

The active data API will expose:

- `available_languages(data_root)` — sorted language codes and row counts;
- `load_language_benchmark(data_root, language)` — one validated language;
- `load_manifest(data_root)` — dataset revision, split, file hashes, and source
  item mapping;
- a deterministic whole-set digest derived from the sorted language file hashes.

The vendored layout is:

```text
data/translations/manifest.json
data/translations/<lang>/v3-final-<lang>.csv
```

## Results and publication

`RunMetadata.language` is required for active records. A run is written at:

```text
results/<language>/<filesystem-safe-model-id>.json
```

The result store recursively reads only active result paths and skips every
`archive` path. A record without `language` raises a clear archive-only error.

The detailed leaderboard has one row per `(model_id, language)` pair. A second
aggregate file has one row per model with an unweighted macro-average of the
classification metrics over available languages, language count, and F1
minimum/maximum/standard deviation. Detailed rows remain the source of truth for
per-language inspection.

The Hugging Face card and uploaded folder are generated only from these active
records. Historical files are never included merely because they exist below the
results root.

## CLI and execution architecture

- `lrb languages` lists every available language and row count.
- `lrb run <model>` runs all 85 languages by default.
- `lrb run <model> --language <code>` narrows one run; repeated and comma-separated
  selectors are accepted.
- `lrb run-all` runs every rostered model across the selected languages.
- Checkpoints are keyed by `(model, language)` and an existing valid result is
  skipped deterministically.
- `lrb report` and `lrb publish` accept the same language filter and write both
  detailed and aggregate outputs.

The Grid'5000 planner is pure and deterministic. It sorts the model-language
matrix, assigns node shards by stable index, and can print a status table of
complete, running, and unclaimed pairs. Node submission loads one model for its
assigned languages, uses the 4096-token budget, and writes one checkpoint per
pair.

Multi-site submission assigns disjoint slices before submission using a checked-in
site configuration and explicit weights. Collection verifies the union covers the
requested matrix, no pair appears twice, every shard reports the same source
commit, and site-local results are copied off `/home` promptly.

## Issue-by-issue implementation order

1. #3 — domain identity and manifest contract.
2. #2 — vendored data and validated language loader.
3. #4 — language-aware run records, result paths, leaderboards, aggregates, and
   archive exclusion.
4. #5 — language-aware CLI and resumable execution.
5. #6 — ADR for the fixed English prompt and unparsed verdicts.
6. #7 — 4096-token timing and Grid'5000 reservation/checkpoint changes.
7. #9 — deterministic node sharding and status/collection support.
8. #10 — deterministic multi-site submission and merge verification.
9. #8 — move historical artifacts into explicit archives and remove active
   references from docs and publication inputs.
10. #1 — update the complete documentation and acceptance contract.

Each issue is implemented in its own focused commit with a failing test first,
minimal implementation, focused green tests, and a repository-wide verification
checkpoint before moving to the next issue.

## Verification contract

Every slice must pass its focused unit tests and the relevant acceptance tests.
The final gate is:

```text
ruff format --check .
ruff check .
ty check
pytest tests/unit --cov
pytest tests/property
pytest tests/acceptance
pytest tests/architecture
docs-build --strict
```

The full production model sweep and Hugging Face publication are separate
infrastructure operations. They may be run only when the corresponding hardware
and credentials are available, and their live completion must be verified
independently from the code checks.
