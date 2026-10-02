# Multilingual Benchmark Design

**Status:** Approved on 2026-09-20

## Goal

Make the multilingual golden human set the only active benchmark. The benchmark has these
properties:

- 85 language configurations
- 300 aligned items for each language
- language-aware item identity
- language-aware results and leaderboards
- resumable Grid'5000 execution
- an explicit archive boundary for all historical single-language artifacts

## Constraints and decisions

- The active benchmark has all 85 Dataset Viewer configurations. Each configuration has
  its `train` split and 300 rows.
- Every model-language pair is a separate deterministic greedy run. There are no repeated
  greedy runs. There are no standard deviations from repeated identical decoding.
- The prompt stays one English template for every language. Only the sentence changes.
  The expected verdict stays the lowercase `yes` or `no` token.
- A verdict in another language is unparsed. Therefore, it is an error. The full
  `raw_output` stays in the result. Therefore, you can recompute alternative parsing later
  without a new run of the models.
- The historical single-language data, results, documentation, and card content are
  archive-only. The active loaders, globs, reports, leaderboards, documentation, and
  publication must not read them, link to them, or describe them.
- Active commands reject the legacy result records that do not have a language. They do
  not upgrade these records and they do not merge them into the multilingual output.

## Identity and dataset architecture

The upstream rows do not have a source-item ID. Therefore, the vendoring step creates the
identity one time. It does not guess the identity from translated text.

1. Build a canonical source key from the language-neutral fields: `label`,
   `polygon_name`, `h3_cell`, `latitude`, `longitude`, `source`, `region`, and
   `source_url`. Normalize nulls, numeric values, and strings before hashing.
2. Add a deterministic occurrence number when an otherwise identical source key occurs
   more than once in the canonical English configuration.
3. Define `source_item_id` as the first 16 hexadecimal characters of the SHA-256 digest of
   that canonical key plus the occurrence number.
4. Store the generated mapping in `data/translations/manifest.json`. Validate every
   language file against the manifest for the exact 300 source IDs, the source metadata,
   the labels, and the row count. Only `sentence` can be different between languages.
5. Define `item_id` as the first 16 hexadecimal characters of the SHA-256 digest of
   `source_item_id + "\\0" + language`.

Therefore, `BenchmarkItem` carries `source_item_id`, `language`, and `item_id`. The
content hash does not use the translated sentence text now. Therefore, translations stay
joinable. Identical translated text cannot collide across languages.

The active data API gives these items:

- `available_languages(data_root)` - sorted language codes and row counts.
- `load_language_benchmark(data_root, language)` - one validated language.
- `load_manifest(data_root)` - the dataset revision, the split, the file hashes, and the
  source item mapping.
- a deterministic digest of the whole set. It comes from the sorted language file hashes.

The vendored layout is:

```text
data/translations/manifest.json
data/translations/<lang>/v3-final-<lang>.csv
```

## Results and publication

`RunMetadata.language` is required for active records. The code writes a run at:

```text
results/<language>/<filesystem-safe-model-id>.json
```

The result store reads only the active result paths, recursively. It skips every
`archive` path. A record without `language` raises a clear archive-only error.

The detailed leaderboard has one row for each `(model_id, language)` pair. A second
aggregate file has one row for each model. A row has these values:

- an unweighted macro-average of the classification metrics over the available languages
- the language count
- the F1 minimum, maximum, and standard deviation

The detailed rows stay the source of truth for inspection of each language.

The code generates the Hugging Face card and the uploaded folder only from these active
records. It never includes historical files because they exist below the results root.

## CLI and execution architecture

- `lrb languages` lists every available language and row count.
- `lrb run <model>` runs all 85 languages by default.
- `lrb run <model> --language <code>` narrows one run. The command accepts repeated
  selectors and comma-separated selectors.
- `lrb run-all` runs every model on the roster across the selected languages.
- Checkpoints use the key `(model, language)`. The command skips an existing valid result
  deterministically.
- `lrb report` and `lrb publish` accept the same language filter. They write the
  detailed outputs and the aggregate outputs.

The Grid'5000 planner is pure and deterministic. It sorts the model-language matrix. It
assigns node shards by stable index. It can print a status table of complete, running,
and unclaimed pairs. The node submission loads one model for its assigned languages. It
uses the budget of 4096 tokens. It writes one checkpoint for each pair.

The multi-site submission assigns disjoint slices before the submission. It uses a
checked-in site configuration and explicit weights. The collection step verifies four
points:

- The union covers the requested matrix.
- No pair appears two times.
- Every shard reports the same source commit.
- The site-local results are copied off `/home` promptly.

## Order of the implementation by issue

1. #3 - domain identity and manifest contract.
2. #2 - vendored data and validated language loader.
3. #4 - language-aware run records, result paths, leaderboards, aggregates, and archive
   exclusion.
4. #5 - language-aware CLI and resumable execution.
5. #6 - ADR for the fixed English prompt and the unparsed verdicts.
6. #7 - timing for 4096 tokens, and Grid'5000 reservation and checkpoint changes.
7. #9 - deterministic node sharding and support for status and collection.
8. #10 - deterministic multi-site submission and merge verification.
9. #8 - move the historical artifacts into explicit archives. Remove the active
   references from the documentation and from the publication inputs.
10. #1 - update the complete documentation and the acceptance contract.

Implement each issue in its own focused commit. Use this order for each issue: a failing
test, a minimal implementation, focused green tests, and a checkpoint of verification for
the complete repository before the next issue.

## Verification contract

Every slice must pass its focused unit tests and the applicable acceptance tests. The
final gate is:

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

The complete production model sweep and the Hugging Face publication are separate
infrastructure operations. Run them only when the applicable hardware and credentials are
available. Verify their live completion independently from the code checks.
