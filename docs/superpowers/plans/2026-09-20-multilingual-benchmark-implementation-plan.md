# Multilingual Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active single-file benchmark with the vendored 85-language golden human set. Keep the aligned item identity. Produce language-aware results and aggregates. Support resumable node execution and multi-site execution. Archive the historical artifacts outside every active path.

**Architecture:** Keep the existing `cli → adapters → domain` boundary. Add a pure language, identity, and shard layer. Add a manifest and data loader that the adapters own. Add language-aware result persistence and publication. Add thin CLI and Grid'5000 orchestration around these seams. The active readers reject legacy records and skip the archive paths explicitly.

**Tech Stack:** Python 3.11-3.12, Typer, pytest, Hypothesis, Ruff, ty, Hugging Face Dataset Viewer HTTP API, POSIX shell, Grid'5000 OAR, MkDocs Material.

---

## Execution rules

- Work only in `/Users/noeflandre/.codex/worktrees/benchmark-v3-multilingual/benchmark-llms-landuse-relevance` on branch `codex/benchmark-v3-multilingual`.
- Do not change the main checkout.
- For every change of behavior, do these steps in order:
    1. Write the smallest failing test.
    2. Run it and observe the expected failure.
    3. Implement the minimum.
    4. Run the focused test until it is green.
    5. Run the affected suite.
- Commit after each task of the size of an issue. Use a Conventional Commit message.
- Use `UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache` for all `uv` commands.
- Do not run a production model sweep of 459,000 prompts. Do not publish to Hugging Face as part of local code verification. These operations need separate live infrastructure and credentials.

## Task 1: Implement language-aware item identity (#3)

**Files:**
- Modify: `src/landuse_relevance_bench/domain/dataset.py`
- Modify: `tests/unit/test_dataset.py`
- Modify: `tests/property/test_domain_invariants.py`
- Modify: `tests/conftest.py` fixtures. The synthetic rows must include `language` and `source_item_id` when the domain API needs them.

- [ ] **Step 1: Write the failing unit tests.** Add the tests `test_build_item_keeps_source_identity_and_language`, `test_same_source_item_gets_different_item_ids_per_language`, and `test_build_item_rejects_missing_source_identity_or_language`. Assert that two rows with the same `source_item_id` and different languages have equal source identity, unequal `item_id`, and the same sentence-independent join key.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_dataset.py -q
  ```

  Expected: failures, because `BenchmarkItem` and `build_item` do not accept the new fields.

- [ ] **Step 3: Implement the minimum identity contract.** Add `source_item_id` and `language` to `BenchmarkItem`. Validate a non-empty language and a non-empty source identity. Define `item_id_for(source_item_id, language)` as the first 16 hex characters of SHA-256 over `source_item_id + "\\0" + language`. Make `build_item` use these fields. It must not hash the translated sentence text.

- [ ] **Step 4: Run the focused unit tests and property tests until they are green.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_dataset.py tests/property/test_domain_invariants.py -q
  ```

- [ ] **Step 5: Commit the issue.**

  ```bash
  git add src/landuse_relevance_bench/domain/dataset.py tests/unit/test_dataset.py tests/property/test_domain_invariants.py tests/conftest.py
  git commit -m "feat: make benchmark item identity language-aware"
  ```

## Task 2: Vendor and load the 85-language dataset (#2)

**Files:**
- Create: `src/landuse_relevance_bench/adapters/translations.py`
- Create: `scripts/vendor_multilingual_dataset.py`
- Create: `tests/unit/test_translations.py`
- Create: `data/translations/manifest.json`
- Create: `data/translations/<lang>/v3-final-<lang>.csv` for the 85 verified configs.
- Modify: `src/landuse_relevance_bench/adapters/benchmark_csv.py`
- Modify: `tests/unit/test_benchmark_csv.py`
- Modify: `.gitignore` only if it must exclude generated temporary vendor files that it would track otherwise.

- [ ] **Step 1: Write the failing loader tests and manifest tests.** Cover these cases: sorted language discovery; an unknown-language error that lists the available codes; exactly 300 rows for each configured language; manifest digest stability; the digest of each language file; and rejection when a translated file has a missing, extra, duplicated, or mismatched source item.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_translations.py -q
  ```

  Expected: import and API failures, because the translation adapter does not exist.

- [ ] **Step 3: Implement the manifest and the loader.** In the vendoring script, use the Dataset Viewer endpoints `/splits`, `/rows`, and `/size` through a standard-library HTTP client. Normalize the language-neutral fields. Assign the canonical English occurrence numbers. Emit `manifest.json` with the dataset revision, config, split, source IDs, and file hashes. Write deterministic CSV files. The runtime adapter must validate the manifest before it returns `BenchmarkItem` tuples.

- [ ] **Step 4: Download the live source data into the worktree.** Fetch all 85 `train` configs. Verify 300 rows for each config. Verify identical language-neutral source records and labels. Fail if a language would be partially written. Keep the generated files sorted and newline-stable.

- [ ] **Step 5: Run the loader tests against the generated data.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_translations.py tests/unit/test_benchmark_csv.py -q
  ```

- [ ] **Step 6: Commit the dataset and the loader.**

  ```bash
  git add src/landuse_relevance_bench/adapters/translations.py scripts/vendor_multilingual_dataset.py tests/unit/test_translations.py src/landuse_relevance_bench/adapters/benchmark_csv.py tests/unit/test_benchmark_csv.py data/translations
  git commit -m "feat: vendor multilingual benchmark configurations"
  ```

## Task 3: Carry language through records, files, leaderboards, and aggregates (#4)

**Files:**
- Modify: `src/landuse_relevance_bench/domain/records.py`
- Modify: `src/landuse_relevance_bench/adapters/results_store.py`
- Modify: `src/landuse_relevance_bench/adapters/pipeline.py`
- Modify: `src/landuse_relevance_bench/adapters/hf_publish.py`
- Create or modify: `tests/unit/test_aggregates.py`
- Modify: `tests/unit/test_records.py`, `tests/unit/test_results_store.py`, `tests/unit/test_pipeline.py`, `tests/unit/test_hf_publish.py`

- [ ] **Step 1: Write the failing record tests and path tests.** Require `RunMetadata.language`. Assert the round-trip serialization. Assert that a missing language raises a clear archive-only error. Assert that `run_filename(model, language)` produces `results/<language>/<model>.json`. Assert that two languages cannot overwrite each other. Assert that the active recursive readers skip `results/archive/`.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_records.py tests/unit/test_results_store.py tests/unit/test_pipeline.py tests/unit/test_hf_publish.py -q
  ```

- [ ] **Step 3: Implement the language-required metadata and the persistence.** Add `language` to `RunMetadata`. Reject absent legacy metadata. Pass the language through `RunRequest` and `execute`. Write nested language/model paths. Make `read_runs` and `read_published_runs` read recursively only the active paths. They must exclude any archive component.

- [ ] **Step 4: Add the detailed rows and the aggregates.** Include `language` in the detailed leaderboard rows. Add deterministic aggregate rows grouped by model. Each row has the language count, the macro averages for every classification metric, and the F1 minimum, maximum, and standard deviation. Write `aggregates.csv` beside `leaderboard.csv`.

- [ ] **Step 5: Make the publication language-aware.** Include the language and the benchmark-set metadata in the card rows. Recompute the metrics from the predictions. Refuse legacy records and archive records. Make sure that the upload folder cannot contain an archive path.

- [ ] **Step 6: Run the focused suite until it is green. Then commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_records.py tests/unit/test_results_store.py tests/unit/test_pipeline.py tests/unit/test_hf_publish.py tests/unit/test_aggregates.py -q
  git add src/landuse_relevance_bench/domain/records.py src/landuse_relevance_bench/adapters/results_store.py src/landuse_relevance_bench/adapters/pipeline.py src/landuse_relevance_bench/adapters/hf_publish.py tests/unit
  git commit -m "feat: persist language-aware benchmark results"
  ```

## Task 4: Make the CLI multilingual and resumable (#5)

**Files:**
- Modify: `src/landuse_relevance_bench/cli.py`
- Modify: `src/landuse_relevance_bench/adapters/pipeline.py`
- Create or modify: `tests/unit/test_cli.py`
- Modify: `tests/acceptance/features/benchmark.feature`
- Modify: `tests/acceptance/test_benchmark_feature.py`

- [ ] **Step 1: Write the failing CLI tests.** Assert these points:
    - `lrb languages` lists all available codes and 300 rows.
    - `lrb run model` runs every language by default.
    - Repeated `--language` values and comma-separated `--language` values narrow the run.
    - A valid `(model, language)` result is skipped.
    - The report accepts a language filter and writes both CSV files.
    - Publish refuses archive-only records and legacy records.

- [ ] **Step 2: Run the CLI tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_cli.py tests/acceptance/test_benchmark_feature.py -q
  ```

- [ ] **Step 3: Implement the language selection.** Replace the active single-CSV default with the translation data root. Normalize the selectors into sorted unique language codes. Loop one `RunRequest` for each selected language. Keep the checkpoint behavior for each pair. For test-only temporary benchmark injection, use the new data-root fixture. Do not restore the historical default.

- [ ] **Step 4: Implement `languages`, the report filters, and the aggregate output.** Print deterministic language/count rows. Filter the stored runs before ranking. Write `leaderboard.csv` and `aggregates.csv`.

- [ ] **Step 5: Run the acceptance tests and CLI tests until they are green. Then commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_cli.py tests/acceptance -q
  git add src/landuse_relevance_bench/cli.py src/landuse_relevance_bench/adapters/pipeline.py tests/unit/test_cli.py tests/acceptance
  git commit -m "feat: add multilingual benchmark CLI workflows"
  ```

## Task 5: Record the fixed prompt policy (#6)

**Files:**
- Create: `docs/adr/0006-multilingual-prompt-language.md`
- Modify: `docs/benchmark.md`, `docs/index.md`, `README.md`
- Create or modify: `tests/unit/test_prompt_file.py` and the acceptance wording where the prompt contract is asserted.

- [ ] **Step 1: Add a failing documentation/contract test.** The test checks that the ADR exists. It checks that the ADR states the English prompt, the exact lowercase `yes`/`no` outputs, the non-English verdicts as unparsed errors, and the recomputation from the stored `raw_output`.

- [ ] **Step 2: Run the focused documentation contract test and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_prompt_file.py -q
  ```

- [ ] **Step 3: Add ADR 0006 and update the active benchmark documentation.** Describe the multilingual set. Do not mention the historical artifacts.

- [ ] **Step 4: Run the focused test. Then commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_prompt_file.py -q
  git add docs/adr/0006-multilingual-prompt-language.md docs/benchmark.md docs/index.md README.md tests/unit/test_prompt_file.py
  git commit -m "docs: record multilingual prompt policy"
  ```

## Task 6: Align the generation budget and the per-language Grid’5000 checkpoints (#7)

**Files:**
- Modify: `src/landuse_relevance_bench/adapters/pipeline.py`
- Modify: `scripts/g5k_node_run.sh`, `scripts/g5k_both_configs.sh`, `docs/grid5000.md`
- Modify: `tests/unit/test_pipeline.py`
- Create or modify: `tests/scripts/test_g5k_node_run.py` if you extract the shell behavior into a planner that you can test.

- [ ] **Step 1: Write a failing test.** It asserts that the active default generation budget is 4096. It also asserts that a request records that budget.

- [ ] **Step 2: Run the focused test and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_pipeline.py -q
  ```

- [ ] **Step 3: Set the active default to 4096. Update the node script.** The node script must iterate over the model-language pairs, save the nested result files as checkpoints, and print the measured pair count before it runs.

- [ ] **Step 4: Add a dry-run and sizing path.** It reports the rows, pairs, estimated prompts, batch size, and token budget. It does not load a model. Document the measurement procedure and the arithmetic of the reservation.

- [ ] **Step 5: Run the focused tests and the shell syntax checks. Then commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_pipeline.py -q
  bash -n scripts/g5k_node_run.sh scripts/g5k_both_configs.sh
  git add src/landuse_relevance_bench/adapters/pipeline.py scripts/g5k_node_run.sh scripts/g5k_both_configs.sh docs/grid5000.md tests/unit/test_pipeline.py
  git commit -m "feat: checkpoint multilingual runs at the published budget"
  ```

## Task 7: Implement deterministic node sharding (#9)

**Files:**
- Create: `src/landuse_relevance_bench/domain/sharding.py`
- Create: `tests/unit/test_sharding.py`
- Modify: `src/landuse_relevance_bench/cli.py`
- Modify: `scripts/g5k_node_run.sh`, `scripts/g5k_submit.sh`, `docs/grid5000.md`

- [ ] **Step 1: Write the failing sharding tests.** Assert these points: stable sorted model-language pairs; disjoint and complete `shard_index/shard_count` partitions; rejection of invalid shard values; and deterministic status classification from the existing result files.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_sharding.py -q
  ```

- [ ] **Step 3: Implement the pure planner and the status functions.** Do not import the filesystem or subprocess in the domain layer.

- [ ] **Step 4: Add `--shard-index`, `--shard-count`, and a status command** to the CLI and the node wrapper. Make sure that each node loads one model for its assigned languages. Make sure that it writes only its assigned pair checkpoints.

- [ ] **Step 5: Run the tests and the shell syntax checks. Then commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_sharding.py tests/unit/test_cli.py -q
  bash -n scripts/g5k_node_run.sh scripts/g5k_submit.sh
  git add src/landuse_relevance_bench/domain/sharding.py tests/unit/test_sharding.py src/landuse_relevance_bench/cli.py scripts/g5k_node_run.sh scripts/g5k_submit.sh docs/grid5000.md
  git commit -m "feat: shard multilingual sweeps deterministically"
  ```

## Task 8: Implement deterministic multi-site submission and merge verification (#10)

**Files:**
- Create: `scripts/g5k_sites.json`
- Create: `scripts/g5k_submit_multisite.sh`
- Create: `scripts/g5k_collect.py`
- Create: `tests/unit/test_g5k_collect.py`
- Modify: `scripts/g5k_submit.sh`, `docs/grid5000.md`

- [ ] **Step 1: Write the failing collector tests and planner tests.** Assert these points: the weighted deterministic site slices are disjoint and complete; duplicate pairs fail; mixed source commits fail; missing pairs are reported; and a complete union is accepted.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_g5k_collect.py -q
  ```

- [ ] **Step 3: Implement the site configuration and the dry-run submission script.** Require an explicit site list. Run `usagepolicycheck -t` for each site. Assign fixed disjoint shard ranges before the SSH/OAR submission. Print every remote command in dry-run mode.

- [ ] **Step 4: Implement the collection verification.** Copy the result tree of each site off the site-local `/home`. Validate the source commits and the pair coverage. Reject duplicate pairs and conflicting pairs. Merge only after the validation.

- [ ] **Step 5: Run the tests and the shell syntax checks. Then commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_g5k_collect.py -q
  bash -n scripts/g5k_submit_multisite.sh scripts/g5k_submit.sh
  git add scripts/g5k_sites.json scripts/g5k_submit_multisite.sh scripts/g5k_collect.py tests/unit/test_g5k_collect.py scripts/g5k_submit.sh docs/grid5000.md
  git commit -m "feat: coordinate multilingual sweeps across Grid5000 sites"
  ```

## Task 9: Archive the historical artifacts and remove the active references (#8)

**Files:**
- Move: `data/benchmark.csv` to an explicit archive location with a checksum README.
- Move: the existing historical JSON, CSV, and card files under `results/archive/`.
- Modify: `README.md`, `docs/benchmark.md`, `docs/results.md`, `docs/index.md`, `docs/grid5000.md`, `results/README.md`
- Modify: `tests/conftest.py`, the tests that load `data/benchmark.csv` now, and the publication tests.
- Create: the archive provenance README. It has the historical dataset digest and the file inventory.

- [ ] **Step 1: Write the failing archive-boundary tests.** Assert these points: active data discovery cannot find the archived single-file dataset; the active result readers skip every archive path; the active cards contain only multilingual metadata; and no active documentation path contains the historical benchmark path or the old item-count claim.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_hf_publish.py tests/unit/test_results_store.py tests/acceptance -q
  ```

- [ ] **Step 3: Move the exact tracked historical files** into `results/archive/` and the explicit data archive. Keep the checksums and add provenance. Do not delete any historical content.

- [ ] **Step 4: Rewrite the active documentation and the fixtures** around `data/translations/manifest.json`. Remove the active references to historical files and old result paths.

- [ ] **Step 5: Run the boundary tests. Then commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit tests/acceptance -q
  git add -A
  git commit -m "refactor: archive historical benchmark artifacts"
  ```

## Task 10: Complete the multilingual epic and the acceptance contract (#1)

**Files:**
- Modify: `README.md`, `docs/architecture.md`, `docs/benchmark.md`, `docs/results.md`, `docs/grid5000.md`, `docs/debt.md`
- Modify: `tests/acceptance/features/benchmark.feature`, `tests/acceptance/test_benchmark_feature.py`
- Modify: `.github/workflows/ci.yml`, `Makefile` only where the new checks or generated artifacts need it.

- [ ] **Step 1: Write the failing acceptance scenarios** for these items: 85-language discovery; aligned source IDs; one result for each model-language pair; archive exclusion; macro aggregation; and deterministic shard coverage.

- [ ] **Step 2: Run the acceptance tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/acceptance -q
  ```

- [ ] **Step 3: Update the active documentation and the architecture diagram.** Describe only the multilingual benchmark, its manifest, the result layout, the aggregates, and the execution workflow.

- [ ] **Step 4: Run the complete local quality gates.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache make check
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --with mkdocs-material mkdocs build --strict
  ```

- [ ] **Step 5: Commit the integration of the epic.**

  ```bash
  git add README.md docs .github/workflows/ci.yml Makefile tests/acceptance
  git commit -m "feat: complete multilingual benchmark migration"
  ```

## Final verification and handoff

- [ ] Run `git diff --check`. Inspect the complete diff against `main`.
- [ ] Run the complete `make check` and the strict docs build again after the final commit.
- [ ] Run `git status --short --branch`. Make sure that only intentional committed changes remain.
- [ ] Verify the 85-language manifest inventory, the 25,500 total rows, the hashes of each file, and the digest of the whole set from the generated data.
- [ ] Verify that the active result discovery excludes archive paths. Verify that it rejects metadata without a language.
- [ ] Verify that the shard partitions are complete and disjoint for representative shard counts.
- [ ] Report the completion of the code and tests separately from any production Grid'5000 operation or Hugging Face publication that nobody ran.
