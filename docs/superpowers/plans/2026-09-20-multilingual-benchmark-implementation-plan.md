# Multilingual Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active single-file benchmark with the vendored 85-language golden human set, preserve aligned item identity, produce language-aware results and aggregates, support resumable node and multi-site execution, and archive historical artifacts outside every active path.

**Architecture:** Keep the existing `cli → adapters → domain` boundary. Add a pure language/identity and shard layer, an adapter-owned manifest/data loader, language-aware result persistence and publication, and thin CLI/Grid'5000 orchestration around those seams. Active readers reject legacy records and explicitly skip archive paths.

**Tech Stack:** Python 3.11–3.12, Typer, pytest, Hypothesis, Ruff, ty, Hugging Face Dataset Viewer HTTP API, POSIX shell, Grid'5000 OAR, MkDocs Material.

---

## Execution rules

- Work only in `/Users/noeflandre/.codex/worktrees/benchmark-v3-multilingual/benchmark-llms-landuse-relevance` on branch `codex/benchmark-v3-multilingual`.
- Keep the main checkout untouched.
- For every behavior change: write the smallest failing test, run it and observe the expected failure, implement the minimum, run the focused test green, then run the affected suite.
- Commit after each issue-sized task with a Conventional Commit message.
- Use `UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache` for all `uv` commands.
- Do not run a production 459,000-prompt model sweep or publish to Hugging Face as part of local code verification. Those require separate live infrastructure and credentials.

## Task 1: Implement language-aware item identity (#3)

**Files:**
- Modify: `src/landuse_relevance_bench/domain/dataset.py`
- Modify: `tests/unit/test_dataset.py`
- Modify: `tests/property/test_domain_invariants.py`
- Modify: `tests/conftest.py` fixtures so synthetic rows include `language` and `source_item_id` when the domain API requires them.

- [ ] **Step 1: Write the failing unit tests.** Add tests named `test_build_item_keeps_source_identity_and_language`, `test_same_source_item_gets_different_item_ids_per_language`, and `test_build_item_rejects_missing_source_identity_or_language`. Assert that two rows with the same `source_item_id` and different languages have equal source identity, unequal `item_id`, and the same sentence-independent join key.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_dataset.py -q
  ```

  Expected: failures because `BenchmarkItem` and `build_item` do not accept the new fields.

- [ ] **Step 3: Implement the minimum identity contract.** Add `source_item_id` and `language` to `BenchmarkItem`; validate non-empty language and source identity; define `item_id_for(source_item_id, language)` as the first 16 hex characters of SHA-256 over `source_item_id + "\\0" + language`; and make `build_item` use those fields instead of hashing translated sentence text.

- [ ] **Step 4: Run the focused unit and property tests green.**

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
- Modify: `.gitignore` only if generated temporary vendor files would otherwise be tracked.

- [ ] **Step 1: Write failing loader and manifest tests.** Cover: sorted language discovery; unknown-language error listing available codes; exactly 300 rows per configured language; manifest digest stability; per-language file digest; and rejection when a translated file has a missing, extra, duplicated, or mismatched source item.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_translations.py -q
  ```

  Expected: import/API failures because the translation adapter does not exist.

- [ ] **Step 3: Implement the manifest and loader.** Use the Dataset Viewer `/splits`, `/rows`, and `/size` endpoints through a standard-library HTTP client in the vendoring script. Normalize the language-neutral fields, assign the canonical English occurrence numbers, emit `manifest.json` with dataset revision/config/split/source IDs/file hashes, and write deterministic CSV files. The runtime adapter must validate the manifest before returning `BenchmarkItem` tuples.

- [ ] **Step 4: Download the live source data into the worktree.** Fetch all 85 `train` configs, verify 300 rows each, verify identical language-neutral source records and labels, and fail rather than partially writing a language. Keep the generated files sorted and newline-stable.

- [ ] **Step 5: Run loader tests against the generated data.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_translations.py tests/unit/test_benchmark_csv.py -q
  ```

- [ ] **Step 6: Commit the dataset and loader.**

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

- [ ] **Step 1: Write failing record and path tests.** Require `RunMetadata.language`; assert round-trip serialization; assert a missing language raises a clear archive-only error; assert `run_filename(model, language)` produces `results/<language>/<model>.json`; assert two languages cannot overwrite each other; and assert active recursive readers skip `results/archive/`.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_records.py tests/unit/test_results_store.py tests/unit/test_pipeline.py tests/unit/test_hf_publish.py -q
  ```

- [ ] **Step 3: Implement language-required metadata and persistence.** Add `language` to `RunMetadata`, reject absent legacy metadata, pass language through `RunRequest` and `execute`, write nested language/model paths, and make `read_runs`/`read_published_runs` recursively read only active paths while excluding any archive component.

- [ ] **Step 4: Add detailed rows and aggregates.** Include `language` in detailed leaderboard rows. Add deterministic aggregate rows grouped by model with language count, macro averages for every classification metric, and F1 min/max/standard deviation. Write `aggregates.csv` beside `leaderboard.csv`.

- [ ] **Step 5: Make publication language-aware.** Include language and benchmark-set metadata in card rows, recompute metrics from predictions, refuse legacy/archive records, and ensure the upload folder cannot contain an archive path.

- [ ] **Step 6: Run the focused suite green and commit.**

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

- [ ] **Step 1: Write failing CLI tests.** Assert `lrb languages` lists all available codes and 300 rows; `lrb run model` invokes every language by default; repeated and comma-separated `--language` values narrow the run; a valid `(model, language)` result is skipped; report accepts a language filter and writes both CSVs; and publish refuses archive-only/legacy records.

- [ ] **Step 2: Run CLI tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_cli.py tests/acceptance/test_benchmark_feature.py -q
  ```

- [ ] **Step 3: Implement language selection.** Replace the active single-CSV default with the translation data root, normalize selectors into sorted unique language codes, loop one `RunRequest` per selected language, and preserve per-pair checkpoint behavior. Keep test-only temporary benchmark injection through the new data-root fixture rather than restoring the historical default.

- [ ] **Step 4: Implement `languages`, report filters, and aggregate output.** Print deterministic language/count rows, filter stored runs before ranking, and write `leaderboard.csv` plus `aggregates.csv`.

- [ ] **Step 5: Run acceptance and CLI tests green and commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_cli.py tests/acceptance -q
  git add src/landuse_relevance_bench/cli.py src/landuse_relevance_bench/adapters/pipeline.py tests/unit/test_cli.py tests/acceptance
  git commit -m "feat: add multilingual benchmark CLI workflows"
  ```

## Task 5: Record the fixed prompt policy (#6)

**Files:**
- Create: `docs/adr/0006-multilingual-prompt-language.md`
- Modify: `docs/benchmark.md`, `docs/index.md`, `README.md`
- Create or modify: `tests/unit/test_prompt_file.py` and acceptance wording where the prompt contract is asserted.

- [ ] **Step 1: Add a failing documentation/contract test** that checks the ADR exists and states the English prompt, exact lowercase `yes`/`no` outputs, non-English verdicts as unparsed errors, and recomputation from stored `raw_output`.

- [ ] **Step 2: Run the focused documentation contract test and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_prompt_file.py -q
  ```

- [ ] **Step 3: Add ADR 0006 and update active benchmark documentation** to describe the multilingual set without mentioning historical artifacts.

- [ ] **Step 4: Run the focused test and commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_prompt_file.py -q
  git add docs/adr/0006-multilingual-prompt-language.md docs/benchmark.md docs/index.md README.md tests/unit/test_prompt_file.py
  git commit -m "docs: record multilingual prompt policy"
  ```

## Task 6: Align generation budget and per-language Grid’5000 checkpoints (#7)

**Files:**
- Modify: `src/landuse_relevance_bench/adapters/pipeline.py`
- Modify: `scripts/g5k_node_run.sh`, `scripts/g5k_both_configs.sh`, `docs/grid5000.md`
- Modify: `tests/unit/test_pipeline.py`
- Create or modify: `tests/scripts/test_g5k_node_run.py` if shell behavior is extracted into a testable planner.

- [ ] **Step 1: Write a failing test** asserting the active default generation budget is 4096 and a request records that budget.

- [ ] **Step 2: Run the focused test and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_pipeline.py -q
  ```

- [ ] **Step 3: Set the active default to 4096 and update the node script** to iterate model-language pairs, checkpoint nested result files, and print the measured pair count before execution.

- [ ] **Step 4: Add a dry-run/sizing path** that reports rows, pairs, estimated prompts, batch size, and token budget without loading a model. Document the measurement procedure and reservation arithmetic.

- [ ] **Step 5: Run focused tests and shell syntax checks, then commit.**

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

- [ ] **Step 1: Write failing sharding tests.** Assert stable sorted model-language pairs, disjoint and complete `shard_index/shard_count` partitions, rejection of invalid shard values, and deterministic status classification from existing result files.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_sharding.py -q
  ```

- [ ] **Step 3: Implement the pure planner and status functions** without filesystem or subprocess imports in the domain layer.

- [ ] **Step 4: Add `--shard-index`, `--shard-count`, and a status command** to the CLI/node wrapper. Ensure each node loads one model for its assigned languages and writes only its assigned pair checkpoints.

- [ ] **Step 5: Run tests, shell syntax checks, and commit.**

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

- [ ] **Step 1: Write failing collector/planner tests.** Assert weighted deterministic site slices are disjoint and complete, duplicate pairs fail, mixed source commits fail, missing pairs are reported, and a complete union is accepted.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_g5k_collect.py -q
  ```

- [ ] **Step 3: Implement the site configuration and dry-run submission script.** Require an explicit site list, run `usagepolicycheck -t` per site, assign fixed disjoint shard ranges before SSH/OAR submission, and print every remote command in dry-run mode.

- [ ] **Step 4: Implement collection verification.** Copy each site result tree off site-local `/home`, validate source commits and pair coverage, reject duplicate or conflicting pairs, and merge only after validation.

- [ ] **Step 5: Run tests and shell syntax checks, then commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_g5k_collect.py -q
  bash -n scripts/g5k_submit_multisite.sh scripts/g5k_submit.sh
  git add scripts/g5k_sites.json scripts/g5k_submit_multisite.sh scripts/g5k_collect.py tests/unit/test_g5k_collect.py scripts/g5k_submit.sh docs/grid5000.md
  git commit -m "feat: coordinate multilingual sweeps across Grid5000 sites"
  ```

## Task 9: Archive historical artifacts and remove active references (#8)

**Files:**
- Move: `data/benchmark.csv` to an explicit archive location with a checksum README.
- Move: existing historical JSON/CSV/card files under `results/archive/`.
- Modify: `README.md`, `docs/benchmark.md`, `docs/results.md`, `docs/index.md`, `docs/grid5000.md`, `results/README.md`
- Modify: `tests/conftest.py`, tests that currently load `data/benchmark.csv`, and publication tests.
- Create: archive provenance README with the historical dataset digest and file inventory.

- [ ] **Step 1: Write failing archive-boundary tests.** Assert active data discovery cannot find the archived single-file dataset, active result readers skip every archive path, active cards contain only multilingual metadata, and no active documentation path contains the historical benchmark path or old item-count claim.

- [ ] **Step 2: Run the focused tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit/test_hf_publish.py tests/unit/test_results_store.py tests/acceptance -q
  ```

- [ ] **Step 3: Move the exact tracked historical files** into `results/archive/` and the explicit data archive, preserving checksums and adding provenance. Do not delete any historical content.

- [ ] **Step 4: Rewrite active documentation and fixtures** around `data/translations/manifest.json`; remove active references to historical files and old result paths.

- [ ] **Step 5: Run boundary tests and commit.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/unit tests/acceptance -q
  git add -A
  git commit -m "refactor: archive historical benchmark artifacts"
  ```

## Task 10: Complete the multilingual epic and acceptance contract (#1)

**Files:**
- Modify: `README.md`, `docs/architecture.md`, `docs/benchmark.md`, `docs/results.md`, `docs/grid5000.md`, `docs/debt.md`
- Modify: `tests/acceptance/features/benchmark.feature`, `tests/acceptance/test_benchmark_feature.py`
- Modify: `.github/workflows/ci.yml`, `Makefile` only where the new checks or generated artifacts require it.

- [ ] **Step 1: Write failing acceptance scenarios** for 85-language discovery, aligned source IDs, one result per model-language pair, archive exclusion, macro aggregation, and deterministic shard coverage.

- [ ] **Step 2: Run acceptance tests and observe RED.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest tests/acceptance -q
  ```

- [ ] **Step 3: Update the active docs and architecture diagram** to describe only the multilingual benchmark, its manifest, result layout, aggregates, and execution workflow.

- [ ] **Step 4: Run the full local quality gates.**

  ```bash
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache make check
  UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --with mkdocs-material mkdocs build --strict
  ```

- [ ] **Step 5: Commit the epic integration.**

  ```bash
  git add README.md docs .github/workflows/ci.yml Makefile tests/acceptance
  git commit -m "feat: complete multilingual benchmark migration"
  ```

## Final verification and handoff

- [ ] Run `git diff --check` and inspect the complete diff against `main`.
- [ ] Run the full `make check` and strict docs build again after the final commit.
- [ ] Run `git status --short --branch` and confirm only intentional committed changes remain.
- [ ] Verify the 85-language manifest inventory, 25,500 total rows, per-file hashes, and whole-set digest from the generated data.
- [ ] Verify active result discovery excludes archive paths and rejects metadata without language.
- [ ] Verify shard partitions are complete/disjoint for representative shard counts.
- [ ] Report code/test completion separately from any unrun production Grid'5000 or Hugging Face publication operation.
