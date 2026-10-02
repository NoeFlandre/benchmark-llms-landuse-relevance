# Multilingual Reranker Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task by task with review checkpoints.

**Goal:** Add `Alibaba-NLP/gte-multilingual-reranker-base` and `mixedbread-ai/mxbai-rerank-base-v2` to the existing 85-language benchmark. Keep the existing results. Collect the thresholded classification metrics, the throughput, and the VRAM. Publish a verified Hugging Face update.

**Architecture:** The current dataset, prompt, generative outputs, and Qwen reranker outputs stay immutable. Extend the scoring domain with a typed `(rendered_prompt, sentence)` input and an optional native score. Add adapters for specific models for Hugging Face. Save the performance telemetry in the run metadata. Generate threshold reports and scorer-summary reports. Stage and publish through the existing report pipeline.

**Tech Stack:** Python 3.11+, uv, pytest, Ruff, ty, PyTorch, Transformers, Hugging Face Hub CLI, Grid5000 runner scripts.

## Tasks

- [ ] Add failing unit tests for these items: typed scorer inputs, native scores, the two scorer roster entries, threshold summaries, telemetry, and compatibility with legacy metadata.
- [ ] Implement the typed scoring contract. Implement the model adapters for GTE, mxbai-rerank-v2, and the existing Qwen scorer. Do not change the behavior of the existing scorer.
- [ ] Add the measurement of time and of CUDA peak memory. Save it in the run metadata. Show it in the leaderboard outputs and the summary outputs.
- [ ] Expand the threshold reporting. Make `report` and `publish` write the CSV files for the thresholds and for the best-metric summary.
- [ ] Update the Hugging Face card generation, the README, and the ignore rules. Remove the Transformers import slowdown that is not necessary from the loader test.
- [ ] Run the RED-to-GREEN focused tests. Then run the complete local quality gates and a no-change check for `results/`.
- [ ] Push the implementation branch. Run both models across all 85 languages on a GPU, with outputs that are resumable and isolated. Collect and validate every run.
- [ ] Merge the validated outputs with the current Hugging Face snapshot. Publish without deletion of existing files. Download the published revision again. Verify the files, rows, hashes, metrics, throughput, and VRAM.
- [ ] Complete the final hygiene of the branch and the worktree. Report the exact artifacts, the verification evidence, and each external limitation.

## Verification gates

- The existing `results/` is byte-for-byte unchanged before and after the work.
- Every new model has 85 languages and 300 scored items for each language.
- Every scoring run records a relevance score for every item, the threshold-sweep metrics, the throughput, and the peak VRAM when CUDA shows it.
- The existing Qwen artifacts and generative artifacts stay present and readable.
- The local tests, Ruff, and ty pass. The remote GPU outputs have valid run metadata and complete rows.
- Download the published Hugging Face revision independently. It must match the validated staged artifacts.

## Commands

```bash
UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest -q
UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync ruff check .
UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync ty check src tests
git diff --check
```
