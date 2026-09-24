# Multilingual Reranker Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task with review checkpoints.

**Goal:** Add `Alibaba-NLP/gte-multilingual-reranker-base` and `mixedbread-ai/mxbai-rerank-base-v2` to the existing 85-language benchmark, preserve existing results, collect thresholded classification metrics plus throughput and VRAM, and publish a verified Hugging Face update.

**Architecture:** Keep the current dataset, prompt, generative outputs, and Qwen reranker outputs immutable. Extend the scoring domain with a typed `(rendered_prompt, sentence)` input and an optional native score; add model-specific Hugging Face adapters; persist performance telemetry in run metadata; generate threshold and scorer-summary reports; and stage/publish through the existing report pipeline.

**Tech Stack:** Python 3.11+, uv, pytest, Ruff, ty, PyTorch, Transformers, Hugging Face Hub CLI, Grid5000 runner scripts.

## Tasks

- [ ] Add failing unit tests for typed scorer inputs, native scores, the two scorer roster entries, threshold summaries, telemetry, and legacy metadata compatibility.
- [ ] Implement the typed scoring contract and model adapters for GTE, mxbai-rerank-v2, and the existing Qwen scorer without changing existing scorer behavior.
- [ ] Add timing and CUDA peak-memory measurement, persist it in run metadata, and expose it in leaderboard/summary outputs.
- [ ] Expand threshold reporting and make `report`/`publish` emit threshold and best-metric summary CSVs.
- [ ] Update the Hugging Face card generation, README, and ignore rules; remove the avoidable Transformers import slowdown in the loader test.
- [ ] Run RED-to-GREEN focused tests, then the full local quality gates and a no-change check for `results/`.
- [ ] Push the implementation branch, run both models across all 85 languages on GPU with resumable isolated outputs, and collect/validate every run.
- [ ] Merge the validated outputs with the current Hugging Face snapshot, publish without deleting existing files, re-download the published revision, and verify files, rows, hashes, metrics, throughput, and VRAM.
- [ ] Complete final branch/worktree hygiene and report exact artifacts, verification evidence, and any external limitation.

## Verification gates

- Existing `results/` is byte-for-byte unchanged before and after the work.
- Every new model has 85 languages and 300 scored items per language.
- Every scoring run records a relevance score for every item, threshold-sweep metrics, throughput, and peak VRAM when CUDA exposes it.
- Existing Qwen and generative artifacts remain present and readable.
- Local tests, Ruff, and ty pass; remote GPU outputs contain valid run metadata and complete rows.
- The published Hugging Face revision is independently downloaded and matches the validated staged artifacts.

## Commands

```bash
UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync pytest -q
UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync ruff check .
UV_CACHE_DIR=/private/tmp/landuse-relevance-bench-uv-cache uv run --no-sync ty check src tests
git diff --check
```
