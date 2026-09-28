# Two multilingual reranker benchmark design

## Goal

Add and benchmark `Alibaba-NLP/gte-multilingual-reranker-base` and
`mixedbread-ai/mxbai-rerank-base-v2` on the existing 85-language, 300-item-per-language
benchmark, reporting normalized relevance scores, threshold sweeps, best classification
metrics, inference throughput, and peak CUDA VRAM. Existing local result artifacts must
remain byte-for-byte unchanged.

## Scope and isolation

- The existing `results/` tree is read-only for this work.
- New run checkpoints and derived reports live under the ignored
  `benchmark-runs/rerankers-20260922/` directory.
- Hugging Face publication is staged separately by downloading the current dataset
  snapshot, adding only the new verified artifacts, rebuilding the card from the complete
  staged run set, and uploading the resulting folder.
- No branch-history rewrite, deletion of unrelated branches, or deletion of existing Hub
  files is allowed.

## Architecture

The current scorer pipeline remains the source of truth. Its scorer input is extended to
carry both the rendered scoring prompt and the raw benchmark sentence, allowing each
adapter to use the input form required by its checkpoint without parsing rendered text.

The scoring roster dispatches model IDs to three adapter families:

- the existing causal next-token Qwen reranker;
- GTE's encoder-only sequence-classification model, whose scalar relevance logit is
  sigmoid-normalized for the shared threshold space;
- mxbai's official binary relevance formulation, using the `0`/`1` next-token logit
  difference and its documented sigmoid normalization.

Each adapter returns the normalized `no`/`yes` scores used by the existing decision and
threshold code, plus its native score for auditability. A scoring run records inference
throughput and peak CUDA allocator reservation after model loading. Model instances are
cached across all selected languages.

## Reporting

The existing detailed leaderboard and aggregate report remain available. Scoring reports
also write:

- `threshold_sweep.csv`, containing every configured threshold, macro classification
  metrics, and macro ROC-AUC;
- `scoring_summary.csv`, containing each model's best MCC, F1, balanced accuracy,
  precision, recall, the threshold producing each best value, ROC-AUC, throughput, and
  peak VRAM.

The report and dataset card explain that best threshold metrics are selected on this same
benchmark and are therefore an upper-bound diagnostic, while ROC-AUC is threshold-free.

## Compatibility and cleanup

New optional metadata fields default when reading older run files, so existing artifacts
remain readable without rewriting them. Documentation describes both scorer families and
the performance measurement contract. The existing Transformers-loading test is changed
to use a lightweight module seam instead of importing the full Transformers package for a
loader-selection unit test, preserving coverage while making the local suite complete.

## Verification and release gates

1. RED→GREEN unit tests cover scorer dispatch, score normalization, input construction,
   telemetry, metadata compatibility, summary selection, and CLI report generation.
2. Ruff, ty, the focused tests, the full local test suite, shell syntax checks, and the
   documentation build pass.
3. Each model has exactly 85 valid language checkpoints and 25,500 predictions with the
   pinned benchmark/prompt hashes and resolved model revision.
4. Derived reports are regenerated from stored predictions and checked for the requested
   best metrics, ROC-AUC, throughput, and VRAM fields.
5. The pre-run local `results/` inventory and hashes are unchanged.
6. The staged Hub tree is independently checked before publication; after upload, the Hub
   revision and file inventory are refreshed and the new model rows/artifacts are verified
   live.
