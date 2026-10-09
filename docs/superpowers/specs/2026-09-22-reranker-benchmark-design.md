# Design of the two multilingual reranker benchmarks

## Goal

Add `Alibaba-NLP/gte-multilingual-reranker-base` and `mixedbread-ai/mxbai-rerank-base-v2`
to the existing benchmark of 85 languages and 300 items for each language. Benchmark
them. Report these values:

- the normalized relevance scores
- the threshold sweeps
- the best classification metrics
- the inference throughput
- the peak CUDA VRAM

The existing local result artifacts must stay byte-for-byte unchanged.

## Scope and isolation

- The existing `results/` tree is read-only for this work.
- New run checkpoints and derived reports go in the ignored directory
  `benchmark-runs/rerankers-20260922/`.
- The publication to Hugging Face is a separate stage. Do these steps in order:
    1. Download the current dataset snapshot.
    2. Add only the new verified artifacts.
    3. Rebuild the card from the complete staged run set.
    4. Upload the resulting folder.
- Do not rewrite the branch history. Do not delete unrelated branches. Do not delete
  existing Hub files.

## Architecture

The current scorer pipeline stays the source of truth. The scorer input now carries two
values: the rendered scoring prompt and the raw benchmark sentence. Each adapter can then
use the input form that its checkpoint needs. The adapter does not parse rendered text.

The scoring roster sends model ids to three adapter families:

- the existing causal next-token Qwen reranker
- the encoder-only sequence-classification model of GTE. Its scalar relevance logit is
  sigmoid-normalized for the shared threshold space.
- the official binary relevance formulation of mxbai. It uses the `0`/`1` next-token logit
  difference and its documented sigmoid normalization.

Each adapter returns the normalized `no`/`yes` scores that the existing decision code and
threshold code use. It also returns its native score for audit. A scoring run records the
inference throughput. It also records the peak CUDA allocator reservation after the model
loads. The code caches the model instances across all the selected languages.

## Reporting

The existing detailed leaderboard and aggregate report stay available. Scoring reports
also write two files:

- `threshold_sweep.csv`. It has every configured threshold, the macro classification
  metrics, and the macro ROC-AUC.
- `scoring_summary.csv`. It has these values for each model:
    - the best MCC, F1, balanced accuracy, precision, and recall
    - the threshold that gives each best value
    - ROC-AUC
    - throughput
    - peak VRAM

The report and the dataset card explain two points. First, the benchmark selects the best
threshold metrics on this same benchmark. Therefore, they are an upper-bound diagnostic.
Second, ROC-AUC does not depend on a threshold.

## Compatibility and cleanup

New optional metadata fields have defaults when the code reads older run files. Therefore,
the existing artifacts stay readable and the project does not rewrite them.

The documentation describes both scorer families and the contract for performance
measurement.

The existing Transformers-loading test changes. It uses a lightweight module seam. It
does not import the complete Transformers package for a loader-selection unit test. This
keeps the coverage and makes the local suite complete.

## Verification and release gates

1. RED-to-GREEN unit tests cover these items: scorer dispatch, score normalization, input
   construction, telemetry, metadata compatibility, summary selection, and CLI report
   generation.
2. Ruff, ty, the focused tests, the complete local test suite, the shell syntax checks,
   and the documentation build pass.
3. Each model has exactly 85 valid language checkpoints and 25,500 predictions. They have
   the pinned benchmark and prompt hashes and the resolved model revision.
4. The derived reports are regenerated from the stored predictions. Check them for the
   requested best metrics, ROC-AUC, throughput, and VRAM fields.
5. The local `results/` inventory and hashes from before the run are unchanged.
6. Check the staged Hub tree independently before publication. After the upload, refresh
   the Hub revision and the file inventory. Verify the new model rows and artifacts on the
   live Hub.
