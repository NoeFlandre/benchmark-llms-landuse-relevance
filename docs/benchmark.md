# The benchmark

## Data

`data/translations/manifest.json` is the active benchmark inventory: 85 language
configurations, each with 300 aligned rows from the same golden human-set source.
Each language file carries the translated sentence, adjudicated label, source item
identity, language, place (`polygon_name`, `h3_cell`, latitude/longitude), `region`,
and `source_url`.

Run `uv run lrb languages` to list the deterministic language inventory and row counts.
Only `sentence` and `label` drive scoring; the geographic columns are retained for
auditing and future slices.

Each source item has one language-neutral `source_item_id`. A translated item gets a
language-aware `item_id` derived from that source ID and its language, so translated
rows join across languages without collisions — see [ADR-0004](adr/0004-content-addressed-ids.md).

## Prompt

`data/prompt.txt` holds the single template, with `{}` marking where the target
sentence goes. Substitution is literal, not `str.format`, so braces inside a sentence
pass through untouched.

Every model sees the byte-identical template, wrapped in that model's own chat
template with one user turn. The prompt's sha256 is recorded in every run.
The fixed English-prompt policy is documented in
[ADR-0006](adr/0006-multilingual-prompt-language.md).

## Scoring

Positive class is `yes`. Generative models report accuracy, precision, recall, F1,
balanced accuracy, Matthews correlation, and `unparsed_rate`. Scoring models emit a
normalised relevance score and optional native score for every item; their report
sweeps thresholds and selects the best MCC, F1, balanced accuracy, precision, recall,
and ROC-AUC. It also records items/second and peak allocated CUDA VRAM per run.

The adapters keep model-native input contracts: GTE uses a sequence-classification
prompt/sentence pair, mxbai uses its documented binary query/document continuation,
and Laya uses four JSON sentence fields with one typed `noul` question per field in
each scorer call. Laya's checkpoint context is 1,024 tokens; each run records the
exact sequence length, runtime dtype, batch, revision, and decision rule used.

A generation that contains no standalone `yes`/`no` token is counted as an error and
kept verbatim in the results — see [ADR-0002](adr/0002-unparsed-as-error.md).

## Results

Results are checkpointed at `results/<language>/<model>.json`. The detailed
`leaderboard.csv` has one row per model-language pair; `aggregates.csv` groups those
rows by model with macro metrics and F1 spread across languages. `threshold_sweep.csv`
contains the threshold grid, while `scoring_summary.csv` contains one best-operating-
point row per scoring model. New benchmark runs may be kept in an ignored
`benchmark-runs/` directory so the established `results/` archive is not modified.
