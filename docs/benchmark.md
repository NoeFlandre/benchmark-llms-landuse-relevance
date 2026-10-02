# The benchmark

## Data

`data/translations/manifest.json` is the active benchmark inventory. It has 85 language
configurations. Each configuration has 300 aligned rows from the same golden human-set
source.

Each language file has these fields for each row:

- the translated sentence
- the adjudicated label
- the source item identity
- the language
- the place (`polygon_name`, `h3_cell`, latitude and longitude)
- `region`
- `source_url`

Run `uv run lrb languages` to list the deterministic language inventory and the row
counts.

Only `sentence` and `label` drive the scoring. The benchmark keeps the geographic columns
for audits and future slices.

Each source item has one language-neutral `source_item_id`. A translated item gets a
language-aware `item_id`. The code derives it from the source id and the language.
Therefore, translated rows join across languages and do not collide. Refer to
[ADR-0004](adr/0004-content-addressed-ids.md).

## Prompt

`data/prompt.txt` holds the single template. The characters `{}` show where the target
sentence goes. The substitution is literal. It is not `str.format`. Therefore, braces
inside a sentence pass through unchanged.

Every model receives the byte-identical template. The chat template of the model wraps
it, with one user turn. Each run records the prompt sha256. The fixed English-prompt
policy is in [ADR-0006](adr/0006-multilingual-prompt-language.md).

## Scoring

The positive class is `yes`. A generative model reports accuracy, precision, recall, F1,
balanced accuracy, Matthews correlation, and `unparsed_rate`.

A scoring model gives a normalised relevance score for each item. It can also give a
native score. The report sweeps thresholds and selects the best MCC, F1, balanced
accuracy, precision, recall, and ROC-AUC. The report also records the items per second
and the peak allocated CUDA VRAM of each run.

The adapters keep the native input contract of each model:

- The Qwen rerankers score the yes/no continuation of their chat turn.
- GTE uses a sequence-classification prompt/sentence pair.
- mxbai uses its documented binary query/document continuation.
- Laya uses four JSON sentence fields. Each scorer call has one typed `noul` question
  for each field.
- `LiquidAI/LFM2.5-2.6B@logprob` reads the generative model in one forward pass. The
  input is the generative chat turn with an empty think block at the end. The output is
  `P(yes)` against `P(no)` for the next token.
- The NLI models (bge-m3, mDeBERTa, mmBERT) run the zero-shot classification pipeline.
- GLiClass and GLiNER2 score one land-use label through their SDKs.
- The NLI models, GLiClass, and GLiNER2 share the hypothesis in `data/prompt_zeroshot.txt`.

Each scorer declares its own prompt file. Therefore, the recorded prompt digest matches
the input of the scorer.

The checkpoint context of Laya is 1,024 tokens. Each run records the exact sequence
length, the runtime dtype, the batch, the revision, the device name, and the decision
rule.

A generation that has no standalone `yes`/`no` token is an error. The results keep it
unchanged. Refer to [ADR-0002](adr/0002-unparsed-as-error.md).

## Results

The benchmark saves the results as checkpoints at `results/<language>/<model>.json`.

The detailed `leaderboard.csv` has one row for each model-language pair. `aggregates.csv`
groups these rows by model. It has macro metrics and the F1 spread across languages.
`threshold_sweep.csv` has the threshold grid. `scoring_summary.csv` has one
best-operating-point row for each scoring model.

You can keep new benchmark runs in an ignored `benchmark-runs/` directory. Then the
established `results/` archive does not change.
