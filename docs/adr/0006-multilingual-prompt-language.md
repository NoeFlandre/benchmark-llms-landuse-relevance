# ADR-0006: Use one English prompt for every benchmark language

Status: accepted

## Context

The active benchmark has 85 translated views of the same 300 labelled source items. The
task is the classification of land-use and geographic-environment signal. It is not a
test of translation quality. A prompt for each language adds a second variable that
changes. This makes the comparisons between models and languages harder to interpret.

## Decision

Every run uses the byte-identical English prompt in `data/prompt.txt`. This is true for
all languages of the target sentence. The prompt requires exactly one lowercase `yes` or
`no` token.

If the final output of a generation does not contain a valid verdict, the benchmark
stores the generation with `predicted=null`. It counts the generation as an unparsed
error. The benchmark does not translate a non-English answer. It does not guess a label
and it does not force one.

All language runs have the same `prompt_sha256`. The code computes it from these exact
prompt bytes. The benchmark CSV and its `benchmark_sha256` stay specific to each
language.

The alternatives that the project rejected:

- Translate the prompt for each language. This mixes the language of the sentence with
  the wording of the task and with the translation quality of the model.
- Map localized verdict words, for example `oui` or `はい`, to English labels. This
  changes the output contract. It hides the failures to follow the requested vocabulary.
- Translate the model output before parsing. This adds a second model or a translation
  system to the measured pipeline.

Each prediction keeps the complete `raw_output`. The code recomputes the metrics from the
stored predictions. Therefore, you can audit other parsing or scoring conventions
without a new run of the model. The run metadata records the benchmark language, the
prompt digest, the benchmark digest, the decoding policy, and the generation settings.

## Consequences

- The prompt hashes are comparable across all language splits.
- All 85 languages must have one prompt digest. If a digest is different, the runs do
  not belong to the same benchmark that the prompt defines.
- The unparsed outputs and the outputs that do not follow the contract stay visible.
  They do not become labels without notice.
- The published cards and aggregates can report the language explicitly. They share one
  task definition.
- A future prompt for a specific language needs a new decision and a new prompt digest.
