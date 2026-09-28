# ADR-0006: Use one English prompt for every benchmark language

Status: accepted

## Context

The active benchmark contains 85 translated views of the same 300 labelled source
items. The task is classification of land-use and geographic-environment signal, not
translation quality. A language-specific prompt would introduce a second changing
variable and make model-language comparisons harder to interpret.

## Decision

Every run uses the byte-identical English prompt in `data/prompt.txt`, regardless of
the language of the target sentence. The prompt requires exactly one lowercase `yes`
or `no` token. A generation whose final output does not contain a valid verdict is
stored with `predicted=null` and counted as an unparsed error; a non-English answer is
not translated, guessed, or coerced into a label.

All language runs share the same `prompt_sha256`, computed from those exact prompt
bytes. The benchmark CSV and its `benchmark_sha256` remain language-specific.

Rejected alternatives:

- Translating the prompt per language would confound sentence language with task
  wording and model translation quality.
- Mapping localized verdict words such as `oui` or `はい` to English labels would
  change the output contract and hide failures to follow the requested vocabulary.
- Translating model output before parsing would add a second model or translation
  system to the measured pipeline.

The complete `raw_output` is retained in each prediction. Metrics are recomputed from
the stored predictions, so alternative parsing or scoring conventions can be audited
without running the model again. Run metadata records the benchmark language, prompt
digest, benchmark digest, decoding policy, and generation settings.

## Consequences

- Prompt hashes are comparable across all language splits.
- A single prompt digest is expected across all 85 languages; a mismatch means the
  runs do not belong to the same prompt-defined benchmark.
- Unparsed and non-contract outputs remain visible rather than becoming silent labels.
- Published cards and aggregates can report language explicitly while sharing one task
  definition.
- Any future language-specific prompt requires a new decision and a new prompt digest.
