# The benchmark

## Data

`data/benchmark.csv` — 154 sentences, 80 labelled `yes` and 74 `no`, drawn from
Wikipedia articles and institutional websites describing 74 regions from Antarctica
to Madagascar. Each row carries the sentence, the adjudicated label, the place it
describes (`polygon_name`, `h3_cell`, latitude/longitude), its `region`, and its
`source_url`.

Only `sentence` and `label` drive scoring; the geographic columns are kept so results
can be sliced by region later.

Items are identified by `sha256(sentence)[:16]`, so predictions join across models and
across benchmark revisions — see [ADR-0004](adr/0004-content-addressed-ids.md).

## Prompt

`data/prompt.txt` holds the single template, with `{}` marking where the target
sentence goes. Substitution is literal, not `str.format`, so braces inside a sentence
pass through untouched.

Every model sees the byte-identical template, wrapped in that model's own chat
template with one user turn. The prompt's sha256 is recorded in every run.

## Scoring

Positive class is `yes`. Reported per model: accuracy, precision, recall, F1,
balanced accuracy, Matthews correlation, and `unparsed_rate`.

A generation that contains no standalone `yes`/`no` token is counted as an error and
kept verbatim in the results — see [ADR-0002](adr/0002-unparsed-as-error.md).
