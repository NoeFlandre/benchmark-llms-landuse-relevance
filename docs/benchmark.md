# The benchmark

## Data

`data/benchmark.csv` — 154 sentences, 80 labelled `yes` and 74 `no`, drawn from
Wikipedia articles and institutional websites describing 74 regions from Antarctica
to Madagascar. Each row carries the sentence, the adjudicated label, the place it
describes (`polygon_name`, `h3_cell`, latitude/longitude), its `region`, and its
`source_url`.

Only `sentence` and `label` drive scoring; the geographic columns are kept so results
can be sliced by region later.

The repository's software license is MIT. Sentence quotations remain subject to the
license and reuse terms of their original source; the `source_url` column provides the
attribution link. Check those source terms before reusing or redistributing text. The
software license does not grant rights to upstream quotations.

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
balanced accuracy, Matthews correlation, and `unparsed_rate`. Accuracy, precision and
recall include Wilson 95% intervals; F1 and Matthews correlation include seeded
bootstrap intervals. Paired exact McNemar p-values compare against the top-F1 run when
the benchmark and full item coverage match.

A generation is parsed as an exact answer, then a leading answer, then the final
standalone `yes`/`no` token as a reasoning-first fallback. The selected rule is stored
as `parse_mode`; a generation without a verdict is left unparsed. Raw text remains in
the result files for auditing — see [ADR-0002](adr/0002-unparsed-as-error.md) and
[ADR-0008](adr/0008-verdict-parsing.md).
