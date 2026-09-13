# ADR-0002 — Unparsable generations count as errors

**Status:** accepted · 2026-09-13

## Context

Small models sometimes answer "It depends" or produce an empty string. Such a
generation carries no verdict. The benchmark could drop it, coerce it to the
majority class, or count it against the model.

## Decision

Track unparsable generations as their own confusion bucket and include them in the
denominator of every rate. `unparsed_rate` is reported alongside accuracy and F1.

## Consequences

- Accuracy of a model that never answers is 0.0, not undefined and not 0.5.
- Dropping them would silently shrink the benchmark per model, making leaderboard rows
  incomparable; coercing them would credit a model for a coin flip it never made.
- The raw generation is always stored, so any other convention can be recomputed from
  the published results without re-running the models.
