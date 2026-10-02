# ADR-0002 - Unparsable generations are errors

**Status:** accepted · 2026-09-13

## Context

Small models sometimes answer "It depends" or write an empty string. Such a generation
has no verdict. The benchmark has three options. It can drop the generation. It can
change it to the majority class. It can count it against the model.

## Decision

Track the unparsable generations as their own confusion bucket. Include them in the
denominator of every rate. Report `unparsed_rate` together with accuracy and F1.

## Consequences

- The accuracy of a model that never answers is 0.0. It is not undefined and it is not
  0.5.
- If the benchmark drops these generations, the benchmark becomes smaller for each model
  without notice. The rows of the leaderboard are then not comparable. If the benchmark
  changes them to the majority class, it gives the model credit for a coin flip that the
  model never made.
- The benchmark always stores the raw generation. Therefore, you can recompute any other
  convention from the published results. You do not need to run the models again.
