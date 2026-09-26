# Known weaknesses

**The benchmark is small.** 154 items means roughly ±8 points of 95% confidence on an
accuracy near 0.8. Treat gaps smaller than that between two models as noise. Adding
items is the fix; the content-addressed ids make old results joinable to a larger set.

**One prompt, no variants.** Scores conflate a model's judgement with its sensitivity
to this specific wording. A prompt-variant sweep is the natural next step; the run
already pins the prompt digest so such a sweep stays distinguishable.

**Regions are unbalanced.** Antarctica contributes 14 items, most regions one. Per-region
scoring would be unreliable today, which is why nothing slices by region yet even though
the column is carried through.

**Provenance is resolved, not assumed.** `model_revision` is the explicit `--revision`,
else the loaded config's commit hash, else the commit the Hub reports for the model;
only when all three are unknown is it recorded empty, and then with a warning.
`source_commit` comes from `LRB_SOURCE_COMMIT` (set as a Docker build arg, since the
image has no `.git`) or `git rev-parse HEAD`, and is also warned about when unknown. A
model outside the roster runs on Transformers with no pinned revision, with a warning.

**Verdict extraction is a heuristic.** The verdict is the last standalone `yes`/`no` in
an untruncated generation. A model that concludes and then adds a caveat naming the other
verdict would be misread. Raw generations are stored so any such case is visible.
Constrained decoding would
remove the heuristic entirely, at the cost of no longer measuring instruction-following
— see [ADR-0001](adr/0001-greedy-generation.md).

**Mutation testing covers the domain only.** Adapters are covered by tests but not
mutated; their logic is thin, and mutating filesystem code mostly produces equivalent
mutants. Revisit if an adapter grows real branching.

**Four mutants survive by construction.** `zip(..., strict=True)` in `predict_all` is
unreachable defence — the explicit length check above it already guarantees equal
lengths — so mutating `strict` changes nothing. The check is kept for its error message
and `strict=True` for the lint rule that requires it. One further mutant rewrites
`"utf-8"` as `"UTF-8"`, which Python treats as the same encoding. The mutation gate
allows exactly these four.
