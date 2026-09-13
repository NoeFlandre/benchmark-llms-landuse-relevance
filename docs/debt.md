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

**`model_revision` may be empty.** It is read from the loaded config's `_commit_hash`,
a private Transformers attribute. If that disappears, runs record an empty revision
rather than failing. Pass `--revision` to pin it explicitly and remove the ambiguity.

**Verdict extraction is a heuristic.** The verdict is the last standalone `yes`/`no` in
an untruncated generation. A model that concludes and then adds a caveat naming the other
verdict would be misread. Raw generations are stored so any such case is visible, and the
strict eight-token run is published alongside as a check. Constrained decoding would
remove the heuristic entirely, at the cost of no longer measuring instruction-following
— see [ADR-0001](adr/0001-greedy-generation.md).

**Mutation testing covers the domain only.** Adapters are covered by tests but not
mutated; their logic is thin, and mutating filesystem code mostly produces equivalent
mutants. Revisit if an adapter grows real branching.

**Three mutants survive by construction.** `zip(..., strict=True)` in `predict_all` is
unreachable defence — the explicit length check above it already guarantees equal
lengths — so mutating `strict` changes nothing. The check is kept for its error message
and `strict=True` for the lint rule that requires it. One further mutant rewrites
`"utf-8"` as `"UTF-8"`, which Python treats as the same encoding. The mutation gate
allows exactly these four.
