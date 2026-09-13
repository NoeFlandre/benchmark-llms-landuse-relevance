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

**Mutation testing covers the domain only.** Adapters are covered by tests but not
mutated; their logic is thin, and mutating filesystem code mostly produces equivalent
mutants. Revisit if an adapter grows real branching.
