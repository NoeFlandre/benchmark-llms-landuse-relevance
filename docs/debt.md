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

**Mutation testing targets core decisions.** The mutation run covers the domain and
the result-store scoring and ordering path. Other adapters are exercised by tests but
are not mutated; most of their logic is runtime integration or filesystem plumbing.

**Mutation survivors are pinned by identity.** `mutation-allowlist.txt` records the
exact mutmut ID and review reason for every accepted equivalent survivor. The gate
fails for a new survivor, a stale allowance, or any result that was not killed or
explicitly reviewed. Revisit the list whenever mutmut or the source changes.

**Scores are uncertain at this sample size.** Reports show Wilson 95% intervals for
accuracy, precision and recall, seeded 2,000-resample bootstrap intervals for F1 and
Matthews correlation, and exact paired McNemar p-values against the top-F1 run when full
coverage matches. The intervals describe this 154-item benchmark; they do not estimate
geographic representativeness.
