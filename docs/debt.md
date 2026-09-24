# Known weaknesses

**Per-language samples are small.** Each language has 300 items, so a single-language
accuracy near 0.8 still has wide uncertainty. Treat small gaps between models within
one language as noise. The 85 aligned splits provide a better overall estimate through
macro aggregation; the language-aware ids keep those joins explicit.

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
verdict would be misread. Raw generations are stored so any such case is visible.
Constrained decoding would
remove the heuristic entirely, at the cost of no longer measuring instruction-following
— see [ADR-0001](adr/0001-greedy-generation.md).

**Mutation testing covers the domain only.** Adapters are covered by tests but not
mutated; mutating filesystem and model-runtime code mostly produces equivalent mutants.
The adapters are no longer thin: `hf_scorer.py` (~690 lines, eight scorer families) and
`hf_publish.py` (~580 lines) are now the largest modules. Their branching is covered by
unit tests with fake runtimes, not by mutation.

**No mutant survives, and none is allowed to.** Until the 2026-09 uplift the gate was
silently broken: mutmut's copied workspace lacked the documentation one unit test reads,
so its stats run failed, no mutant was tested, and the survivor count read zero. The
workspace now copies what the tests read, `check_mutants.py` refuses a run that killed
nothing, and the domain avoids constructs whose mutants are equivalent (guarded
`zip(strict=True)` calls, redundant defaults). CI allows zero survivors
(`scripts/check_mutants.py --max-survivors 0`).
