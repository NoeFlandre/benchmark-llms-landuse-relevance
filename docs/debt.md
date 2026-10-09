# Known weaknesses

**The samples for each language are small.** Each language has 300 items. An accuracy of
near 0.8 for one language has a wide uncertainty. Treat small gaps between models in one
language as noise. The 85 aligned splits give a better overall estimate through macro
aggregation. The language-aware ids keep these joins explicit.

**There is one prompt and there are no variants.** A score mixes the judgement of the
model with its sensitivity to this specific wording. The next step is a sweep of prompt
variants. The run already pins the prompt digest. Therefore, such a sweep stays
distinguishable.

**The regions are not balanced.** Antarctica has 14 items. Most regions have one item.
Scoring by region is not reliable today. For this reason, nothing slices by region yet.
The benchmark still carries the column.

**`model_revision` can be empty.** The code reads it from `_commit_hash` of the loaded
config. This is a private Transformers attribute. If it goes away, runs record an empty
revision. They do not fail. To remove the ambiguity, pass `--revision` to pin the
revision.

**Verdict extraction is a heuristic.** The verdict is the last standalone `yes`/`no` in
an untruncated generation. The parser reads a wrong verdict if a model gives its
conclusion and then adds a caveat that names the other verdict. The benchmark stores the
raw generations. Therefore, you can see each such case.

Constrained decoding removes the heuristic fully. But then the benchmark does not measure
instruction-following. Refer to [ADR-0001](adr/0001-greedy-generation.md).

**Mutation testing covers only the domain and `results_store.py`.** Tests cover the other
adapters, but the mutation step does not mutate them. Mutants of filesystem code and
model-runtime code are mostly equivalent mutants.

The adapters are not thin now. `hf_scorer.py` (~730 lines, eight scorer families) and
`hf_publish.py` (~870 lines) are the largest modules. Unit tests with fake runtimes
cover their branches. Mutation does not cover them.

**Survivors are allowed only when reviewed.** Until the 2026-09 uplift, the gate was broken
and did not show it. The copied workspace of mutmut did not have the documentation that one
unit test reads. Therefore, its stats run failed and no mutant was tested. The survivor count
was zero.

The workspace now copies the files that the tests read. `make mutation` runs mutmut, ignores
its exit code, then runs `scripts/check_mutants.py`. That script refuses a run that killed
nothing, refuses any mutant status other than killed or survived, and fails on each survivor
whose exact ID is not in `mutation-allowlist.txt`. It also fails on an allowlist entry that no
longer survives. Each allowlist entry carries its equivalence reason. Some surviving mutants are
equivalent, such as redundant `zip(strict=True)` guards and locale-default encodings that
resolve to UTF-8. CI does not permit zero survivors; it permits only the reviewed ones.

**CRAP ceilings and reviewed exceptions (#73).** CI requires 100% domain line and branch
coverage. The CRAP ceiling is 8. Adapters and the CLI have a ceiling of 15.

A function in an adapter or in the CLI that is above the ceiling must have an explicit
reason and a test reference in `scripts/crap-allowlist.json`. The checker reports each
exception. It fails if an entry becomes stale. Therefore, the exceptions cannot grow
without notice.
