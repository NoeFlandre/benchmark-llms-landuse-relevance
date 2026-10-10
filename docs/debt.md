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

**Mutation testing covers only the domain.** Tests cover the adapters, but the mutation
step does not mutate them. Mutants of filesystem code and model-runtime code are mostly
equivalent mutants.

The adapters are not thin now. `hf_scorer.py` (~730 lines, eight scorer families) is the
largest module. The dataset card is built in `adapters/publishing/`, one module per section,
and `hf_publish.py` keeps the Hub upload. Unit tests with fake runtimes cover their
branches. Mutation does not cover them.

**No mutant survives, and no mutant is permitted to survive.** Until the 2026-09 uplift,
the gate was broken and did not show it. The copied workspace of mutmut did not have the
documentation that one unit test reads. Therefore, its stats run failed and no mutant was
tested. The survivor count was zero.

The workspace now copies the files that the tests read. `check_mutants.py` refuses a run
that killed nothing. The domain does not use constructs that make equivalent mutants. Two
examples are guarded `zip(strict=True)` calls and redundant defaults. CI permits zero
survivors (`scripts/check_mutants.py --max-survivors 0`).

**CRAP ceilings and reviewed exceptions (#73).** CI requires 100% domain line and branch
coverage. The CRAP ceiling is 8. Adapters, the CLI and the scripts in `scripts/` have a
ceiling of 15.

A function in an adapter, the CLI or `scripts/` that is above the ceiling must have an explicit
reason and a test reference in `scripts/crap-allowlist.json`. The checker reports each
exception. It fails if an entry becomes stale. Therefore, the exceptions cannot grow
without notice.
