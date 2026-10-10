# Architecture

The code has four layers. `import-linter` enforces them. A test fails the build if
someone breaks the rule.

```
cli  →  application  →  adapters  →  domain
```

## `domain` - pure code

The `domain` layer has no filesystem, no network, and no model runtime. These imports
are not permitted in this layer: `csv`, `pathlib`, `typer`, `torch`, `transformers`, and
`huggingface_hub`.

The layer holds these items:

- the label type
- prompt rendering
- verdict parsing
- the metrics and the threshold sweeps
- the run records
- the generative and scoring rosters
- the orchestration that drives a benchmark

The orchestration works with anything that satisfies one of two protocols. The first is
`TextGenerator`: prompts in, text out. The second is `LabelScorer`: prompt/sentence
pairs in, yes/no scores out.

These two protocols are the only seams to a model runtime. Every test above the unit
level replaces the generator or scorer with a scripted one through these protocols. For
this reason, the acceptance suite runs end to end in seconds without a GPU.

## `adapters` - the edges

The `adapters` layer holds these items:

- the loading of the translation manifest and the CSV files
- prompt loading
- content digests
- the results store
- Hub publishing

The generators are Transformers and llama.cpp for GGUF quants (`llama_generator.py`).
The scorers are in `hf_scorer.py`: Qwen reranker, GTE, mxbai, Laya, causal
log-probability, the NLI zero-shot pipeline, GLiClass, and GLiNER2. Refer to
[ADR-0007](adr/0007-additional-runtimes.md).

`pipeline.execute` and `pipeline.execute_scoring` are the use cases. Each one loads one
language and runs the model. Then it scores exactly the predictions that the model
produced and writes the nested model-language checkpoint. The active readers exclude
`results/archive/`.

## `cli` - the surface

The commands are `lrb models | languages | run | run-all | scorers | score | status |
report | publish`.

The code imports optional dependencies when it needs them. Therefore, `lrb report` works
on a laptop that has no `torch`.

The scoring adapters give a common typed input. They keep both the normalised relevance
value and the native relevance value. The results adapter derives the threshold sweeps
and the best-operating-point summaries from these stored values.

## `application` - the command use cases

The `application` layer holds the language selection, the run selection, the sharded run
plans, the checkpoint-aware runs, and the report lines. It changes bad input into usage
errors. Therefore, `cli` only declares commands and options.

The model providers are in `adapters/providers.py`. They include the cache that holds
one model at a time. The cache logs the free GPU memory before each load and after each
close.

## Quality gates

The command `make check` runs these checks in this order: ruff → ty → unit → property →
acceptance → architecture → CRAP.

Mutation testing is separate (`make mutation`). CI runs the same commands and also runs
mutation. `make mutation` runs mutmut, ignores its exit code, then runs
`scripts/check_mutants.py`. That script reads the stored mutmut results and fails on any
unreviewed survivor, any stale entry in `mutation-allowlist.txt`, any unresolved mutant
status, a run that killed nothing, or a mutmut results read that fails.
