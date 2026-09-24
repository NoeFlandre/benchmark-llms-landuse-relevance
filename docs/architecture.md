# Architecture

Three layers, enforced by `import-linter` and by a test that fails the build if the
rule is broken:

```
cli  →  adapters  →  domain
```

## `domain` — pure

No filesystem, no network, no model runtime; `csv`, `pathlib`, `typer`, `torch`,
`transformers` and `huggingface_hub` are forbidden imports here. Holds the label type,
prompt rendering, verdict parsing, the metrics and threshold sweeps, the run records,
the generative and scoring rosters, and the orchestration that drives a benchmark
across anything satisfying one of two protocols: `TextGenerator` (prompts in, text
out) or `LabelScorer` (prompt/sentence pairs in, yes/no scores out).

Those two protocols are the only seams to a model runtime. Every test above the unit
level substitutes a scripted generator or scorer through them, which is why the
acceptance suite runs end to end in seconds without a GPU.

## `adapters` — the edges

Translation-manifest and CSV loading, prompt loading, content digests, the results
store, and Hub publishing. Generators: Transformers, and llama.cpp for GGUF quants
(`llama_generator.py`). Scorers (`hf_scorer.py`): Qwen reranker, GTE, mxbai, Laya,
causal log-probability, the NLI zero-shot pipeline, GLiClass and GLiNER2 — see
[ADR-0007](adr/0007-additional-runtimes.md).
`pipeline.execute` and `pipeline.execute_scoring` are the use cases: load one language,
run the model, score exactly the predictions it produced, and write the nested
model-language checkpoint. Active readers explicitly exclude `results/archive/`.

## `cli` — the surface

`lrb models | languages | run | run-all | scorers | score | status | report | publish`.
Optional dependencies are imported lazily, so `lrb report` works on a laptop with no
`torch` installed. Scoring adapters expose a common typed input and preserve both
normalised and native relevance values; the results adapter derives threshold sweeps
and best-operating-point summaries from those stored values.

## Quality gates

`make check` runs the gauntlet: ruff → ty → unit → property → acceptance →
architecture → CRAP. Mutation testing is separate (`make mutation`); CI runs the same
commands plus mutation, gated by `scripts/check_mutants.py --max-survivors 4`.
