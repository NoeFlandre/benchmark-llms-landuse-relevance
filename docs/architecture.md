# Architecture

Three layers, enforced by `import-linter` and by a test that fails the build if the
rule is broken:

```
cli  →  adapters  →  domain
```

## `domain` — pure

No filesystem, no network, no model runtime; `json`, `csv`, `pathlib` and `torch` are
all forbidden imports here. Holds the label type, prompt rendering, verdict parsing,
the metrics, the run records, the roster, and `predict_all`, which drives a benchmark
across anything satisfying the one-method `TextGenerator` protocol.

That protocol is the single seam to a model runtime. Every test above the unit level
substitutes a scripted generator through it, which is why the acceptance suite runs
end to end in seconds without a GPU.

## `adapters` — the edges

Translation-manifest and CSV loading, prompt loading, content digests, the results
store, the Transformers generator and scorer adapters, and Hub publishing.
`pipeline.execute` and `pipeline.execute_scoring` are the use cases: load one language,
run the model, score exactly the predictions it produced, and write the nested
model-language checkpoint. Active readers explicitly exclude `results/archive/`.

## `cli` — the surface

`lrb models | languages | run | run-all | scorers | score | report | publish`.
Optional dependencies are imported lazily, so `lrb report` works on a laptop with no
`torch` installed. Scoring adapters expose a common typed input and preserve both
normalised and native relevance values; the results adapter derives threshold sweeps
and best-operating-point summaries from those stored values.

## Quality gates

`make check` runs the gauntlet: ruff → ty → unit → property → acceptance →
architecture → CRAP → mutation. CI runs the same commands.
