# Architecture

Three layers, enforced by `import-linter` and by a test that fails the build if the
rule is broken:

```
cli  →  adapters  →  domain
```

## `domain` — pure

No filesystem, no network, no model runtime; `json`, `csv`, `pathlib` and `torch` are
all forbidden imports here. Holds the label type, prompt rendering, verdict parsing,
the metrics, the speed summary, the speculative-decoding agreement check, the run
records, the roster, and `predict_all`, which drives a benchmark across anything
satisfying the one-method `TextGenerator` protocol and times each call to it.

That protocol is the single seam to a model runtime. Every test above the unit level
substitutes a scripted generator through it, which is why the acceptance suite runs
end to end in seconds without a GPU.

## `adapters` — the edges

Benchmark CSV loading, prompt loading, content digests, the results store, the
Transformers generators (text and vision-language), the SGLang generator (plain or
with a DSpark draft), the runtime dispatch in `generators`, and Hub publishing. `pipeline.execute` is the one use case:
load inputs, run the model, score exactly the predictions it produced, write the file.

## `cli` — the surface

`lrb models | run | run-all | report | publish`. Optional dependencies are imported
lazily, so `lrb report` works on a laptop with no `torch` installed.

## Quality gates

`make check` runs the gauntlet: ruff → ty → unit → property → acceptance →
architecture → CRAP → mutation. CI runs the same commands.
