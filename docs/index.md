# Land-use relevance LLM benchmark

Do small open-weight language models know when a sentence about a place describes
something that a satellite can see? The benchmark tests this in 85 languages.

Each language split has the same 300 adjudicated source items. An item has the label
`yes` if it has land-use, land-cover, or geographic-environment signal. Examples are
vegetation, water, terrain, buildings, roads, mining, and managed land. An item has the
label `no` if it is only about history, administration, people, or events.

The benchmark uses one English prompt. The expected output is one token. It tests 18
open-weight models that the project tested before. It runs on Grid'5000 GPUs.

```bash
uv sync --extra inference
uv run lrb models
uv run lrb languages
uv run lrb run LiquidAI/LFM2.5-350M
uv run lrb report
```

- [The benchmark and the prompt](benchmark.md)
- [Run the benchmark on Grid'5000](grid5000.md)
- [Results](results.md)
- [The code architecture](architecture.md)
- [Glossary](glossary.md)

The translation manifest defines the active benchmark. The project keeps archived files
outside active discovery and publication.
