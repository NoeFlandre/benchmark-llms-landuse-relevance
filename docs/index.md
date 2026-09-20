# Land-use relevance LLM benchmark

Do small open-weight language models know when a sentence about a place says
something a satellite could see, across 85 languages?

Each language split contains the same 300 adjudicated source items. A row is labelled
`yes` if it carries land-use, land-cover, or geographic-environment signal — vegetation,
water, terrain, buildings, roads, mining, managed land — and `no` if it only concerns
history, administration, people, or events. One English prompt, one token of expected
output, four models, scored end to end on a Grid'5000 GPU.

```bash
uv sync --extra inference
uv run lrb models
uv run lrb languages
uv run lrb run LiquidAI/LFM2.5-350M
uv run lrb report
```

- [The benchmark and the prompt](benchmark.md)
- [Running it on Grid'5000](grid5000.md)
- [Results](results.md)
- [How the code is arranged](architecture.md)

The active benchmark is defined by the translation manifest; archived files are kept
outside active discovery and publication.
