# Land-use relevance LLM benchmark

Do small open-weight language models know when a sentence about a place says
something a satellite could see?

Each of 154 adjudicated sentences is labelled `yes` if it carries land-use,
land-cover, or geographic-environment signal — vegetation, water, terrain,
buildings, roads, mining, managed land — and `no` if it only concerns history,
administration, people, or events. One prompt, one token of expected output, four
models, scored end to end on a Grid'5000 GPU.

```bash
uv sync --extra inference
uv run lrb models
uv run lrb run LiquidAI/LFM2.5-350M
uv run lrb report
```

- [The benchmark and the prompt](benchmark.md)
- [Running it on Grid'5000](grid5000.md)
- [Results](results.md)
- [How the code is arranged](architecture.md)
