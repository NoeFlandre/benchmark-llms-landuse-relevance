# ADR-0007 — Additional runtimes in isolated environments

**Status:** accepted · 2026-09-24

## Context

[ADR-0003](0003-transformers-runtime.md) made Transformers the runtime. Two kinds of
model do not fit it. A GGUF quant can only be run by Transformers after dequantizing
it, which measures a bf16 model with rounding noise rather than quantized inference.
GLiClass and GLiNER2 are documented through their own SDKs, and GLiNER2 pins
Transformers<5, which conflicts with the locked runtime.

## Decision

- GGUF quants run through llama.cpp (`adapters/llama_generator.py`), behind the same
  `TextGenerator` protocol: the GGUF's embedded chat template with thinking disabled,
  greedy decoding, the same token budget, truncation reported. The roster names the
  quant (`repo@quant`); runs record it in a `quantization` field and dtype `gguf`.
- GLiClass and GLiNER2 run through their SDKs behind `LabelScorer`.
- On Grid'5000 each of these, and GTE, gets a per-job environment removed on exit
  (see [Running on Grid'5000](../grid5000.md)). The locked environment is never
  modified.
- Runs also record `device_name`, the accelerator the timing was measured on.
  `quantization` and `device_name` are omitted from the JSON when empty, so every
  full-precision run keeps its exact published bytes.

## Consequences

- A quantized row is distinguishable from its full-precision model and is kept out of
  full-precision comparisons.
- The llama.cpp build needs a CUDA toolkit on the node; a site without one cannot run
  the GGUF path.
- Environments are rebuilt per job, which costs startup time but keeps concurrent jobs
  from uninstalling each other's dependencies.
- The domain is unchanged: each runtime is one more adapter behind an existing seam.
