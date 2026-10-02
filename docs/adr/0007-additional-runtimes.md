# ADR-0007 - Additional runtimes in isolated environments

**Status:** accepted · 2026-09-24

## Context

[ADR-0003](0003-transformers-runtime.md) selected Transformers as the runtime. Two kinds
of model do not fit it.

Transformers can run a GGUF quant only after it dequantizes the quant. Then the
benchmark measures a bf16 model with rounding noise. It does not measure quantized
inference.

The SDKs of GLiClass and GLiNER2 are the documented interfaces. GLiNER2 pins
Transformers<5. This conflicts with the locked runtime.

## Decision

- GGUF quants run through llama.cpp (`adapters/llama_generator.py`). They are behind the
  same `TextGenerator` protocol. The setup is:
    - the chat template embedded in the GGUF, with thinking disabled
    - greedy decoding
    - the same token budget
    - truncation is reported

  The roster names the quant (`repo@quant`). Runs record it in a `quantization` field
  and the dtype `gguf`.
- GLiClass and GLiNER2 run through their SDKs behind `LabelScorer`.
- On Grid'5000, each of these models and GTE gets an environment for each job. The job
  removes it on exit. Refer to [Run on Grid'5000](../grid5000.md). The job never changes
  the locked environment.
- Runs also record `device_name`. This is the accelerator on which the run measured the
  timing. The JSON omits `quantization` and `device_name` when they are empty. Therefore,
  each full-precision run keeps its exact published bytes.

## Consequences

- A quantized row is distinguishable from its full-precision model. The benchmark keeps
  it out of full-precision comparisons.
- The llama.cpp build needs a CUDA toolkit on the node. A site that does not have one
  cannot run the GGUF path.
- Each job rebuilds its environments. This costs startup time. But concurrent jobs do
  not uninstall the dependencies of each other.
- The domain does not change. Each runtime is one more adapter behind an existing seam.
