# Results

Published runs belong to the `v3-multilingual` benchmark set in
[NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance).
Each model-language checkpoint is explicit, and the public release uses a 4096-token
generation budget so the benchmark measures classification rather than output-length
compliance.

## Reading a run file

Each run file holds the metadata needed to audit it — language, model revision, prompt
and benchmark sha256, decoding settings, seed, host duration, throughput, peak CUDA
VRAM when available, and the source commit — then every prediction with its raw
generation or relevance scores, and finally the metrics computed from exactly those
predictions.

Each prediction carries `truncated`. A truncated generation is scored unparsed however
many verdict words appear in it: the model ran out of budget mid-thought and never
committed. `unparsed_rate` therefore covers both "never answered" and "was cut off",
and the flag says which.

## Reproducing a number

```bash
uv run lrb run LiquidAI/LFM2.5-2.6B --revision <revision from the run file>
uv run lrb report
```

Decoding is greedy and every input is pinned by digest, but cross-GPU output identity is
not guaranteed. Compare runs only when runtime, mode and GPU model match. Cross-GPU
probes found 474/4,500 verdict changes across 15 overlapping 2.6B SGLang-throughput
languages, 31/300 for 8B-A1B, and 158/25,500 for the VL-3B SGLang-throughput run; the
same-GPU VL-3B SGLang/DSpark pair matched all items. LFM2.5-Encoder-350M predicted
`yes` on 298–300 of 300 items in each of its 15 supported-language runs; this is observed
model behavior. See [ADR-0011](adr/0011-sglang-throughput-mode.md) for hardware/mode
details and the LFM2 continuous-batching limitation.

The dataset card is a deterministic model-level aggregate table. It includes one row
per model, with `language_count` showing how many language checkpoints contributed;
`lrb publish` discovers result JSONs recursively and recomputes every aggregate from
their predictions.

The card also states what was measured: the prompt verbatim, the label set, the token
budget, decoding, dtype, batch size and seed. `lrb publish --prompt` supplies the
prompt, and the card refuses to build unless it hashes to the digest the runs recorded,
so the published prompt is always the one the scores came from.

Use `lrb publish --language fr` to publish one language slice. The generated card,
leaderboards, Dataset Viewer rows, and uploaded run files are restricted to that
language; the source result JSONs for other languages remain in place locally.

For scoring models, `threshold_sweep.csv` recomputes classification metrics from the
stored `yes` relevance scores without another model call. `scoring_summary.csv` gives
the best threshold for each requested classification metric and includes ROC-AUC,
throughput, and maximum observed allocated VRAM.

The Hub release includes `data/train.csv` as the default Dataset Viewer split. It is a
validated, row-oriented export of the languages represented by the published runs,
with stable `item_id` and `source_item_id` fields; detailed model checkpoints remain
separate JSON files. A language-filtered publish exports only those language rows and
run files.

## Artifact size and sweep planning

The 2026-09-23 four-model LFM2.5 snapshot contains 340 model-language JSON files and
102,000 predictions. Keeping complete `raw_output` is an acceptable release cost:
the files total 165,426,650 bytes (0.154 GiB), and the largest file is 1.52 MiB.
Raw generations stay available for parser audits and alternative scoring without
another inference run.

Recorded walltime for that 85-language snapshot was:

| Model | GPU-hours | Median language | 95th percentile | Longest language |
|---|---:|---:|---:|---:|
| LFM2.5-350M | 0.098 | 2.5 s | 13.1 s | 17.2 s |
| LFM2.5-1.2B-Instruct | 0.214 | 3.3 s | 28.3 s | 51.2 s |
| LFM2.5-2.6B | 40.114 | 1,002.2 s | 5,209.5 s | 6,339.2 s |
| LFM2.5-8B-A1B | 32.863 | 1,535.8 s | 2,645.6 s | 3,540.4 s |

The snapshot totals 73.289 measured GPU-hours. Its run metadata does not record the GPU
model, so these durations are workload evidence, not portable reservation estimates.
For a new site, measure one full 300-item language on the target GPU, then request a
per-model-language shard reservation using the observed 95th percentile plus model
load time and a 20% margin. A completed language file is the checkpoint; an
interrupted job should lose at most its current language.
