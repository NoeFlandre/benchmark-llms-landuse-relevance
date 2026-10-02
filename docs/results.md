# Results

The published runs belong to the `v3-multilingual` benchmark set in
[NoeFlandre/benchmark-llms-landuse-relevance](https://huggingface.co/datasets/NoeFlandre/benchmark-llms-landuse-relevance).
Each model-language checkpoint is explicit. The public release uses a generation budget
of 4096 tokens. Therefore, the benchmark measures classification. It does not measure
compliance with the output length.

## Read a run file

A run file holds the metadata that you need to audit it:

- the language
- the model revision
- the prompt and benchmark sha256
- the decoding settings
- the seed
- the host duration
- the throughput
- the peak CUDA VRAM, when available
- the source commit

Then it holds every prediction with its raw generation or its relevance scores. At the
end, it holds the metrics. The code computes the metrics from exactly these predictions.

Each prediction has the flag `truncated`. The benchmark scores a truncated generation as
unparsed, for all the verdict words in it. The model used its complete budget in the
middle of its reasoning and did not commit to an answer.

Therefore, `unparsed_rate` covers two cases: "never answered" and "was cut off". The flag
shows which case applies.

## Reproduce a number

```bash
uv run lrb run LiquidAI/LFM2.5-2.6B --revision <revision from the run file>
uv run lrb report
```

Decoding is greedy and a digest pins every input. But the output can be different on
different GPUs. Compare runs only when the runtime, the mode, and the GPU model are the
same.

Cross-GPU probes found these changes in verdicts:

- 474/4,500 for 15 overlapping 2.6B SGLang-throughput languages
- 31/300 for 8B-A1B
- 158/25,500 for the VL-3B SGLang-throughput run

The VL-3B SGLang/DSpark pair on the same GPU matched on all items.

LFM2.5-Encoder-350M predicted `yes` on 298-300 of 300 items in each of its 15
supported-language runs. This is observed model behavior. Refer to
[ADR-0011](adr/0011-sglang-throughput-mode.md) for the hardware and mode details and for
the LFM2 continuous-batching limitation.

The dataset card is a deterministic model-level aggregate table. It has one row for each
model. `language_count` shows how many language checkpoints gave data to the row. The
command `lrb publish` finds the result JSON files recursively. It recomputes every
aggregate from their predictions.

The card also shows what the benchmark measured: the prompt (verbatim), the label set,
the token budget, the decoding, the dtype, the batch size, and the seed. The option
`lrb publish --prompt` supplies the prompt. The card does not build if the prompt hash
differs from the digest that the runs recorded. Therefore, the published prompt is
always the prompt that gave the scores.

To publish one language slice, use `lrb publish --language fr`. Then the generated card,
the leaderboards, the Dataset Viewer rows, and the uploaded run files include only that
language. The source result JSON files of the other languages stay in place locally.

For scoring models, `threshold_sweep.csv` recomputes the classification metrics from the
stored `yes` relevance scores. It does not call a model again. `scoring_summary.csv`
gives the best threshold for each requested classification metric. It also has ROC-AUC,
throughput, and the maximum observed allocated VRAM.

The Hub release includes `data/train.csv` as the default Dataset Viewer split. It is a
validated export, by row, of the languages in the published runs. It has the stable
fields `item_id` and `source_item_id`. The detailed model checkpoints stay as separate
JSON files. A publish that has a language filter exports only the rows and the run files
of those languages.

## Artifact size and sweep planning

The four-model LFM2.5 snapshot of 2026-09-23 has 340 model-language JSON files and
102,000 predictions. It is acceptable to keep the complete `raw_output` in the release.
The files total 165,426,650 bytes (0.154 GiB). The largest file is 1.52 MiB. The raw
generations stay available for parser audits and for alternative scoring. You do not
need another inference run.

The recorded walltime for that 85-language snapshot was:

| Model | GPU-hours | Median language | 95th percentile | Longest language |
|---|---:|---:|---:|---:|
| LFM2.5-350M | 0.098 | 2.5 s | 13.1 s | 17.2 s |
| LFM2.5-1.2B-Instruct | 0.214 | 3.3 s | 28.3 s | 51.2 s |
| LFM2.5-2.6B | 40.114 | 1,002.2 s | 5,209.5 s | 6,339.2 s |
| LFM2.5-8B-A1B | 32.863 | 1,535.8 s | 2,645.6 s | 3,540.4 s |

The snapshot totals 73.289 measured GPU-hours. Its run metadata does not record the GPU
model. Therefore, these durations are evidence of the workload. They are not portable
estimates for a reservation.

For a new site, do these steps:

1. Measure one full 300-item language on the target GPU.
2. Request a reservation for each model-language shard. Use the observed 95th percentile,
   plus the model load time, plus a margin of 20%.

A completed language file is the checkpoint. A job that stops early loses at most its
current language.
