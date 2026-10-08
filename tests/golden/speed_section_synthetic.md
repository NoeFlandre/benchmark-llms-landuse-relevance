## Runtime performance

Timings are generation wall seconds summed across language runs; latency and throughput
are recomputed from prediction telemetry. Different devices and runtimes are not directly
comparable. Full per-language measurements are in `leaderboard.csv`.

| model_id | runtime | device | generation_mode | batch_size | language_count | cumulative_wall_seconds | sentences_per_second | latency_mean_seconds | latency_p50_seconds | latency_p95_seconds | generated_tokens | output_tokens_per_second | mean_accept_length | draft_accept_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g/bare | transformers | NVIDIA A100 | static-batched | 16 | 2 | 2.0 | 1.0 |  |  |  |  |  |  |  |
| g/full | sglang | NVIDIA A100 | static-batched | 16 | 2 | 3.0 | 1.0 | 0.2 | 0.2 | 0.29 | 12 | 4.0 | 1.714 | 0.4545 |