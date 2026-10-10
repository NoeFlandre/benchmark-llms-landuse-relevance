"""The runtime-performance section, recomputed from prediction telemetry."""

from collections.abc import Sequence
from typing import Any

from landuse_relevance_bench.adapters.publishing.card_format import markdown_table, single_setting
from landuse_relevance_bench.adapters.results_store import group_by_model, rounded
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.speed import summarise_speed

SPEED_COLUMNS = (
    "model_id",
    "runtime",
    "device",
    "generation_mode",
    "batch_size",
    "language_count",
    "cumulative_wall_seconds",
    "sentences_per_second",
    "latency_mean_seconds",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "generated_tokens",
    "output_tokens_per_second",
    "mean_accept_length",
    "draft_accept_rate",
)

EXPECTED_FULL_SWEEP_LANGUAGE_COUNT = 85


def speed_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    rows = []
    for model_id, runs in sorted(group_by_model(results).items()):
        predictions = [prediction for run in runs for prediction in run.predictions]
        speed = summarise_speed(predictions, sum(run.metadata.duration_seconds for run in runs))
        rows.append(
            {
                "model_id": model_id,
                "runtime": single_setting(runs, "runtime"),
                "device": single_setting(runs, "device_name"),
                "generation_mode": single_setting(runs, "generation_mode"),
                "batch_size": single_setting(runs, "batch_size"),
                "language_count": len({run.metadata.language for run in runs}),
                "cumulative_wall_seconds": round(speed.wall_seconds, 2),
                "sentences_per_second": rounded(speed.sentences_per_second, 3),
                "latency_mean_seconds": rounded(speed.latency_mean_seconds, 4),
                "latency_p50_seconds": rounded(speed.latency_p50_seconds, 4),
                "latency_p95_seconds": rounded(speed.latency_p95_seconds, 4),
                "generated_tokens": speed.generated_tokens,
                "output_tokens_per_second": rounded(speed.output_tokens_per_second, 2),
                "mean_accept_length": rounded(speed.mean_accept_length, 3),
                "draft_accept_rate": rounded(speed.draft_accept_rate, 4),
            }
        )
    return rows


def speed_section(results: Sequence[RunResult]) -> str:
    sections = [
        "## Runtime performance",
        """Timings are generation wall seconds summed across language runs; latency and throughput
are recomputed from prediction telemetry. Different devices and runtimes are not directly
comparable. Full per-language measurements are in `leaderboard.csv`.""",
    ]
    reproducibility = _reproducibility_note(results)
    if reproducibility:
        sections.append(reproducibility)
    sections.append(markdown_table(SPEED_COLUMNS, speed_rows(results)))
    return "\n\n".join(sections)


def _reproducibility_note(results: Sequence[RunResult]) -> str:
    run_ids = {
        "LiquidAI/LFM2.5-VL-3B@sglang-throughput-b16",
        "LiquidAI/LFM2.5-VL-3B+DSpark-throughput-b16",
    }
    languages_by_run = {run_id: set() for run_id in run_ids}
    for result in results:
        run_id = result.metadata.run_id
        if run_id in languages_by_run:
            languages_by_run[run_id].add(result.metadata.language)
    if any(
        len(languages) != EXPECTED_FULL_SWEEP_LANGUAGE_COUNT
        for languages in languages_by_run.values()
    ):
        return ""
    return (
        "Reproducibility probes show GPU sensitivity: on 15 overlapping LFM2.5-2.6B "
        "SGLang-throughput languages, A40 versus RTX 6000 Ada changed 474/4,500 "
        "verdicts; LFM2.5-8B-A1B changed 31/300 verdicts across GPU types. VL-3B "
        "SGLang throughput changed 158/25,500 verdicts between RTX A6000 and RTX "
        "6000 Ada, while its same-GPU DSpark comparison changed 0/25,500. Throughput "
        "mode versus batch size 1 changed 2/300 VL-3B English verdicts. Transformers "
        "continuous batching fails on LFM2 with `Invalid group type: conv`. Compare runs "
        "only with the same runtime, mode and GPU model."
    )
