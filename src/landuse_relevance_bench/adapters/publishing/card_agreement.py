"""The DSpark speculative-decoding section: speed and lossless agreement with the baseline."""

from collections.abc import Sequence

from landuse_relevance_bench.adapters.publishing.card_format import markdown_table
from landuse_relevance_bench.adapters.publishing.card_speed import speed_rows
from landuse_relevance_bench.adapters.results_store import group_by_model
from landuse_relevance_bench.domain.agreement import (
    Agreement,
    is_lossless_pair,
    speculative_agreements,
)
from landuse_relevance_bench.domain.records import RunResult

DSPARK_COLUMNS = (
    "target_model",
    "sglang_baseline",
    "dspark_run",
    "baseline_output_tokens_per_second",
    "dspark_output_tokens_per_second",
    "speedup",
    "identical_predictions",
    "languages_compared",
    "items_compared",
    "verdict_differences",
    "text_differences",
)


def agreement_section(results: Sequence[RunResult]) -> str:
    by_name = group_by_model(results)
    agreements: dict[tuple[str, str], list[Agreement]] = {}
    for agreement in speculative_agreements(results):
        if agreement.same_runtime:
            agreements.setdefault((agreement.speculative_run, agreement.baseline_run), []).append(
                agreement
            )

    speed_by_model = {row["model_id"]: row for row in speed_rows(results)}
    rows = []
    for (draft_name, baseline_name), values in sorted(agreements.items()):
        draft_runs = by_name[draft_name]
        baseline_runs = by_name[baseline_name]
        baseline_tps = speed_by_model[baseline_name]["output_tokens_per_second"]
        draft_tps = speed_by_model[draft_name]["output_tokens_per_second"]
        rows.append(
            {
                "target_model": draft_runs[0].metadata.model_id,
                "sglang_baseline": baseline_name,
                "dspark_run": draft_name,
                "baseline_output_tokens_per_second": baseline_tps,
                "dspark_output_tokens_per_second": draft_tps,
                "speedup": (
                    None
                    if not baseline_tps or draft_tps is None
                    else f"{draft_tps / baseline_tps:.2f}x"
                ),
                "identical_predictions": (
                    "yes" if is_lossless_pair(draft_runs, baseline_runs, values) else "no"
                ),
                "languages_compared": len({agreement.language for agreement in values}),
                "items_compared": sum(value.n_compared for value in values),
                "verdict_differences": sum(value.verdicts_differ for value in values),
                "text_differences": sum(value.texts_differ for value in values),
            }
        )
    if not rows:
        return ""
    return f"""## DSpark speculative decoding

Greedy DSpark runs should match the same-target SGLang baseline across every language;
the check requires complete item coverage and identical generated text.

{markdown_table(DSPARK_COLUMNS, rows)}"""
