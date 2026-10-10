"""Log-probability scoring compared with parsed generation, with same-GPU timing when available."""

from collections.abc import Sequence

from landuse_relevance_bench.adapters.publishing.card_format import format_card_float
from landuse_relevance_bench.adapters.results_store import (
    aggregate_rows,
    group_by_model,
    scoring_summary_rows,
)
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.scorers import logprob_pairs


def logprob_comparison(
    results: Sequence[RunResult], timing_results: Sequence[RunResult] = ()
) -> str:
    """Compare log-probability scoring against parsing the same model's generations.

    ``timing_results`` are generative reruns on the log-prob runs' GPU, kept out of the
    leaderboards; they add a same-hardware time row over their languages.
    """
    by_model = group_by_model(results)
    timing = group_by_model(timing_results)
    sections = [
        _logprob_section(scored_id, generated_id, by_model, timing)
        for scored_id, generated_id in logprob_pairs().items()
    ]
    return "\n\n".join(section for section in sections if section)


def _logprob_section(
    scored_id: str,
    generated_id: str,
    by_model: dict[str, list[RunResult]],
    timing: dict[str, list[RunResult]],
) -> str:
    scored, generated = by_model.get(scored_id), by_model.get(generated_id)
    if not scored or not generated:
        return ""
    shared = {r.metadata.language for r in scored} & {r.metadata.language for r in generated}
    rows = [
        _comparison_row(method, [r for r in runs if r.metadata.language in shared], scored_id)
        for method, runs in (("generation + parsing", generated), ("yes/no log-probs", scored))
    ]
    same_gpu = _same_gpu_row(scored, timing.get(generated_id, []))
    if same_gpu:
        rows.append(same_gpu)
    name = generated_id.rsplit("/", 1)[-1]
    return (
        f"### {name}: log-probabilities vs generation\n\n"
        "Same model, prompt and languages; log-probs call yes when P(yes) > P(no).\n\n"
        "| method | F1 | MCC | unparsed | ROC-AUC | GPU hours | ms/item |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows)
    )


def _comparison_row(method: str, subset: Sequence[RunResult], scored_id: str) -> str:
    (aggregate,) = aggregate_rows(subset)
    seconds = sum(r.metadata.duration_seconds for r in subset)
    items = sum(r.metrics.n_items for r in subset)
    auc = "n/a"
    if not subset[0].metadata.is_generative:
        (summary,) = (row for row in scoring_summary_rows(subset) if row["model_id"] == scored_id)
        value = summary["roc_auc_macro"]
        auc = "n/a" if value is None else format_card_float(float(value))
    return (
        f"| {method} | {aggregate['f1_macro']} | {aggregate['matthews_corrcoef_macro']} "
        f"| {aggregate['unparsed_rate_macro']} | {auc} | {seconds / 3600:.2f} "
        f"| {1000 * seconds / items:.1f} |"
    )


def _same_gpu_row(scored: Sequence[RunResult], reruns: Sequence[RunResult]) -> str:
    """Time generation and log-probs over the same languages on the same GPU model."""
    if not reruns:
        return ""
    languages = {r.metadata.language for r in reruns}
    devices = {r.metadata.device_name for r in reruns}
    matched = [
        r for r in scored if r.metadata.language in languages and r.metadata.device_name in devices
    ]
    if len(devices) != 1 or len(matched) != len(reruns):
        return ""
    label = f"same GPU ({devices.pop()}; {', '.join(sorted(languages))})"
    generated_s = sum(r.metadata.duration_seconds for r in reruns)
    scored_s = sum(r.metadata.duration_seconds for r in matched)
    return (
        f"| {label}: generation vs log-probs | | | | | {generated_s:.0f} s vs "
        f"{scored_s:.1f} s ({generated_s / scored_s:.0f}x) | |"
    )
