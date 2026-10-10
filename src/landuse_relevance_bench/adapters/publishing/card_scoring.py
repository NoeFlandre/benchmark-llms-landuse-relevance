"""The scoring-models section: setup, prompts, thresholded metrics and encoder behaviour."""

from collections.abc import Sequence
from typing import Any

from landuse_relevance_bench.adapters.hashing import sha256_of_text
from landuse_relevance_bench.adapters.publishing.card_format import (
    format_card_float,
    format_card_threshold,
    metric_rankings,
    ranked_cell,
    setting_or_varying,
)
from landuse_relevance_bench.adapters.results_store import group_by_model, scoring_summary_rows
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.scorers import DEFAULT_CARD, scorer_for

SCORING_SUMMARY_CARD_COLUMNS = (
    ("model_id", "model", None),
    ("language_count", "languages", None),
    ("best_mcc", "MCC @ threshold", "best_mcc_threshold"),
    ("best_f1", "F1 @ threshold", "best_f1_threshold"),
    (
        "best_balanced_accuracy",
        "balanced accuracy @ threshold",
        "best_balanced_accuracy_threshold",
    ),
    ("best_precision", "precision @ threshold", "best_precision_threshold"),
    ("best_recall", "recall @ threshold", "best_recall_threshold"),
    ("roc_auc_macro", "ROC-AUC", None),
    ("throughput_items_per_second_macro", "items/s", None),
    ("peak_vram_bytes_max", "peak VRAM (GiB)", None),
)


def scoring_section(
    scoring: Sequence[RunResult],
    *,
    prompts: Sequence[tuple[str, str]],
) -> str:
    """Render compact public-facing scoring details and tables."""
    summary = scoring_summary_rows(scoring)
    metric_columns = tuple(
        value_column
        for value_column, _, _ in SCORING_SUMMARY_CARD_COLUMNS
        if value_column not in {"model_id", "language_count"}
    )
    rankings = metric_rankings(
        summary, metric_columns, lower_is_better=frozenset({"peak_vram_bytes_max"})
    )
    summary_header = "| " + " | ".join(label for _, label, _ in SCORING_SUMMARY_CARD_COLUMNS) + " |"
    summary_divider = "|" + "|".join(["---"] * len(SCORING_SUMMARY_CARD_COLUMNS)) + "|"
    summary_body = "\n".join(
        "| "
        + " | ".join(
            ranked_cell(
                row,
                value_column,
                _card_summary_value(row, value_column, threshold_column),
                rankings,
            )
            for value_column, _, threshold_column in SCORING_SUMMARY_CARD_COLUMNS
        )
        + " |"
        for row in summary
    )
    setup_header = (
        "| model | handling | relevance score / decision rule | sequence length (tokens) | "
        "dtype / batch / seed | revision |"
    )
    setup_divider = "|" + "|".join(["---"] * 6) + "|"
    by_model = group_by_model(scoring)
    setup_body = "\n".join(_scoring_setup_row(by_model[model_id]) for model_id in sorted(by_model))
    prompt_blocks = "\n\n".join(
        f"{label}:\n\n```text\n{text}```" if text else f"{label}." for label, text in prompts
    )
    behavior_note = _encoder_output_behavior_note(scoring)
    behavior_block = f"\n\n{behavior_note}" if behavior_note else ""
    return f"""## Scoring models

Scores are normalized to [0, 1]. Best thresholds are selected on this benchmark (an upper
bound); ROC-AUC needs no threshold. Full sweep: [`threshold_sweep.csv`](threshold_sweep.csv).

### Scoring setup

{setup_header}
{setup_divider}
{setup_body}

### Scoring prompts

{prompt_blocks}

### Best thresholded scoring metrics

{summary_header}
{summary_divider}
{summary_body}{behavior_block}"""


def _encoder_output_behavior_note(scoring: Sequence[RunResult]) -> str:
    """Summarize the encoder's observed yes rate without assuming its behavior."""
    encoder_id = "LiquidAI/LFM2.5-Encoder-350M"
    runs = [result for result in scoring if result.metadata.model_id == encoder_id]
    if not runs:
        return ""

    yes_counts = [
        sum(prediction.predicted is Label.YES for prediction in result.predictions)
        for result in runs
    ]
    item_counts = [len(result.predictions) for result in runs]
    yes_range = (
        str(yes_counts[0])
        if min(yes_counts) == max(yes_counts)
        else f"{min(yes_counts)} to {max(yes_counts)}"
    )
    languages = {result.metadata.language for result in runs}
    if len(languages) == len(runs) and len(set(item_counts)) == 1:
        language_label = "language" if len(languages) == 1 else "languages"
        return (
            "**Observed output behavior:** In these runs, LFM2.5-Encoder-350M predicted "
            f"`yes` for {yes_range} of {item_counts[0]} items per language across "
            f"{len(languages)} {language_label}."
        )

    item_range = (
        str(item_counts[0])
        if min(item_counts) == max(item_counts)
        else f"{min(item_counts)} to {max(item_counts)}"
    )
    return (
        "**Observed output behavior:** In these runs, LFM2.5-Encoder-350M predicted "
        f"`yes` for {yes_range} of {item_range} items per run across {len(runs)} runs "
        f"in {len(languages)} languages."
    )


def scoring_prompt_blocks(
    scoring: Sequence[RunResult], llm_prompt: str, candidates: Sequence[str]
) -> list[tuple[str, str]]:
    """Match every scoring run's prompt digest to a supplied text, labelled by its users.

    A scorer that reads the LLM prompt (log-probability scoring) is pointed back at the
    task prompt above instead of repeating it.
    """
    llm_digest = sha256_of_text(llm_prompt)
    texts = {sha256_of_text(text): text for text in candidates if text}
    users: dict[str, list[str]] = {}
    for result in scoring:
        users.setdefault(result.metadata.prompt_sha256, [])
        if result.metadata.model_id not in users[result.metadata.prompt_sha256]:
            users[result.metadata.prompt_sha256].append(result.metadata.model_id)
    blocks: list[tuple[str, str]] = []
    for digest, models in users.items():
        if digest == llm_digest:
            continue
        if digest not in texts:
            raise ValueError(
                "scoring prompt text does not match the digest recorded in the scoring runs "
                f"of {', '.join(models)}"
            )
        blocks.append((", ".join(f"`{model}`" for model in sorted(models)), texts[digest]))
    if llm_digest in users:
        names = sorted(users[llm_digest])
        verb = "uses" if len(names) == 1 else "use"
        models = ", ".join(f"`{model}`" for model in names)
        blocks.append((f"{models} {verb} the task prompt above", ""))
    return blocks


def _scoring_setup_row(results: Sequence[RunResult]) -> str:
    """Render the reproducibility-relevant adapter settings for one model."""
    model_id = results[0].metadata.model_id
    try:
        spec = scorer_for(model_id)
    except KeyError:
        spec = None
    family, input_handling, score = spec.card if spec else DEFAULT_CARD
    rule = spec.decision_rule if spec else results[0].metadata.decision_rule or "recorded per run"
    if rule.endswith(" at 0.5"):
        rule = "yes if score ≥ 0.5"
    sequence = setting_or_varying(results, "sequence length", lambda r: r.metadata.sequence_length)
    if sequence == "None":
        sequence = "model-defined"
    dtype = setting_or_varying(results, "dtype", lambda r: r.metadata.dtype)
    batch = setting_or_varying(results, "batch size", lambda r: r.metadata.batch_size)
    seed = setting_or_varying(results, "seed", lambda r: r.metadata.seed)
    revision = setting_or_varying(results, "revision", lambda r: r.metadata.model_revision)
    cells = (
        model_id,
        f"{family}: {input_handling}",
        f"{score}; {rule}",
        sequence,
        f"{dtype}; batch {batch}; seed {seed}",
        revision,
    )
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _card_summary_value(
    row: dict[str, Any], value_column: str, threshold_column: str | None
) -> str:
    """Format compact card values without changing the downloadable CSV schema."""
    value = row.get(value_column)
    if value is None:
        return "n/a"
    if threshold_column is not None:
        threshold = row[threshold_column]
        return f"{format_card_float(float(value))} @ {format_card_threshold(float(threshold))}"
    if value_column == "peak_vram_bytes_max":
        return f"{float(value) / 1024**3:.2f}"
    if value_column == "throughput_items_per_second_macro":
        return f"{float(value):.2f}"
    return str(value) if isinstance(value, str | int) else format_card_float(float(value))
