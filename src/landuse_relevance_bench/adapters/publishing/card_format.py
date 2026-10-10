"""Formatting helpers shared by the dataset-card sections."""

from collections.abc import Sequence
from typing import Any

from landuse_relevance_bench.domain.records import RunResult


def markdown_table(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| "
        + " | ".join("" if row.get(column) is None else str(row[column]) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def single_setting(results: Sequence[RunResult], field: str) -> Any:
    values = {getattr(result.metadata, field) for result in results}
    return values.pop() if len(values) == 1 else "varies"


def metric_rankings(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    lower_is_better: frozenset[str] = frozenset(),
) -> dict[str, tuple[float, float | None]]:
    """Return the best and next distinct value for each metric column."""
    rankings = {}
    for column in columns:
        values = sorted(
            {float(row[column]) for row in rows if row.get(column) is not None},
            reverse=column not in lower_is_better,
        )
        if values:
            rankings[column] = (values[0], values[1] if len(values) > 1 else None)
    return rankings


def ranked_cell(
    row: dict[str, Any],
    column: str,
    display: str,
    rankings: dict[str, tuple[float, float | None]],
) -> str:
    """Emphasize tied best values and the runner-up without changing the data."""
    if column not in rankings or row.get(column) is None:
        return display
    best, second = rankings[column]
    value = float(row[column])
    if value == best:
        return f"**{display}**"
    if second is not None and value == second:
        return f"<u>{display}</u>"
    return display


def format_card_float(value: float) -> str:
    """Format a score to four useful decimals, omitting redundant zeroes."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_card_threshold(value: float) -> str:
    """Keep small score boundaries readable without losing significant digits."""
    return f"{value:.6g}"


def setting_or_varying(results: Sequence[RunResult], name: str, value_of: Any) -> str:
    """Describe a setting without rejecting a mixed, explicitly recorded roster."""
    values = sorted({value_of(result) for result in results}, key=str)
    if len(values) == 1:
        return str(values[0])
    return f"varies across runs ({name} is recorded per run)"
