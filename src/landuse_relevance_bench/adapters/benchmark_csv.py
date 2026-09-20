"""Loading the labelled benchmark from its CSV file."""

import csv
from collections.abc import Sequence
from pathlib import Path

from landuse_relevance_bench.domain.dataset import BenchmarkItem, InvalidRowError, build_item

REQUIRED_COLUMNS = ("sentence", "label", "source_item_id", "language")
HEADER_LINE = 1


class BenchmarkFileError(ValueError):
    """Raised when a benchmark CSV cannot be loaded as a set of items."""


def load_benchmark(
    path: Path,
    *,
    expected_language: str | None = None,
    expected_source_item_ids: Sequence[str] | None = None,
) -> tuple[BenchmarkItem, ...]:
    """Read every row of ``path`` into a validated, uniquely identified item."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _check_columns(path, reader.fieldnames)
        items = tuple(
            _build(path, row, line, expected_language) for line, row in enumerate(reader, start=2)
        )
    if not items:
        raise BenchmarkFileError(f"benchmark {path} has no rows")
    _check_unique(path, items)
    if expected_source_item_ids is not None:
        _check_source_item_ids(path, items, expected_source_item_ids)
    return items


def _check_columns(path: Path, fieldnames: Sequence[str] | None) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in (fieldnames or ())]
    if missing:
        raise BenchmarkFileError(f"benchmark {path} is missing the column(s) {missing}")


def _build(
    path: Path,
    row: dict[str, str],
    line: int,
    expected_language: str | None,
) -> BenchmarkItem:
    try:
        item = build_item(row)
    except InvalidRowError as exc:
        raise BenchmarkFileError(f"benchmark {path}, line {line}: {exc}") from exc
    if expected_language is not None and item.language != expected_language:
        raise BenchmarkFileError(
            f"benchmark {path}, line {line}: expected language {expected_language!r}, "
            f"got {item.language!r}"
        )
    return item


def _check_unique(path: Path, items: tuple[BenchmarkItem, ...]) -> None:
    seen: dict[str, int] = {}
    for line, item in enumerate(items, start=HEADER_LINE + 1):
        if item.item_id in seen:
            raise BenchmarkFileError(
                f"benchmark {path}, line {line}: duplicate source item, already seen on line "
                f"{seen[item.item_id]}"
            )
        seen[item.item_id] = line


def _check_source_item_ids(
    path: Path,
    items: tuple[BenchmarkItem, ...],
    expected_source_item_ids: Sequence[str],
) -> None:
    expected = set(expected_source_item_ids)
    actual = {item.source_item_id for item in items}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BenchmarkFileError(
            f"benchmark {path}: source item set mismatch; missing={missing}, extra={extra}"
        )
