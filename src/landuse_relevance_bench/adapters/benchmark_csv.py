"""Loading the labelled benchmark from its CSV file."""

import csv
from pathlib import Path

from landuse_relevance_bench.domain.dataset import BenchmarkItem, InvalidRowError, build_item

REQUIRED_COLUMNS = ("sentence", "label")
HEADER_LINE = 1


class BenchmarkFileError(ValueError):
    """Raised when a benchmark CSV cannot be loaded as a set of items."""


def load_benchmark(path: Path) -> tuple[BenchmarkItem, ...]:
    """Read every row of ``path`` into a validated, uniquely identified item."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _check_columns(path, reader.fieldnames)
        items = tuple(_build(path, row, line) for line, row in enumerate(reader, start=2))
    if not items:
        raise BenchmarkFileError(f"benchmark {path} has no rows")
    _check_unique(path, items)
    return items


def _check_columns(path: Path, fieldnames: list[str] | None) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in (fieldnames or ())]
    if missing:
        raise BenchmarkFileError(f"benchmark {path} is missing the column(s) {missing}")


def _build(path: Path, row: dict[str, str], line: int) -> BenchmarkItem:
    try:
        return build_item(row)
    except InvalidRowError as exc:
        raise BenchmarkFileError(f"benchmark {path}, line {line}: {exc}") from exc


def _check_unique(path: Path, items: tuple[BenchmarkItem, ...]) -> None:
    seen: dict[str, int] = {}
    for line, item in enumerate(items, start=HEADER_LINE + 1):
        if item.item_id in seen:
            raise BenchmarkFileError(
                f"benchmark {path}, line {line}: duplicate sentence, already seen on line "
                f"{seen[item.item_id]}"
            )
        seen[item.item_id] = line
