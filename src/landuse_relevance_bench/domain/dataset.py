"""The benchmark items under test."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from landuse_relevance_bench.domain.labels import Label

ITEM_ID_LENGTH = 16


class InvalidRowError(ValueError):
    """Raised when a benchmark row cannot be turned into an item."""


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One labelled sentence, identified by the content of that sentence."""

    item_id: str
    sentence: str
    label: Label
    polygon_name: str = ""
    region: str = ""
    source: str = ""


def item_id_for(sentence: str) -> str:
    """A stable, content-addressed id, so runs stay comparable across orderings."""
    digest = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
    return digest[:ITEM_ID_LENGTH]


def build_item(row: Mapping[str, str]) -> BenchmarkItem:
    """Validate one raw benchmark row and build the corresponding item."""
    sentence = _required(row, "sentence").strip()
    if not sentence:
        raise InvalidRowError("benchmark row has a blank sentence")
    raw_label = _required(row, "label").strip().lower()
    try:
        label = Label(raw_label)
    except ValueError as exc:
        raise InvalidRowError(f"unknown label {raw_label!r}; expected 'yes' or 'no'") from exc
    return BenchmarkItem(
        item_id=item_id_for(sentence),
        sentence=sentence,
        label=label,
        polygon_name=(row.get("polygon_name") or "").strip(),
        region=(row.get("region") or "").strip(),
        source=(row.get("source") or "").strip(),
    )


def _required(row: Mapping[str, str], field: str) -> str:
    value = row.get(field)
    if value is None:
        raise InvalidRowError(f"benchmark row is missing the required field {field!r}")
    return value
