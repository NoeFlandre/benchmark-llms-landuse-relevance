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
    """One labelled sentence with stable cross-language source identity."""

    item_id: str
    source_item_id: str
    language: str
    sentence: str
    label: Label
    polygon_name: str = ""
    region: str = ""
    source: str = ""


def item_id_for(source_item_id: str, language: str) -> str:
    """Identify one translated source item without hashing translated text."""
    digest = hashlib.sha256(f"{source_item_id}\0{language}".encode()).hexdigest()
    return digest[:ITEM_ID_LENGTH]


def build_item(row: Mapping[str, str]) -> BenchmarkItem:
    """Validate one raw benchmark row and build the corresponding item."""
    sentence = _required(row, "sentence").strip()
    if not sentence:
        raise InvalidRowError("benchmark row has a blank sentence")
    source_item_id = _required(row, "source_item_id").strip()
    language = _required(row, "language").strip().lower()
    if not source_item_id:
        raise InvalidRowError("benchmark row has a blank source_item_id")
    if not language:
        raise InvalidRowError("benchmark row has a blank language")
    raw_label = _required(row, "label").strip().lower()
    try:
        label = Label(raw_label)
    except ValueError as exc:
        raise InvalidRowError(f"unknown label {raw_label!r}; expected 'yes' or 'no'") from exc
    return BenchmarkItem(
        item_id=item_id_for(source_item_id, language),
        source_item_id=source_item_id,
        language=language,
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
