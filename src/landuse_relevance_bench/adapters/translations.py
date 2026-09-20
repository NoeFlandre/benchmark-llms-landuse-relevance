"""Manifest and loader for the vendored multilingual benchmark."""

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from landuse_relevance_bench.adapters.benchmark_csv import BenchmarkFileError, load_benchmark
from landuse_relevance_bench.adapters.hashing import sha256_of_file
from landuse_relevance_bench.domain.dataset import BenchmarkItem

MANIFEST_FILENAME = "manifest.json"
LANGUAGE_PATTERN = re.compile(r"^[a-z]{2,3}$")
LANGUAGE_NEUTRAL_COLUMNS = (
    "label",
    "polygon_name",
    "h3_cell",
    "latitude",
    "longitude",
    "source",
    "region",
    "source_url",
)
NUMERIC_COLUMNS = {"latitude", "longitude"}


class TranslationDataError(ValueError):
    """Raised when the multilingual manifest or one of its files is invalid."""


@dataclass(frozen=True, slots=True)
class TranslationFile:
    path: str
    rows: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TranslationManifest:
    dataset: str
    revision: str
    split: str
    languages: tuple[str, ...]
    row_count: int
    source_item_ids: tuple[str, ...]
    files: dict[str, TranslationFile]
    whole_set_sha256: str


def load_manifest(root: Path) -> TranslationManifest:
    """Read and validate the translation manifest without loading CSV rows."""
    path = root / MANIFEST_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        languages = tuple(sorted(payload["languages"]))
        source_item_ids = tuple(payload["source_item_ids"])
        files = {language: TranslationFile(**payload["files"][language]) for language in languages}
        manifest = TranslationManifest(
            dataset=payload["dataset"],
            revision=payload["revision"],
            split=payload["split"],
            languages=languages,
            row_count=payload["row_count"],
            source_item_ids=source_item_ids,
            files=files,
            whole_set_sha256=payload["whole_set_sha256"],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TranslationDataError(f"invalid translation manifest at {path}: {exc}") from exc
    _validate_manifest(manifest, path)
    return manifest


def available_languages(root: Path) -> tuple[str, ...]:
    """Return the sorted language codes declared by the manifest."""
    return load_manifest(root).languages


def load_language_benchmark(root: Path, language: str) -> tuple[BenchmarkItem, ...]:
    """Load one language after validating its manifest entry and source IDs."""
    manifest = load_manifest(root)
    normalized = language.strip().lower()
    if normalized not in manifest.languages:
        available = ", ".join(manifest.languages)
        raise TranslationDataError(f"unknown language {language!r}; available: {available}")
    entry = manifest.files[normalized]
    path = _safe_child(root, entry.path)
    if not path.is_file():
        raise TranslationDataError(f"missing translation file for {normalized!r}: {path}")
    actual_sha256 = sha256_of_file(path)
    if actual_sha256 != entry.sha256:
        raise TranslationDataError(
            f"translation file {path} has sha256 {actual_sha256}, expected {entry.sha256}"
        )
    try:
        items = load_benchmark(
            path,
            expected_language=normalized,
            expected_source_item_ids=manifest.source_item_ids,
        )
    except BenchmarkFileError as exc:
        raise TranslationDataError(str(exc)) from exc
    if len(items) != entry.rows or len(items) != manifest.row_count:
        raise TranslationDataError(
            f"translation {normalized!r} has {len(items)} rows; expected {manifest.row_count}"
        )
    return items


def source_item_ids_for_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Create stable source IDs from fields shared by all translations."""
    occurrences: dict[tuple[str, ...], int] = {}
    result: list[str] = []
    for row in rows:
        key = tuple(
            _canonical_value(column, row.get(column)) for column in LANGUAGE_NEUTRAL_COLUMNS
        )
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        payload = json.dumps(
            {"key": key, "occurrence": occurrence},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        result.append(hashlib.sha256(payload).hexdigest()[:16])
    return tuple(result)


def whole_set_sha256(files: Mapping[str, TranslationFile]) -> str:
    """Hash the sorted language/file digest inventory."""
    inventory = "\n".join(f"{language}:{files[language].sha256}" for language in sorted(files))
    return hashlib.sha256(inventory.encode()).hexdigest()


def _validate_manifest(manifest: TranslationManifest, path: Path) -> None:
    if not manifest.languages:
        raise TranslationDataError(f"translation manifest {path} declares no languages")
    if tuple(sorted(set(manifest.languages))) != manifest.languages:
        raise TranslationDataError(
            f"translation manifest {path} has duplicate or unsorted languages"
        )
    if any(not LANGUAGE_PATTERN.fullmatch(language) for language in manifest.languages):
        raise TranslationDataError(f"translation manifest {path} has an invalid language code")
    if len(set(manifest.source_item_ids)) != len(manifest.source_item_ids):
        raise TranslationDataError(f"translation manifest {path} has duplicate source item IDs")
    if len(manifest.source_item_ids) != manifest.row_count:
        raise TranslationDataError(f"translation manifest {path} has an invalid source item count")
    if set(manifest.files) != set(manifest.languages):
        raise TranslationDataError(f"translation manifest {path} has an incomplete file inventory")
    if whole_set_sha256(manifest.files) != manifest.whole_set_sha256:
        raise TranslationDataError(f"translation manifest {path} has an invalid whole-set digest")


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise TranslationDataError(f"manifest path escapes translation root: {relative!r}") from exc
    return path


def _canonical_value(column: str, value: Any) -> str:
    if value is None:
        return ""
    if column in NUMERIC_COLUMNS:
        try:
            number = Decimal(str(value).strip())
        except (InvalidOperation, ValueError) as exc:
            raise TranslationDataError(
                f"invalid numeric source field {column!r}: {value!r}"
            ) from exc
        if not number.is_finite():
            raise TranslationDataError(f"non-finite numeric source field {column!r}: {value!r}")
        normalized = format(number, "f")
        normalized = normalized.rstrip("0").rstrip(".")
        return normalized or "0"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise TranslationDataError(f"non-finite source field {column!r}: {value!r}")
    return str(value).strip()
