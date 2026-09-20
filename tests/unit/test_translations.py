import hashlib
import json
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.translations import (
    TranslationDataError,
    available_languages,
    load_language_benchmark,
    load_manifest,
    source_item_ids_for_rows,
)

UPSTREAM_COLUMNS = (
    "sentence",
    "label",
    "polygon_name",
    "h3_cell",
    "latitude",
    "longitude",
    "source",
    "region",
    "source_url",
)
VENDORED_COLUMNS = (*UPSTREAM_COLUMNS, "source_item_id", "language")


def _rows(language: str, first_sentence: str = "Trees.") -> list[dict[str, object]]:
    return [
        {
            "sentence": first_sentence if language == "en" else "Arbres.",
            "label": "yes",
            "polygon_name": "Rabi",
            "h3_cell": "839116fffffffff",
            "latitude": -16.4,
            "longitude": -122.1,
            "source": "wikipedia",
            "region": "fiji",
            "source_url": "https://example.org/a",
        },
        {
            "sentence": "Rocks." if language == "en" else "Roches.",
            "label": "no",
            "polygon_name": "Cruzen Island",
            "h3_cell": "83f35efffffffff",
            "latitude": -74.7,
            "longitude": -140.3,
            "source": "website",
            "region": "antarctica",
            "source_url": "https://example.org/b",
        },
    ]


def _write_language(
    root: Path, language: str, rows: list[dict[str, object]], ids: list[str]
) -> str:
    directory = root / language
    directory.mkdir(parents=True)
    path = directory / f"v3-final-{language}.csv"
    lines = [",".join(VENDORED_COLUMNS)]
    for row, source_item_id in zip(rows, ids, strict=True):
        values = [*(str(row[column]) for column in UPSTREAM_COLUMNS), source_item_id, language]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(root: Path, languages: tuple[str, ...] = ("en", "fr")) -> None:
    source_ids = list(source_item_ids_for_rows(_rows("en")))
    files = {}
    for language in sorted(languages):
        digest = _write_language(root, language, _rows(language), source_ids)
        files[language] = {
            "path": f"{language}/v3-final-{language}.csv",
            "rows": len(source_ids),
            "sha256": digest,
        }
    manifest = {
        "dataset": "test/dataset",
        "revision": "revision-1",
        "split": "train",
        "languages": list(languages),
        "row_count": len(source_ids),
        "source_item_ids": source_ids,
        "files": files,
    }
    manifest["whole_set_sha256"] = hashlib.sha256(
        "\n".join(
            f"{language}:{files[language]['sha256']}" for language in sorted(languages)
        ).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_available_languages_are_sorted_from_the_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path, languages=("fr", "en"))

    assert available_languages(tmp_path) == ("en", "fr")


def test_loads_a_language_and_preserves_shared_source_identity(tmp_path: Path) -> None:
    _write_manifest(tmp_path)

    english = load_language_benchmark(tmp_path, "en")
    french = load_language_benchmark(tmp_path, "fr")

    assert len(english) == len(french) == 2
    assert [item.source_item_id for item in english] == [item.source_item_id for item in french]
    assert [item.language for item in french] == ["fr", "fr"]
    assert [item.item_id for item in english] != [item.item_id for item in french]


def test_unknown_language_lists_the_available_codes(tmp_path: Path) -> None:
    _write_manifest(tmp_path)

    with pytest.raises(TranslationDataError, match=r"unknown language 'xx'.*en.*fr"):
        load_language_benchmark(tmp_path, "xx")


def test_manifest_round_trips_with_file_inventory(tmp_path: Path) -> None:
    _write_manifest(tmp_path)

    manifest = load_manifest(tmp_path)

    assert manifest.languages == ("en", "fr")
    assert manifest.row_count == 2
    assert manifest.source_item_ids == tuple(source_item_ids_for_rows(_rows("en")))


def test_rejects_a_language_file_with_a_mismatched_source_set(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    path = tmp_path / "fr" / "v3-final-fr.csv"
    source_ids = list(source_item_ids_for_rows(_rows("en")))
    path.write_text(
        path.read_text(encoding="utf-8").replace(source_ids[1], "source-extra"),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["fr"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["whole_set_sha256"] = hashlib.sha256(
        "\n".join(
            f"{language}:{manifest['files'][language]['sha256']}"
            for language in sorted(manifest["files"])
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TranslationDataError, match="source item set"):
        load_language_benchmark(tmp_path, "fr")
