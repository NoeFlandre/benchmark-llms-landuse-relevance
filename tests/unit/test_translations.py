import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.translations import (
    TranslationDataError,
    TranslationFile,
    TranslationManifest,
    _canonical_value,
    _safe_child,
    _validate_manifest,
    available_languages,
    load_language_benchmark,
    load_manifest,
    source_item_ids_for_rows,
    whole_set_sha256,
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


def _valid_manifest() -> TranslationManifest:
    files = {
        language: TranslationFile(f"{language}.csv", 2, language * 64) for language in ("en", "fr")
    }
    return TranslationManifest(
        dataset="test/dataset",
        revision="revision-1",
        split="train",
        languages=("en", "fr"),
        row_count=2,
        source_item_ids=("one", "two"),
        files=files,
        whole_set_sha256=whole_set_sha256(files),
    )


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


def test_rejects_a_language_file_with_reordered_source_ids(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    path = tmp_path / "fr" / "v3-final-fr.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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

    with pytest.raises(TranslationDataError, match="source item sequence"):
        load_language_benchmark(tmp_path, "fr")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"languages": ()}, "no languages"),
        ({"languages": ("fr", "en")}, "duplicate or unsorted"),
        (
            {"languages": ("x",), "files": {"x": TranslationFile("x.csv", 2, "x" * 64)}},
            "invalid language",
        ),
        ({"source_item_ids": ("one", "one")}, "duplicate source item IDs"),
        ({"row_count": 3}, "invalid source item count"),
        (
            {"files": {**_valid_manifest().files, "de": TranslationFile("de.csv", 2, "d" * 64)}},
            "incomplete file inventory",
        ),
        ({"whole_set_sha256": "0" * 64}, "invalid whole-set digest"),
    ],
)
def test_manifest_validation_rejects_inconsistent_inventory(changes, message: str) -> None:
    manifest = replace(_valid_manifest(), **changes)

    with pytest.raises(TranslationDataError, match=message):
        _validate_manifest(manifest, Path("manifest.json"))


def test_load_language_rejects_missing_file_bad_hash_and_wrong_row_count(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    _write_manifest(missing_root)
    missing = missing_root / "fr" / "v3-final-fr.csv"
    missing.unlink()
    with pytest.raises(TranslationDataError, match="missing translation file"):
        load_language_benchmark(missing_root, "fr")

    hash_root = tmp_path / "hash"
    _write_manifest(hash_root)
    target = hash_root / "fr" / "v3-final-fr.csv"
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(TranslationDataError, match="has sha256"):
        load_language_benchmark(hash_root, "fr")

    rows_root = tmp_path / "rows"
    _write_manifest(rows_root)
    manifest_path = rows_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["fr"]["rows"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TranslationDataError, match="has 2 rows; expected 2"):
        load_language_benchmark(rows_root, "fr")


def test_load_language_wraps_invalid_csv_and_rejects_escaping_paths(tmp_path: Path) -> None:
    escape_root = tmp_path / "escape"
    _write_manifest(escape_root)
    manifest_path = escape_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["fr"]["path"] = "../../outside.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TranslationDataError, match="escapes translation root"):
        load_language_benchmark(escape_root, "fr")

    invalid_root = tmp_path / "invalid"
    _write_manifest(invalid_root)
    target = invalid_root / "fr" / "v3-final-fr.csv"
    target.write_text(
        target.read_text(encoding="utf-8").replace(",fr\n", ",de\n"), encoding="utf-8"
    )
    manifest_path = invalid_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["files"]["fr"]["sha256"] = digest
    manifest["whole_set_sha256"] = hashlib.sha256(
        "\n".join(
            f"{language}:{manifest['files'][language]['sha256']}"
            for language in sorted(manifest["files"])
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TranslationDataError, match="expected language"):
        load_language_benchmark(invalid_root, "fr")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("latitude", "not-a-number", "invalid numeric source field"),
        ("longitude", "nan", "non-finite numeric source field"),
        ("label", float("inf"), "non-finite source field"),
    ],
)
def test_canonical_values_reject_invalid_numeric_data(column, value, message: str) -> None:
    with pytest.raises(TranslationDataError, match=message):
        source_item_ids_for_rows([{column: value}])


def test_canonical_numeric_zero_has_a_stable_spelling(tmp_path: Path) -> None:
    assert _canonical_value("latitude", "0.000") == "0"
    with pytest.raises(TranslationDataError, match="escapes translation root"):
        _safe_child(tmp_path, "../outside.csv")
