import csv
import hashlib
import io
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from scripts.vendor_multilingual_dataset import UPSTREAM_COLUMNS, vendor_dataset

from landuse_relevance_bench.adapters.translations import (
    TranslationDataError,
    available_languages,
    load_language_benchmark,
    load_manifest,
)


def _row(sentence: str, *, label: str, polygon_name: str) -> dict[str, object]:
    return {
        "sentence": sentence,
        "label": label,
        "polygon_name": polygon_name,
        "h3_cell": "839116fffffffff",
        "latitude": -16.4,
        "longitude": -122.1,
        "source": "wikipedia",
        "region": "fiji",
        "source_url": "https://example.org/a",
    }


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=UPSTREAM_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def test_vendor_writes_verified_language_files_and_manifest(tmp_path: Path) -> None:
    rows = {
        "en": [
            _row("Trees.", label="yes", polygon_name="Rabi"),
            _row("Rocks.", label="no", polygon_name="Cruzen Island"),
        ],
        "fr": [
            _row("Arbres.", label="yes", polygon_name="Rabi"),
            _row("Roches.", label="no", polygon_name="Cruzen Island"),
        ],
    }

    def fake_fetch(url: str) -> object:
        parsed = urlparse(url)
        if parsed.path == "/api/datasets/test/dataset":
            return {"sha": "revision-1"}
        if parsed.path == "/splits":
            return {
                "splits": [
                    {"config": language, "split": "train", "num_rows": len(language_rows)}
                    for language, language_rows in rows.items()
                ]
            }
        if parsed.path == "/rows":
            language = parse_qs(parsed.query)["config"][0]
            return {"rows": [{"row": row} for row in rows[language]]}
        raise AssertionError(f"unexpected URL: {url}")

    output = tmp_path / "translations"
    vendor_dataset(
        dataset="test/dataset",
        split="train",
        output=output,
        expected_row_count=2,
        expected_language_count=2,
        source="viewer",
        fetcher=fake_fetch,
    )

    assert available_languages(output) == ("en", "fr")
    manifest = load_manifest(output)
    assert manifest.revision == "revision-1"
    assert manifest.row_count == 2
    assert [item.source_item_id for item in load_language_benchmark(output, "en")] == [
        item.source_item_id for item in load_language_benchmark(output, "fr")
    ]
    assert [item.item_id for item in load_language_benchmark(output, "en")] != [
        item.item_id for item in load_language_benchmark(output, "fr")
    ]
    assert hashlib.sha256((output / "en" / "v3-final-en.csv").read_bytes()).hexdigest() == (
        manifest.files["en"].sha256
    )


def test_vendor_rejects_language_with_wrong_row_count(tmp_path: Path) -> None:
    def fake_fetch(url: str) -> object:
        parsed = urlparse(url)
        if parsed.path == "/api/datasets/test/dataset":
            return {"sha": "revision-1"}
        if parsed.path == "/splits":
            return {"splits": [{"config": "en", "split": "train", "num_rows": 3}]}
        if parsed.path == "/rows":
            return {"rows": [{"row": _row("Trees.", label="yes", polygon_name="Rabi")}]}
        raise AssertionError(f"unexpected URL: {url}")

    with pytest.raises(TranslationDataError, match="expected 3 rows"):
        vendor_dataset(
            dataset="test/dataset",
            split="train",
            output=tmp_path / "translations",
            expected_row_count=3,
            source="viewer",
            fetcher=fake_fetch,
        )


def test_vendor_hub_source_uses_the_pinned_repository_file_inventory(tmp_path: Path) -> None:
    rows = {
        "en": [
            _row("Trees.", label="yes", polygon_name="Rabi"),
            _row("Rocks.", label="no", polygon_name="Cruzen Island"),
        ],
        "fr": [
            _row("Arbres.", label="yes", polygon_name="Rabi"),
            _row("Roches.", label="no", polygon_name="Cruzen Island"),
        ],
    }

    def fake_fetch(url: str) -> object:
        parsed = urlparse(url)
        if parsed.path == "/api/datasets/test/dataset":
            return {"sha": "revision-1"}
        if parsed.path == "/api/datasets/test/dataset/tree/revision-1":
            return [
                {"type": "file", "path": f"data/translations/{language}/v3-final-{language}.csv"}
                for language in rows
            ]
        raise AssertionError(f"unexpected JSON URL: {url}")

    def fake_content(url: str) -> bytes:
        language = urlparse(url).path.split("/")[-2]
        return _csv_bytes(rows[language])

    output = tmp_path / "translations"
    vendor_dataset(
        dataset="test/dataset",
        split="train",
        output=output,
        expected_row_count=2,
        expected_language_count=2,
        source="hub",
        fetcher=fake_fetch,
        content_fetcher=fake_content,
    )

    assert available_languages(output) == ("en", "fr")
