"""Validation of each vendored source, through ``vendor_dataset`` with injected fetchers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from landuse_relevance_bench.adapters.translations import TranslationDataError
from scripts import vendor_multilingual_dataset as vendor
from scripts.vendor_multilingual_dataset import UPSTREAM_COLUMNS, vendor_dataset

DATASET = "test/multilingual"


def _row(sentence: str, *, label: object, polygon_name: str) -> dict[str, object]:
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


TREES = _row("Trees.", label="yes", polygon_name="Rabi")
ROCKS = _row("Rocks.", label="no", polygon_name="Cruzen Island")


def _viewer_fetcher(
    splits: object,
    rows: Mapping[str, list[dict[str, object]]],
) -> Callable[[str], object]:
    """Route dataset-viewer and revision requests to canned answers."""

    def fetch(url: str) -> object:
        parsed = urlparse(url)
        if parsed.path == "/splits":
            return splits
        if parsed.path == "/rows":
            query = parse_qs(parsed.query)
            page = rows[query["config"][0]]
            offset, length = int(query["offset"][0]), int(query["length"][0])
            return {"rows": [{"row": row} for row in page[offset : offset + length]]}
        if parsed.path.startswith("/api/datasets/"):
            return {"sha": "revision-1"}
        raise AssertionError(f"unexpected request {url}")

    return fetch


def _splits(*entries: dict[str, object]) -> dict[str, object]:
    return {"splits": list(entries)}


def _entry(language: str, *, rows: object = 2, split: str = "train") -> dict[str, object]:
    return {"config": language, "split": split, "num_rows": rows}


def _viewer(
    tmp_path: Path,
    fetcher: Callable[[str], object],
    *,
    expected_row_count: int = 2,
    expected_language_count: int = 2,
    force: bool = False,
) -> None:
    vendor_dataset(
        dataset=DATASET,
        split="train",
        output=tmp_path / "out",
        expected_row_count=expected_row_count,
        expected_language_count=expected_language_count,
        source="viewer",
        fetcher=fetcher,
        force=force,
    )


def test_viewer_source_writes_every_language_in_source_order(tmp_path: Path) -> None:
    rows = {"en": [TREES, ROCKS], "fr": [TREES, ROCKS]}

    _viewer(tmp_path, _viewer_fetcher(_splits(_entry("en"), _entry("fr")), rows))

    assert (tmp_path / "out" / "en" / "v3-final-en.csv").is_file()
    assert (tmp_path / "out" / "fr" / "v3-final-fr.csv").is_file()


def test_viewer_pages_through_rows_until_the_expected_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vendor, "PAGE_LENGTH", 1)
    rows = {"en": [TREES, ROCKS], "fr": [TREES, ROCKS]}

    _viewer(tmp_path, _viewer_fetcher(_splits(_entry("en"), _entry("fr")), rows))

    assert (tmp_path / "out" / "fr" / "v3-final-fr.csv").read_text(encoding="utf-8").count(
        "\n"
    ) == 3


def test_viewer_source_refuses_a_language_with_different_source_order(tmp_path: Path) -> None:
    rows = {"en": [TREES, ROCKS], "fr": [ROCKS, TREES]}

    with pytest.raises(TranslationDataError, match="'fr' does not preserve the source item order"):
        _viewer(tmp_path, _viewer_fetcher(_splits(_entry("en"), _entry("fr")), rows))


def test_viewer_source_writes_booleans_as_yes_and_no_and_missing_values_as_empty(
    tmp_path: Path,
) -> None:
    yes_row = _row("Trees.", label=True, polygon_name="Rabi")
    no_row = _row("Rocks.", label=False, polygon_name="Cruzen Island")
    no_row["latitude"] = None
    rows = {"en": [yes_row, no_row], "fr": [yes_row, no_row]}

    _viewer(tmp_path, _viewer_fetcher(_splits(_entry("en"), _entry("fr")), rows))

    text = (tmp_path / "out" / "en" / "v3-final-en.csv").read_text(encoding="utf-8")
    assert "Trees.,yes,Rabi" in text
    assert "Rocks.,no,Cruzen Island,839116fffffffff,,-122.1" in text


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        ("not a mapping", "splits response is not a JSON object"),
        ({"other": []}, "splits response has no split list"),
        ({"splits": "train"}, "splits response has no split list"),
        ({"splits": ["train"]}, "split entry is not a JSON object"),
        ({"splits": [_entry("EN")]}, "invalid language config in splits response: 'EN'"),
        ({"splits": [_entry("en"), _entry("en")]}, "duplicate language config 'en'"),
        ({"splits": [_entry("en", rows=3)]}, "'en' declares 3 rows; expected 2 rows"),
        ({"splits": [_entry("en", rows="2")]}, "'en' declares '2' rows; expected 2 rows"),
        ({"splits": [_entry("en", split="test")]}, "has no 'train' language splits"),
    ],
)
def test_viewer_split_discovery_refuses_inconsistent_listings(
    tmp_path: Path, splits: object, message: str
) -> None:
    with pytest.raises(TranslationDataError, match=message):
        _viewer(tmp_path, _viewer_fetcher(splits, {}))


def test_viewer_split_without_a_declared_row_count_uses_the_expected_count(
    tmp_path: Path,
) -> None:
    rows = {"en": [TREES, ROCKS], "fr": [TREES, ROCKS]}
    splits = _splits(_entry("en", rows=None), _entry("fr", rows=None))

    _viewer(tmp_path, _viewer_fetcher(splits, rows))

    assert (tmp_path / "out" / "en" / "v3-final-en.csv").is_file()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ({"en": "rows"}, "rows response for 'en' has no row list"),
        ({"en": [{"row": TREES}, "not a row"]}, "row entry for 'en' is not a JSON object"),
        ({"en": [{"missing": TREES}, {"row": ROCKS}]}, "row payload for 'en' is not a JSON"),
        (
            {"en": [{"row": {"sentence": "Trees."}}, {"row": ROCKS}]},
            "'en' is missing upstream column",
        ),
        ({"en": [{"row": TREES}]}, "'en' returned 1 rows; expected 2 rows"),
    ],
)
def test_viewer_row_pages_must_be_complete_and_well_formed(
    tmp_path: Path, rows: Mapping[str, object], message: str
) -> None:
    def fetch(url: str) -> object:
        parsed = urlparse(url)
        if parsed.path == "/splits":
            return _splits(_entry("en"))
        if parsed.path.startswith("/api/datasets/"):
            return {"sha": "revision-1"}
        return {"rows": rows["en"]}

    with pytest.raises(TranslationDataError, match=message):
        _viewer(tmp_path, fetch, expected_language_count=1)


def test_language_count_must_match_the_expected_languages(tmp_path: Path) -> None:
    rows = {"en": [TREES, ROCKS], "fr": [TREES, ROCKS]}

    with pytest.raises(TranslationDataError, match="found 2 language splits; expected 3"):
        _viewer(
            tmp_path,
            _viewer_fetcher(_splits(_entry("en"), _entry("fr")), rows),
            expected_language_count=3,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expected_row_count": 0}, "expected_row_count must be positive"),
        ({"expected_language_count": 0}, "expected_language_count must be positive"),
    ],
)
def test_nonpositive_expectations_are_refused_before_any_fetch(
    tmp_path: Path, kwargs: dict[str, int], message: str
) -> None:
    def never(url: str) -> object:
        raise AssertionError(f"fetched {url}")

    with pytest.raises(TranslationDataError, match=message):
        vendor_dataset(
            dataset=DATASET,
            split="train",
            output=tmp_path / "out",
            fetcher=never,
            **kwargs,
        )


def test_unknown_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TranslationDataError, match="unknown vendor source 'ftp'"):
        vendor_dataset(
            dataset=DATASET,
            split="train",
            output=tmp_path / "out",
            source="ftp",
            fetcher=lambda url: {},
        )


# Hub translations: one CSV per language in the pinned repository tree.

HEADER = ",".join(UPSTREAM_COLUMNS)


def _csv(*lines: str) -> bytes:
    return ("\n".join([HEADER, *lines]) + "\n").encode("utf-8")


TREE_EN = "Trees.,yes,Rabi,839116fffffffff,-16.4,-122.1,wikipedia,fiji,https://example.org/a"
TREE_FR = "Arbres.,yes,Rabi,839116fffffffff,-16.4,-122.1,wikipedia,fiji,https://example.org/a"
ROCK_EN = "Rocks.,no,Cruzen Island,83f35efffffffff,-74.7,-140.3,website,antarctica,https://e/b"
ROCK_FR = "Roches.,no,Cruzen Island,83f35efffffffff,-74.7,-140.3,website,antarctica,https://e/b"


def _tree(*languages: str) -> list[dict[str, object]]:
    return [
        {"type": "file", "path": f"data/translations/{language}/v3-final-{language}.csv"}
        for language in languages
    ]


def _hub_fetcher(tree: object, revision: object = "revision-1") -> Callable[[str], object]:
    def fetch(url: str) -> object:
        parsed = urlparse(url)
        if "/tree/" in parsed.path:
            return tree
        if parsed.path.startswith("/api/datasets/"):
            return {"sha": revision} if revision is not None else {}
        raise AssertionError(f"unexpected request {url}")

    return fetch


def _hub_contents(files: Mapping[str, bytes]) -> Callable[[str], bytes]:
    def fetch(url: str) -> bytes:
        relative = urlparse(url).path.split("/resolve/", 1)[1].split("/", 1)[1]
        return files[relative]

    return fetch


def _hub(tmp_path: Path, tree: object, files: Mapping[str, bytes], **options: Any) -> None:
    settings: dict[str, Any] = {
        "expected_language_count": 2,
        "expected_row_count": 2,
        "revision": "revision-1",
        "split": "train",
        "force": False,
        **options,
    }
    vendor_dataset(
        dataset=DATASET,
        split=settings["split"],
        output=tmp_path / "out",
        expected_row_count=settings["expected_row_count"],
        expected_language_count=settings["expected_language_count"],
        source="hub",
        fetcher=_hub_fetcher(tree, settings["revision"]),
        content_fetcher=_hub_contents(files),
        force=settings["force"],
    )


def _two_language_files(en: bytes | None = None, fr: bytes | None = None) -> dict[str, bytes]:
    return {
        "data/translations/en/v3-final-en.csv": en or _csv(TREE_EN, ROCK_EN),
        "data/translations/fr/v3-final-fr.csv": fr or _csv(TREE_FR, ROCK_FR),
    }


def test_hub_source_refuses_splits_other_than_train(tmp_path: Path) -> None:
    with pytest.raises(TranslationDataError, match="only provides the train split"):
        _hub(tmp_path, tree=_tree("en", "fr"), files={}, split="validation")


@pytest.mark.parametrize(
    ("tree", "message"),
    [
        ({"tree": "not a list"}, "Hub tree response has no file list"),
        ([], "Hub dataset 'test/multilingual' has no translation CSV files"),
        (
            _tree("en", "en"),
            "duplicate Hub translation file for 'en'",
        ),
    ],
)
def test_hub_tree_must_list_each_translation_once(
    tmp_path: Path, tree: object, message: str
) -> None:
    with pytest.raises(TranslationDataError, match=message):
        _hub(tmp_path, tree=tree, files=_two_language_files())


def test_hub_tree_ignores_entries_that_are_not_translation_files(tmp_path: Path) -> None:
    tree = [
        {"type": "directory", "path": "data/translations/en"},
        {"type": "file", "path": 42},
        {"type": "file", "path": "README.md"},
        {"type": "file", "path": "data/other/en/v3-final-en.csv"},
        {"type": "file", "path": "data/translations/en/notes.csv"},
        {"type": "file", "path": "data/translations/EN/v3-final-EN.csv"},
        *_tree("en", "fr"),
    ]

    _hub(tmp_path, tree={"tree": tree}, files=_two_language_files())

    assert (tmp_path / "out" / "manifest.json").is_file()


def test_hub_language_count_must_match_the_expected_languages(tmp_path: Path) -> None:
    with pytest.raises(TranslationDataError, match="found 2 language splits; expected 3"):
        _hub(
            tmp_path,
            tree=_tree("en", "fr"),
            files=_two_language_files(),
            expected_language_count=3,
        )


def test_hub_revision_must_be_present_in_the_dataset_metadata(tmp_path: Path) -> None:
    with pytest.raises(TranslationDataError, match="has no revision SHA"):
        _hub(tmp_path, tree=_tree("en", "fr"), files={}, revision="  ")


@pytest.mark.parametrize(
    ("en", "message"),
    [
        (b"\xff\xfe bad", "Hub translation 'en' is not UTF-8"),
        ("﻿".encode() + _csv(TREE_EN, ROCK_EN), "Hub translation 'en' has a UTF-8 BOM"),
        (b"sentence,label\nTrees.,yes\n", "Hub translation 'en' has header"),
        (_csv(TREE_EN, ROCK_EN + ",extra"), "Hub translation 'en' has a malformed row at line 3"),
        (_csv("Trees.,yes,Rabi"), "Hub translation 'en' has a malformed row at line 2"),
        (_csv(TREE_EN, ROCK_EN, ROCK_EN), "'en' returned 3 rows; expected 2 rows"),
    ],
)
def test_hub_csv_files_must_be_clean_utf8_with_the_upstream_header(
    tmp_path: Path, en: bytes, message: str
) -> None:
    with pytest.raises(TranslationDataError, match=message):
        _hub(tmp_path, tree=_tree("en", "fr"), files=_two_language_files(en=en))


def test_hub_languages_must_share_the_same_source_item_order(tmp_path: Path) -> None:
    files = _two_language_files(fr=_csv(ROCK_FR, TREE_FR))

    with pytest.raises(TranslationDataError, match="'fr' does not preserve the source item order"):
        _hub(tmp_path, tree=_tree("en", "fr"), files=files)


# Installing the output directory.


def test_existing_output_is_kept_unless_forced(tmp_path: Path) -> None:
    existing = tmp_path / "out"
    existing.mkdir()
    (existing / "keep.txt").write_text("previous run", encoding="utf-8")

    with pytest.raises(TranslationDataError, match="pass --force to replace it"):
        _hub(tmp_path, tree=_tree("en", "fr"), files=_two_language_files())

    assert (existing / "keep.txt").read_text(encoding="utf-8") == "previous run"
    assert not list(tmp_path.glob(".out-*"))


def test_forced_install_replaces_the_previous_output(tmp_path: Path) -> None:
    existing = tmp_path / "out"
    existing.mkdir()
    (existing / "stale.txt").write_text("previous run", encoding="utf-8")

    _hub(tmp_path, tree=_tree("en", "fr"), files=_two_language_files(), force=True)

    assert not (existing / "stale.txt").exists()
    assert (existing / "en" / "v3-final-en.csv").is_file()
    assert not list(tmp_path.glob(".out-*"))


def test_output_path_that_is_a_file_is_refused_and_leaves_no_staging(tmp_path: Path) -> None:
    (tmp_path / "out").write_text("not a directory", encoding="utf-8")

    with pytest.raises(TranslationDataError, match="output path is not a directory"):
        _hub(tmp_path, tree=_tree("en", "fr"), files=_two_language_files(), force=True)

    assert (tmp_path / "out").read_text(encoding="utf-8") == "not a directory"
    assert not list(tmp_path.glob(".out-*"))
