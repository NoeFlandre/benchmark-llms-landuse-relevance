"""The Grid'5000 collection command line, over site result trees and a small manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factories import make_result
from landuse_relevance_bench.adapters.results_store import write_run
from landuse_relevance_bench.adapters.translations import TranslationFile, whole_set_sha256
from landuse_relevance_bench.domain.roster import model_ids
from scripts import g5k_collect as collect

GENERATIVE_MODEL = model_ids()[0]


def _data_root(tmp_path: Path, languages: tuple[str, ...] = ("en", "fr")) -> Path:
    """A translation manifest whose digests are consistent; the CSV files are not needed."""
    files = {
        language: TranslationFile(
            path=f"{language}/v3-final-{language}.csv",
            rows=1,
            sha256=f"{index:064x}",
        )
        for index, language in enumerate(languages, start=1)
    }
    manifest = {
        "dataset": "test/multilingual",
        "revision": "rev-1",
        "split": "train",
        "languages": list(languages),
        "row_count": 1,
        "source_item_ids": ["item-1"],
        "files": {
            language: {"path": file.path, "rows": file.rows, "sha256": file.sha256}
            for language, file in files.items()
        },
        "whole_set_sha256": whole_set_sha256(files),
    }
    root = tmp_path / "translations"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _sites(tmp_path: Path, pairs: dict[str, list[tuple[str, str]]], commit: str) -> list[str]:
    """One result tree per site, holding the given (model, language) runs."""
    arguments: list[str] = []
    for site_name, site_pairs in pairs.items():
        site_root = tmp_path / site_name
        for model_id, language in site_pairs:
            write_run(make_result(model_id, language=language, source_commit=commit), site_root)
        arguments += ["--site", f"{site_name}={site_root}"]
    return arguments


def test_main_merges_a_complete_model_and_reports_its_source_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = _data_root(tmp_path)
    sites = _sites(
        tmp_path,
        {"nancy": [("a/model", "en")], "lyon": [("a/model", "fr")]},
        commit="commit-a",
    )
    output = tmp_path / "merged"

    status = collect.main(
        [
            *sites,
            "--data-root",
            str(data_root),
            "--output",
            str(output),
            "--model-id",
            "a/model",
        ]
    )

    assert status == 0
    assert "merged 2 pairs from source commit commit-a" in capsys.readouterr().out
    assert (output / "en" / "a__model.json").is_file()
    assert (output / "fr" / "a__model.json").is_file()


def test_main_refuses_results_from_another_source_commit(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    sites = _sites(tmp_path, {"nancy": [("a/model", "en"), ("a/model", "fr")]}, commit="commit-a")

    with pytest.raises(collect.CollectionError, match="expected 'commit-b'"):
        collect.main(
            [
                *sites,
                "--data-root",
                str(data_root),
                "--output",
                str(tmp_path / "merged"),
                "--model-id",
                "a/model",
                "--source-commit",
                "commit-b",
            ]
        )


def test_main_replaces_an_existing_merge_only_when_forced(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    sites = _sites(tmp_path, {"nancy": [("a/model", "en"), ("a/model", "fr")]}, commit="commit-a")
    arguments = [
        *sites,
        "--data-root",
        str(data_root),
        "--output",
        str(tmp_path / "merged"),
        "--model-id",
        "a/model",
    ]
    assert collect.main(arguments) == 0

    with pytest.raises(collect.CollectionError, match="pass --force"):
        collect.main(arguments)
    assert collect.main([*arguments, "--force"]) == 0


def test_model_id_and_scoring_only_cannot_be_combined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        collect.main(
            [
                "--site",
                f"nancy={tmp_path}",
                "--data-root",
                str(_data_root(tmp_path)),
                "--model-id",
                "a/model",
                "--scoring-only",
            ]
        )

    assert exit_info.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_site_argument_must_name_a_result_directory(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        collect.main(["--site", "nancy", "--data-root", "unused"])

    assert exit_info.value.code == 2
    assert "site must have the form NAME=RESULT_DIRECTORY" in capsys.readouterr().err


def test_default_collection_expects_the_whole_generative_roster(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    empty_site = tmp_path / "nancy"
    empty_site.mkdir()

    with pytest.raises(collect.CollectionError, match="missing pairs"):
        collect.main(
            [
                "--site",
                f"nancy={empty_site}",
                "--data-root",
                str(data_root),
                "--output",
                str(tmp_path / "merged"),
            ]
        )


def test_scoring_only_collection_rejects_generative_runs(tmp_path: Path) -> None:
    data_root = _data_root(tmp_path)
    sites = _sites(tmp_path, {"nancy": [(GENERATIVE_MODEL, "en")]}, commit="commit-a")

    with pytest.raises(collect.CollectionError, match="unexpected pair"):
        collect.main(
            [
                *sites,
                "--data-root",
                str(data_root),
                "--output",
                str(tmp_path / "merged"),
                "--scoring-only",
            ]
        )
