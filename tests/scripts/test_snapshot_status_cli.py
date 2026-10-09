"""The snapshot status command line: option parsing and the file it writes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from factories import make_result
from landuse_relevance_bench.adapters.results_store import write_run
from scripts import snapshot_status


def _store(results: Path, pairs: list[tuple[str, str]]) -> Path:
    for model_id, language in pairs:
        write_run(make_result(model_id, language=language), results)
    return results


def _run(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["snapshot_status.py", *arguments])
    snapshot_status.main()


def test_main_writes_the_status_file_next_to_the_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = _store(tmp_path / "results", [("a/model", "en"), ("a/model", "fr")])

    _run(monkeypatch, str(results), "--benchmark-name", "bench-x", "--model-id", "a/model")

    text = (results / "SNAPSHOT_STATUS.md").read_text(encoding="utf-8")
    assert "- Benchmark: `bench-x`" in text
    assert "- Status: complete." in text
    assert "- Completed model-language runs: 2 of 2" in text
    assert "- `a/model`: complete, 2/2 languages" in text
    assert capsys.readouterr().out.strip() == f"wrote {results / 'SNAPSHOT_STATUS.md'}"


def test_model_language_overrides_narrow_what_each_model_must_cover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = _store(
        tmp_path / "results",
        [("a/model", "en"), ("b/model", "en"), ("b/model", "fr")],
    )

    _run(
        monkeypatch,
        str(results),
        "--model-id",
        "a/model",
        "--model-id",
        "b/model",
        "--model-languages",
        "a/model=en",
    )

    text = (results / "SNAPSHOT_STATUS.md").read_text(encoding="utf-8")
    assert "- `a/model`: complete, 1/1 languages" in text
    assert "- `b/model`: complete, 2/2 languages" in text
    assert "- Status: complete." in text


def test_a_partial_sweep_is_reported_as_in_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = _store(tmp_path / "results", [("a/model", "en")])

    _run(
        monkeypatch,
        str(results),
        "--model-id",
        "a/model",
        "--model-id",
        "b/model",
    )

    text = (results / "SNAPSHOT_STATUS.md").read_text(encoding="utf-8")
    assert "- Status: in progress; not the final release." in text
    assert "- `b/model`: in progress, 0/1 languages" in text


def test_an_empty_results_tree_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="no run results found"):
        _run(monkeypatch, str(empty), "--model-id", "a/model")


def test_overrides_for_models_outside_the_roster_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = _store(tmp_path / "results", [("a/model", "en")])

    with pytest.raises(ValueError, match="unrostered models: \\['c/model'\\]"):
        _run(
            monkeypatch,
            str(results),
            "--model-id",
            "a/model",
            "--model-languages",
            "c/model=en",
        )


@pytest.mark.parametrize(
    "value",
    ["a/model", "=en", "a/model=", "a/model=en,", "a/model=en,en"],
)
def test_malformed_model_languages_are_usage_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    results = _store(tmp_path / "results", [("a/model", "en")])

    with pytest.raises(SystemExit) as exit_info:
        _run(monkeypatch, str(results), "--model-languages", value)

    assert exit_info.value.code == 2
    assert "--model-languages must be MODEL=LANG,LANG with unique non-empty codes" in (
        capsys.readouterr().err
    )


def test_model_languages_may_be_given_once_per_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _store(tmp_path / "results", [("a/model", "en")])

    with pytest.raises(SystemExit) as exit_info:
        _run(
            monkeypatch,
            str(results),
            "--model-languages",
            "a/model=en",
            "--model-languages",
            "a/model=fr",
        )

    assert exit_info.value.code == 2
    assert "--model-languages was repeated for a/model" in capsys.readouterr().err
