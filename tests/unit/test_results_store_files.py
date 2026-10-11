"""Exact bytes, encodings and error rules of the results store's files."""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from factories import make_metadata, make_result
from landuse_relevance_bench.adapters.results_store import (
    SCORING_SUMMARY_COLUMNS,
    THRESHOLD_SWEEP_COLUMNS,
    dumps_run_payload,
    read_run,
    read_runs,
    run_filename,
    write_aggregates_csv,
    write_leaderboard_csv,
    write_reports,
    write_run,
    write_scoring_summary_csv,
    write_threshold_sweep_csv,
)
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunResult
from landuse_relevance_bench.domain.thresholds import DEFAULT_THRESHOLDS

Y, N = Label.YES, Label.NO


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, object]]]:
    """Record the keyword arguments of every ``Path.open`` call, keyed by file name."""
    calls: dict[str, list[dict[str, object]]] = {}
    real_open = Path.open

    def recording_open(self: Path, *args: object, **kwargs: object):
        calls.setdefault(self.name, []).append(dict(kwargs))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    return calls


@pytest.fixture
def text_io_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, object]]]:
    """Record the keyword arguments each ``Path.read_text`` and ``Path.write_text`` call gets.

    The arguments are recorded as the caller passed them, before pathlib resolves a
    missing encoding to the interpreter's default, so an omitted encoding is visible.
    """
    calls: list[tuple[str, str, dict[str, object]]] = []
    real_read, real_write = Path.read_text, Path.write_text

    def read_text(self: Path, *args: object, **kwargs: object) -> str:
        calls.append(("read", self.name, dict(kwargs)))
        return real_read(self, *args, **kwargs)

    def write_text(self: Path, data: str, *args: object, **kwargs: object) -> int:
        calls.append(("write", self.name, dict(kwargs)))
        return real_write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "write_text", write_text)
    return calls


def _scoring_run(model_id: str = "m", language: str = "en") -> RunResult:
    """Two scoring items, both decided correctly: a YES at 0.9 and a NO at 0.2."""
    predictions = (
        Prediction(
            item_id="0" * 16,
            expected=Y,
            predicted=Y,
            raw_output="no=0.100000 yes=0.900000 native=1.0",
        ),
        Prediction(
            item_id="1" * 16,
            expected=N,
            predicted=N,
            raw_output="no=0.800000 yes=0.200000 native=-1.0",
        ),
    )
    metadata = replace(
        make_metadata(model_id=model_id, language=language),
        inference="scoring",
        decision_rule="argmax over the native yes/no scores",
        max_new_tokens=0,
        decoding="none",
        throughput_items_per_second=1.2345,
        peak_vram_bytes=100,
    )
    return RunResult(
        metadata=metadata,
        predictions=predictions,
        metrics=evaluate([p.outcome for p in predictions]),
    )


def _payload_without_language(model_id: str = "m") -> dict:
    payload = make_result(model_id=model_id, language="en").to_dict()
    del payload["metadata"]["language"]
    return payload


def test_dumps_run_payload_is_two_space_indented_sorted_unicode_json() -> None:
    assert dumps_run_payload({"b": 1, "a": "é"}) == '{\n  "a": "é",\n  "b": 1\n}\n'


def test_write_run_writes_the_canonical_payload_as_utf8_under_the_language_directory(
    tmp_path: Path, text_io_calls
) -> None:
    result = make_result(model_id="a/b", language="fr")

    path = write_run(result, tmp_path / "out")

    assert path == tmp_path / "out" / "fr" / "a__b.json"
    assert ("write", "a__b.json", {"encoding": "utf-8"}) in text_io_calls
    assert path.read_text(encoding="utf-8") == dumps_run_payload(result.to_dict())


def test_read_run_keeps_a_language_the_payload_already_declares(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(dumps_run_payload(make_result(language="fr").to_dict()), encoding="utf-8")

    assert read_run(path, legacy_language="en").metadata.language == "fr"


def test_read_run_fills_a_missing_language_from_the_legacy_default(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps(_payload_without_language()), encoding="utf-8")

    assert read_run(path, legacy_language="en").metadata.language == "en"


def test_read_run_without_a_legacy_default_names_the_missing_language(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps(_payload_without_language()), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required language"):
        read_run(path)


def test_read_run_with_no_metadata_reports_it_as_an_invalid_run(tmp_path: Path) -> None:
    payload = make_result(language="en").to_dict()
    del payload["metadata"]
    path = tmp_path / "bare.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(f"{path} is not a valid run result")):
        read_run(path, legacy_language="en")


def test_read_run_decodes_the_file_as_utf8(tmp_path: Path, text_io_calls) -> None:
    path = write_run(make_result(model_id="m", language="en"), tmp_path)
    text_io_calls.clear()

    read_run(path)

    assert text_io_calls == [("read", "m.json", {"encoding": "utf-8"})]


def test_read_runs_gives_archive_results_the_english_legacy_default(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "old.json").write_text(json.dumps(_payload_without_language()), encoding="utf-8")

    (restored,) = read_runs(tmp_path, recursive=True)

    assert restored.metadata.language == "en"


def test_read_runs_does_not_invent_a_language_for_a_current_result(tmp_path: Path) -> None:
    (tmp_path / "fr").mkdir()
    (tmp_path / "fr" / "m.json").write_text(
        json.dumps(_payload_without_language()), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing required language"):
        read_runs(tmp_path)


def test_read_runs_requires_a_nested_result_to_sit_in_its_language_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "fr").mkdir()
    (tmp_path / "fr" / "m.json").write_text(
        dumps_run_payload(make_result(model_id="m", language="en").to_dict()), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=re.escape("expected a language directory")):
        read_runs(tmp_path)


def test_run_filename_rejects_a_language_outside_the_two_or_three_letter_form() -> None:
    with pytest.raises(ValueError, match=re.escape("invalid result language 'english'")):
        run_filename("m", "english")


def test_csv_writers_create_missing_parent_directories(tmp_path: Path) -> None:
    leaderboard = write_leaderboard_csv(
        [make_result(model_id="m")], tmp_path / "0" / "1" / "leaderboard.csv"
    )
    aggregates = write_aggregates_csv(
        [make_result(model_id="m")], tmp_path / "a" / "b" / "aggregates.csv"
    )
    sweep = write_threshold_sweep_csv([_scoring_run()], tmp_path / "c" / "d" / "sweep.csv")
    summary = write_scoring_summary_csv([_scoring_run()], tmp_path / "e" / "f" / "summary.csv")

    assert leaderboard.is_file()
    assert aggregates.is_file()
    assert sweep.is_file()
    assert summary.is_file()


def test_csv_writers_end_every_row_with_a_bare_newline(tmp_path: Path) -> None:
    raw = write_threshold_sweep_csv([_scoring_run()], tmp_path / "sweep.csv").read_bytes()

    assert b"\r" not in raw
    assert raw.endswith(b"\n")


def test_threshold_sweep_csv_writes_one_row_per_boundary(tmp_path: Path) -> None:
    path = write_threshold_sweep_csv([_scoring_run()], tmp_path / "sweep.csv")

    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == ",".join(THRESHOLD_SWEEP_COLUMNS)
    assert len(lines) == 1 + len(DEFAULT_THRESHOLDS)
    assert lines[1].startswith("m,0.0,1,2,")


def test_threshold_sweep_csv_is_only_a_header_when_nothing_scores(tmp_path: Path) -> None:
    path = write_threshold_sweep_csv([make_result(model_id="m")], tmp_path / "sweep.csv")

    assert path.read_text(encoding="utf-8") == ",".join(THRESHOLD_SWEEP_COLUMNS) + "\n"


def test_scoring_summary_csv_holds_the_best_threshold_row_per_scoring_model(
    tmp_path: Path,
) -> None:
    path = write_scoring_summary_csv([_scoring_run()], tmp_path / "summary.csv")

    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == ",".join(SCORING_SUMMARY_COLUMNS)
    # Both items are right for every boundary in (0.2, 0.9]; ties take the lowest boundary,
    # and recall is perfect from the lowest boundary because the YES item is always called.
    assert lines[1] == "m,1,2,1.0,0.3,1.0,0.3,1.0,0.3,1.0,0.3,1.0,0.0,1.0,1.23,100"


def test_write_reports_writes_the_four_report_files_beside_the_leaderboard(
    tmp_path: Path,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"

    written = write_reports([_scoring_run()], leaderboard)

    assert written["leaderboard"] == leaderboard
    assert written["aggregates"] == tmp_path / "aggregates.csv"
    assert written["threshold_sweep"] == tmp_path / "threshold_sweep.csv"
    assert written["scoring_summary"] == tmp_path / "scoring_summary.csv"
    summary = (tmp_path / "scoring_summary.csv").read_text(encoding="utf-8").splitlines()
    assert summary[0] == ",".join(SCORING_SUMMARY_COLUMNS)
    assert summary[1].startswith("m,1,2,1.0,0.3,")


def test_report_csvs_pin_their_encoding_and_newline_arguments(tmp_path: Path, opened) -> None:
    write_reports([_scoring_run()], tmp_path / "leaderboard.csv")

    for name in ("leaderboard.csv", "aggregates.csv", "threshold_sweep.csv"):
        (call,) = [call for call in opened[name] if "newline" in call]
        assert call == {"newline": "", "encoding": "utf-8"}, name
