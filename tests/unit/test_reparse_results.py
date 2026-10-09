import json
from pathlib import Path

from factories import make_result
from landuse_relevance_bench.adapters.hashing import sha256_of_text
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import read_run, run_filename, write_run
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction
from scripts.reparse_results import reparse_directory


def test_reparse_migrates_legacy_labels_and_rebuilds_reports(tmp_path: Path) -> None:
    prompt_text = load_prompt(Path("data/prompt.txt"))
    old_predictions = (
        Prediction(
            item_id="0" * 16,
            expected=Label.YES,
            predicted=Label.NO,
            raw_output="Yes, the prompt mentions no, but answer yes.",
        ),
        Prediction(item_id="1" * 16, expected=Label.NO, predicted=Label.NO, raw_output="no"),
    )
    result = make_result(
        "test/parser",
        old_predictions,
        prompt_sha256=sha256_of_text(prompt_text),
    )
    result_path = tmp_path / run_filename(result.metadata.name, result.metadata.language)
    result_path.parent.mkdir(parents=True)
    payload = result.to_dict()
    payload["predictions"][0].pop("parse_mode")
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    changes = reparse_directory(tmp_path)
    updated = read_run(result_path)

    assert changes[result_path.relative_to(tmp_path).as_posix()] == (1, 2)
    assert [p.predicted for p in updated.predictions] == [Label.YES, Label.NO]
    assert [p.parse_mode for p in updated.predictions] == ["leading", "exact"]
    assert updated.metrics == evaluate([p.outcome for p in updated.predictions])
    assert (tmp_path / "leaderboard.csv").is_file()
    assert (tmp_path / "README.md").is_file()


def test_reparse_keeps_truncated_output_unparsed(tmp_path: Path) -> None:
    prompt_text = load_prompt(Path("data/prompt.txt"))
    prediction = Prediction(
        item_id="1" * 16,
        expected=Label.YES,
        predicted=None,
        raw_output="yes",
        truncated=True,
    )
    result = make_result(
        "test/truncated",
        (prediction,),
        prompt_sha256=sha256_of_text(prompt_text),
    )
    path = tmp_path / run_filename(result.metadata.name, result.metadata.language)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    reparse_directory(tmp_path)

    updated = read_run(path)
    assert updated.predictions[0].predicted is None
    assert updated.predictions[0].parse_mode is None


def test_reparse_leaves_a_canonical_run_file_byte_identical(tmp_path: Path) -> None:
    prompt_text = load_prompt(Path("data/prompt.txt"))
    prediction = Prediction(
        item_id="0" * 16,
        expected=Label.NO,
        predicted=Label.NO,
        raw_output="no",
        parse_mode="exact",
    )
    result = make_result(
        "test/canonical",
        (prediction,),
        prompt_sha256=sha256_of_text(prompt_text),
    )
    path = write_run(result, tmp_path)
    before = path.read_bytes()

    reparse_directory(tmp_path)

    assert path.read_bytes() == before
