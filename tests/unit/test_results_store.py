import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from factories import make_result
from landuse_relevance_bench.adapters import results_store
from landuse_relevance_bench.adapters.results_store import (
    describe_agreement,
    has_result,
    leaderboard_rows,
    read_run,
    run_filename,
    score_columns,
    speed_columns,
    write_leaderboard_csv,
    write_run,
)
from landuse_relevance_bench.domain.agreement import Agreement
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunResult


def _result(
    model_id: str = "LiquidAI/LFM2.5-350M",
    accuracy_pair: tuple[Label, Label] = (Label.YES, Label.YES),
) -> RunResult:
    prediction = Prediction(
        item_id="0" * 16, expected=accuracy_pair[0], predicted=accuracy_pair[1], raw_output="yes"
    )
    return make_result(model_id, (prediction,))


def test_a_written_run_reads_back_identically(tmp_path: Path) -> None:
    path = write_run(_result(), tmp_path)
    assert read_run(path) == _result()


def test_the_filename_is_derived_from_the_model_id(tmp_path: Path) -> None:
    path = write_run(_result(), tmp_path)
    assert path.name == "LiquidAI__LFM2.5-350M.json"
    assert path.parent == tmp_path
    assert run_filename("a/b") == "a__b.json"


def test_has_result_only_accepts_a_non_empty_result_file(tmp_path: Path) -> None:
    assert not has_result(tmp_path, "a/model")
    (tmp_path / "a__model.json").write_text("", encoding="utf-8")
    assert not has_result(tmp_path, "a/model")
    (tmp_path / "a__model.json").write_text("{}", encoding="utf-8")
    assert has_result(tmp_path, "a/model")
    (tmp_path / "a__model.json").write_text("x", encoding="utf-8")
    assert has_result(tmp_path, "a/model")


def test_write_run_creates_parents_and_preserves_deterministic_utf8_json(tmp_path: Path) -> None:
    prediction = Prediction(
        item_id="0" * 16,
        expected=Label.YES,
        predicted=Label.YES,
        raw_output="Café 🏞️",
    )
    result = RunResult(
        metadata=_result().metadata,
        predictions=(prediction,),
        metrics=evaluate([prediction.outcome]),
    )
    path = write_run(result, tmp_path / "nested" / "results")

    expected = json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    assert "Café 🏞️" in path.read_text(encoding="utf-8")


def test_write_run_explicitly_selects_utf8(monkeypatch, tmp_path: Path) -> None:
    original = Path.write_text
    encodings = []

    def record_encoding(path, data, *args, **kwargs):
        if path.name.endswith(".json"):
            encodings.append(kwargs.get("encoding"))
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record_encoding)
    write_run(_result(), tmp_path)

    assert len(encodings) == 1
    assert encodings[0].casefold() == "utf-8"


def test_read_run_explicitly_selects_utf8(monkeypatch, tmp_path: Path) -> None:
    path = write_run(_result(), tmp_path)
    original = Path.read_text
    encodings = []

    def record_encoding(source, *args, **kwargs):
        if source == path:
            encodings.append(kwargs.get("encoding"))
        return original(source, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_encoding)
    assert read_run(path) == _result()

    assert encodings == ["utf-8"]


def test_writing_the_same_run_twice_is_idempotent(tmp_path: Path) -> None:
    first = write_run(_result(), tmp_path).read_text(encoding="utf-8")
    second = write_run(_result(), tmp_path).read_text(encoding="utf-8")
    assert first == second


def test_the_leaderboard_ranks_models_by_descending_f1(tmp_path: Path) -> None:
    weak = _result("weak/model", (Label.YES, Label.NO))
    strong = _result("strong/model", (Label.YES, Label.YES))
    rows = leaderboard_rows([weak, strong])
    assert [r["model_id"] for r in rows] == ["strong/model", "weak/model"]


def test_empty_leaderboard_has_no_top_run() -> None:
    assert leaderboard_rows([]) == []


def test_leaderboard_top_run_uses_the_published_four_decimal_f1(monkeypatch) -> None:
    def fake_result(name: str, f1: float):
        metadata = SimpleNamespace(
            name=name,
            duration_seconds=1.0,
            benchmark_sha256=name,
            runtime="transformers",
            generation_mode="static-batched",
            model_revision=None,
        )
        return SimpleNamespace(
            metadata=metadata,
            metrics=SimpleNamespace(f1=f1, n_items=1),
            predictions=(),
        )

    monkeypatch.setattr(
        results_store,
        "score_columns",
        lambda metrics, _predictions: {"f1": round(metrics.f1, 4)},
    )
    monkeypatch.setattr(results_store, "speed_columns", lambda _result: {})
    rows = leaderboard_rows([fake_result("z/model", 0.50004), fake_result("a/model", 0.50001)])

    assert [row["model_id"] for row in rows] == ["a/model", "z/model"]
    assert {row["mcnemar_top_run"] for row in rows} == {"a/model"}


def test_score_columns_keep_four_decimal_precision() -> None:
    predictions = (
        Prediction("a" * 16, Label.YES, Label.YES, "yes"),
        Prediction("b" * 16, Label.YES, Label.NO, "no"),
        Prediction("c" * 16, Label.YES, Label.NO, "no"),
        Prediction("d" * 16, Label.YES, None, "unsure"),
        Prediction("e" * 16, Label.YES, None, "unsure"),
        Prediction("f" * 16, Label.NO, Label.NO, "no"),
    )
    columns = score_columns(
        evaluate([prediction.outcome for prediction in predictions]), predictions
    )

    assert columns["recall"] == 0.3333
    assert columns["unparsed_rate"] == 0.3333


def test_speed_columns_use_the_documented_rounding_for_every_optional_measurement() -> None:
    result = SimpleNamespace(
        speed=SimpleNamespace(
            sentences_per_second=12.34567,
            latency_p50_seconds=1.23456,
            latency_p95_seconds=2.34567,
            generated_tokens=17,
            output_tokens_per_second=3.26,
            mean_accept_length=1.23456,
            draft_accept_rate=0.123456,
        )
    )

    assert speed_columns(result) == {
        "sentences_per_second": 12.346,
        "latency_p50_seconds": 1.2346,
        "latency_p95_seconds": 2.3457,
        "generated_tokens": 17,
        "output_tokens_per_second": 3.3,
        "mean_accept_length": 1.235,
        "draft_accept_rate": 0.1235,
    }


def test_the_leaderboard_identifies_each_run_generation_mode() -> None:
    row = leaderboard_rows([_result("a/model")])[0]
    assert row["generation_mode"] == "static-batched"


def test_the_leaderboard_csv_has_a_header_and_one_row_per_model(tmp_path: Path) -> None:
    path = write_leaderboard_csv([_result("a/b"), _result("c/d")], tmp_path / "leaderboard.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("model_id,")
    assert len(lines) == 3


def test_leaderboard_csv_creates_parents_and_requests_portable_text_options(
    monkeypatch, tmp_path: Path
) -> None:
    original = Path.open
    options = []

    def record_options(path, *args, **kwargs):
        if path.name == "leaderboard.csv" and args == ("w",):
            options.append(kwargs)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_options)
    path = write_leaderboard_csv([_result()], tmp_path / "outer" / "nested" / "leaderboard.csv")

    assert path.is_file()
    assert len(options) == 1
    assert options[0]["newline"] == ""
    assert options[0]["encoding"].casefold() == "utf-8"
    contents = path.read_bytes()
    assert b"\r" not in contents
    assert contents.endswith(b"\n")


def test_agreement_description_preserves_runtime_verdict_and_partial_coverage() -> None:
    agreement = Agreement(
        speculative_run="spec",
        baseline_run="base",
        same_runtime=False,
        n_compared=2,
        n_speculative=3,
        n_baseline=4,
        verdicts_differ=1,
        texts_differ=2,
        complete_coverage=False,
    )

    assert describe_agreement(agreement) == (
        "spec vs base (different runtime): MISMATCH, 1/2 verdicts and "
        "2/2 generations differ; coverage 2/3 speculative, 2/4 baseline"
    )


def test_agreement_description_reports_lossless_complete_coverage() -> None:
    agreement = Agreement(
        speculative_run="spec",
        baseline_run="base",
        same_runtime=True,
        n_compared=2,
        n_speculative=2,
        n_baseline=2,
        verdicts_differ=0,
        texts_differ=0,
        complete_coverage=True,
    )

    assert describe_agreement(agreement) == (
        "spec vs base (same runtime): lossless, 0/2 verdicts and "
        "0/2 generations differ; coverage 2 items"
    )


def test_reading_a_corrupt_run_file_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        read_run(path)


def test_the_leaderboard_separates_truncated_generations_from_other_failures() -> None:
    """Unparsed covers both; only the truncation count says the model was cut off."""
    metadata = _result().metadata
    predictions = (
        Prediction(
            item_id="a" * 16,
            expected=Label.YES,
            predicted=None,
            raw_output="1. Analyze",
            truncated=True,
        ),
        Prediction(
            item_id="b" * 16,
            expected=Label.YES,
            predicted=None,
            raw_output="I cannot say",
            truncated=False,
        ),
    )
    result = RunResult(
        metadata=metadata,
        predictions=predictions,
        metrics=evaluate([(Label.YES, None), (Label.YES, None)]),
    )
    (row,) = leaderboard_rows([result])
    assert row["unparsed_rate"] == 1.0
    assert row["truncated"] == 1


def test_reading_a_run_without_metadata_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "no_metadata.json"
    payload = _result().to_dict()
    del payload["metadata"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"no_metadata\.json"):
        read_run(path)


def test_reading_a_run_with_an_unknown_metadata_key_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "extra_key.json"
    payload = _result().to_dict()
    payload["metadata"]["surprise"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"extra_key\.json"):
        read_run(path)


def test_reading_a_run_whose_metrics_disagree_with_its_predictions_fails(tmp_path: Path) -> None:
    path = tmp_path / "tampered.json"
    payload = _result().to_dict()
    payload["metrics"] = evaluate([(Label.YES, Label.YES), (Label.NO, Label.NO)]).to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_run(path)


def test_reading_a_run_with_duplicate_item_ids_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.json"
    payload = _result().to_dict()
    payload["predictions"].append(payload["predictions"][0])
    payload["metrics"] = evaluate([(Label.YES, Label.YES), (Label.YES, Label.YES)]).to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicates\.json.*duplicate item id"):
        read_run(path)
