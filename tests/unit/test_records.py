import pytest

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult

META = RunMetadata(
    model_id="LiquidAI/LFM2.5-350M",
    language="en",
    model_revision="abc123",
    prompt_sha256="p" * 64,
    benchmark_sha256="b" * 64,
    max_new_tokens=8,
    batch_size=16,
    seed=0,
    decoding="greedy",
    dtype="bfloat16",
    started_at="2026-09-13T10:00:00Z",
    duration_seconds=12.5,
    source_commit="deadbeef",
)


def _prediction(predicted: Label | None = Label.YES) -> Prediction:
    return Prediction(item_id="0" * 16, expected=Label.YES, predicted=predicted, raw_output="yes")


def test_prediction_round_trips_through_a_plain_dict() -> None:
    prediction = _prediction()
    assert Prediction.from_dict(prediction.to_dict()) == prediction


def test_the_serialised_keys_are_the_published_schema() -> None:
    """Renaming one of these silently invalidates every result file already published."""
    assert _prediction().to_dict() == {
        "item_id": "0" * 16,
        "expected": "yes",
        "predicted": "yes",
        "raw_output": "yes",
        "truncated": False,
    }


def test_a_truncated_prediction_reads_back_as_truncated() -> None:
    prediction = Prediction(
        item_id="0" * 16,
        expected=Label.YES,
        predicted=None,
        raw_output="1. Analyze",
        truncated=True,
    )
    assert prediction.to_dict()["truncated"] is True
    assert Prediction.from_dict(prediction.to_dict()).truncated is True


def test_a_result_file_written_before_truncation_was_tracked_still_loads() -> None:
    legacy = {"item_id": "0" * 16, "expected": "yes", "predicted": "no", "raw_output": "no"}
    assert Prediction.from_dict(legacy).truncated is False


def test_a_null_prediction_round_trips() -> None:
    prediction = _prediction(None)
    assert prediction.to_dict()["predicted"] is None
    assert Prediction.from_dict(prediction.to_dict()) == prediction


def test_run_result_round_trips_through_a_plain_dict() -> None:
    result = RunResult(
        metadata=META,
        predictions=(_prediction(),),
        metrics=evaluate([(Label.YES, Label.YES)]),
    )
    assert RunResult.from_dict(result.to_dict()) == result


def test_run_result_dict_is_json_serialisable() -> None:
    import json

    result = RunResult(
        metadata=META, predictions=(_prediction(),), metrics=evaluate([(Label.YES, Label.YES)])
    )
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


def test_run_result_rejects_metrics_that_disagree_with_its_predictions() -> None:
    with pytest.raises(ValueError, match="metrics cover 1 items but the run holds 2"):
        RunResult(
            metadata=META,
            predictions=(_prediction(), _prediction()),
            metrics=evaluate([(Label.YES, Label.YES)]),
        )


def test_missing_language_is_rejected_as_archive_only_metadata() -> None:
    payload = META.to_dict()
    del payload["language"]

    with pytest.raises(ValueError, match=r"archive-only.*language"):
        RunMetadata.from_dict(payload)
