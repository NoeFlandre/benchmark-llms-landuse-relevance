import pytest

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunMetadata, RunResult

META = RunMetadata(
    model_id="LiquidAI/LFM2.5-350M",
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


def test_the_serialised_keys_are_the_published_schema() -> None:
    """Renaming one of these silently invalidates every result file already published."""
    assert _prediction().to_dict() == {
        "item_id": "0" * 16,
        "expected": "yes",
        "predicted": "yes",
        "raw_output": "yes",
        "truncated": False,
        "parse_mode": None,
        "latency_seconds": None,
        "generated_tokens": None,
        "verify_steps": None,
        "accepted_drafts": None,
        "proposed_drafts": None,
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


def test_a_result_file_from_before_speed_was_tracked_still_loads_and_reports_wall_time() -> None:
    legacy_meta = {
        k: v
        for k, v in META.to_dict().items()
        if k
        not in {
            "run_id",
            "runtime",
            "draft_model_id",
            "draft_model_revision",
            "speculative",
            "generation_mode",
        }
    }
    legacy = {
        "metadata": legacy_meta,
        "metrics": evaluate([(Label.YES, Label.YES)]).to_dict(),
        "predictions": [
            {"item_id": "0" * 16, "expected": "yes", "predicted": "yes", "raw_output": "yes"}
        ],
    }
    result = RunResult.from_dict(legacy)
    assert result.metadata.name == META.model_id
    assert result.metadata.runtime == "transformers"
    assert result.speed.wall_seconds == 12.5
    assert result.speed.latency_p50_seconds is None
    assert result.speed.generated_tokens is None


def test_run_result_round_trips_through_a_plain_dict() -> None:
    result = RunResult(
        metadata=META,
        predictions=(_prediction(),),
        metrics=evaluate([(Label.YES, Label.YES)]),
    )
    assert RunResult.from_dict(result.to_dict()) == result


def test_run_metadata_records_the_generation_mode_and_reads_old_files() -> None:
    payload = META.to_dict()
    assert payload["generation_mode"] == "static-batched"

    payload.pop("generation_mode")
    assert RunMetadata.from_dict(payload).generation_mode == "static-batched"


def test_run_result_rejects_metrics_that_disagree_with_its_predictions() -> None:
    with pytest.raises(ValueError, match="metrics cover 1 items but the run holds 2"):
        RunResult(
            metadata=META,
            predictions=(_prediction(), _prediction()),
            metrics=evaluate([(Label.YES, Label.YES)]),
        )


def test_run_result_rejects_duplicate_item_ids() -> None:
    prediction = _prediction()
    with pytest.raises(ValueError, match="duplicate item id"):
        RunResult(
            metadata=META,
            predictions=(prediction, prediction),
            metrics=evaluate([prediction.outcome, prediction.outcome]),
        )


def test_run_result_rejects_metrics_that_do_not_match_prediction_values() -> None:
    with pytest.raises(ValueError, match="metrics do not match predictions"):
        RunResult(
            metadata=META,
            predictions=(_prediction(),),
            metrics=evaluate([(Label.YES, Label.NO)]),
        )


def test_prediction_parse_mode_round_trips_and_old_files_default_to_none() -> None:
    parsed = Prediction(
        item_id="0" * 16,
        expected=Label.YES,
        predicted=Label.YES,
        raw_output="yes, the prompt says no",
        parse_mode="leading",
    )
    assert Prediction.from_dict(parsed.to_dict()) == parsed
    assert (
        Prediction.from_dict(
            {"item_id": "0" * 16, "expected": "yes", "predicted": "yes", "raw_output": "yes"}
        ).parse_mode
        is None
    )


def test_a_timed_prediction_round_trips_and_the_run_file_carries_its_speed() -> None:
    timed = Prediction(
        item_id="1" * 16,
        expected=Label.NO,
        predicted=Label.NO,
        raw_output="no",
        truncated=False,
        latency_seconds=0.25,
        generated_tokens=4,
        verify_steps=2,
        accepted_drafts=3,
        proposed_drafts=16,
    )
    assert Prediction.from_dict(timed.to_dict()) == timed
    payload = RunResult(
        metadata=META, predictions=(timed,), metrics=evaluate([(Label.NO, Label.NO)])
    ).to_dict()
    assert payload["speed"]["generated_tokens"] == 4
    assert payload["speed"]["mean_accept_length"] == 2.0
