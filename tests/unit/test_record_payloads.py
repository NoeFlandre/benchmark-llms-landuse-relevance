"""Round trips and invariants of the run records' serialised payloads."""

from dataclasses import replace

import pytest

from factories import make_result
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.records import Prediction, RunResult


def test_a_prediction_with_every_optional_field_round_trips() -> None:
    prediction = Prediction(
        item_id="a" * 16,
        expected=Label.YES,
        predicted=Label.NO,
        raw_output="no",
        truncated=True,
        latency_seconds=0.25,
        generated_tokens=4,
        verify_steps=2,
        accepted_drafts=1,
        proposed_drafts=3,
    )

    assert Prediction.from_dict(prediction.to_dict()) == prediction


def test_a_scoring_prediction_serialises_label_scores_without_rounding() -> None:
    import json

    yes_score = 0.5000001
    prediction = Prediction(
        item_id="b" * 16,
        expected=Label.YES,
        predicted=Label.YES,
        raw_output="no=0.500000 yes=0.500000",
        label_scores={Label.NO: 1.0 - yes_score, Label.YES: yes_score},
    )

    payload = prediction.to_dict()

    assert payload["label_scores"] == {"no": 1.0 - yes_score, "yes": yes_score}
    restored = Prediction.from_dict(json.loads(json.dumps(payload)))
    assert restored.label_scores == prediction.label_scores


def test_a_prediction_payload_names_each_optional_field() -> None:
    prediction = Prediction(
        item_id="a" * 16,
        expected=Label.YES,
        predicted=None,
        raw_output="",
        generated_tokens=4,
        verify_steps=2,
        accepted_drafts=1,
        proposed_drafts=3,
    )

    payload = prediction.to_dict()

    assert payload["predicted"] is None
    assert payload["generated_tokens"] == 4
    assert payload["verify_steps"] == 2
    assert payload["accepted_drafts"] == 1
    assert payload["proposed_drafts"] == 3
    assert "label_scores" not in payload


def test_a_run_result_payload_has_exactly_its_four_sections() -> None:
    payload = make_result(model_id="m").to_dict()

    assert set(payload) == {"metadata", "metrics", "speed", "predictions"}


def test_a_run_result_round_trips_with_its_optional_prediction_fields() -> None:
    predictions = (
        Prediction(
            item_id="0" * 16,
            expected=Label.YES,
            predicted=Label.YES,
            raw_output="yes",
            generated_tokens=5,
            verify_steps=2,
            accepted_drafts=1,
            proposed_drafts=2,
        ),
    )
    result = make_result(predictions=predictions)

    restored = RunResult.from_dict(result.to_dict())

    assert restored.predictions == predictions


def test_a_run_result_refuses_predictions_that_repeat_an_item() -> None:
    prediction = Prediction(
        item_id="x" * 16, expected=Label.YES, predicted=Label.YES, raw_output=""
    )
    predictions = (prediction, prediction)

    with pytest.raises(ValueError, match=r"^run contains a duplicate item id$"):
        RunResult(
            metadata=make_result().metadata,
            predictions=predictions,
            metrics=evaluate([p.outcome for p in predictions]),
        )


def test_a_run_result_refuses_metrics_that_do_not_come_from_its_predictions() -> None:
    result = make_result(
        predictions=(
            Prediction(item_id="0" * 16, expected=Label.YES, predicted=Label.YES, raw_output=""),
            Prediction(item_id="1" * 16, expected=Label.NO, predicted=Label.NO, raw_output=""),
        )
    )
    other = evaluate([(Label.YES, Label.NO), (Label.NO, Label.YES)])

    with pytest.raises(ValueError, match=r"^metrics do not match predictions$"):
        replace(result, metrics=other)
