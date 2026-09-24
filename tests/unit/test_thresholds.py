"""Re-deciding stored scores at other boundaries, without re-running a model."""

import pytest

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.records import Prediction
from landuse_relevance_bench.domain.thresholds import (
    DEFAULT_THRESHOLDS,
    ScoreFormatError,
    decide_at,
    parse_scores,
    roc_auc,
    decide_scores,
    expected_labels,
    roc_auc_scores,
    yes_scores,
)


def _scored(expected: Label, yes: float) -> Prediction:
    return Prediction(
        item_id=f"{yes:.6f}",
        expected=expected,
        predicted=Label.YES if yes >= 0.5 else Label.NO,
        raw_output=f"no={1 - yes:.6f} yes={yes:.6f}",
    )


def test_scores_round_trip_out_of_a_stored_prediction() -> None:
    assert parse_scores("no=0.270000 yes=0.730000") == {Label.NO: 0.27, Label.YES: 0.73}


def test_parse_scores_ignores_the_optional_native_relevance_score() -> None:
    assert parse_scores("no=0.270000 yes=0.730000 native=1.250000") == {
        Label.NO: 0.27,
        Label.YES: 0.73,
    }


def test_unreadable_scores_are_refused_rather_than_guessed() -> None:
    with pytest.raises(ScoreFormatError):
        parse_scores("no yes=0.5")
    with pytest.raises(ScoreFormatError):
        parse_scores("no=high yes=0.5")


def test_a_native_score_without_label_scores_is_refused() -> None:
    with pytest.raises(ScoreFormatError, match="no scores"):
        parse_scores("native=1.25")


def test_a_lower_boundary_calls_more_items_yes() -> None:
    predictions = [_scored(Label.YES, 0.02), _scored(Label.NO, 0.001)]

    strict = decide_at(predictions, 0.5)
    lenient = decide_at(predictions, 0.01)

    assert strict.confusion.true_positive == 0
    assert lenient.confusion.true_positive == 1


def test_an_empty_set_cannot_be_redecided() -> None:
    with pytest.raises(ValueError, match="empty set of predictions"):
        decide_at([], 0.5)


def test_pre_parsed_scores_decide_exactly_like_the_predictions() -> None:
    predictions = [_scored(Label.YES, 0.02), _scored(Label.NO, 0.001), _scored(Label.YES, 0.7)]
    expected, scores = expected_labels(predictions), yes_scores(predictions)

    for threshold in DEFAULT_THRESHOLDS:
        assert decide_scores(expected, scores, threshold) == decide_at(predictions, threshold)
    assert roc_auc_scores(expected, scores) == roc_auc(predictions)


def test_auc_is_one_when_every_positive_outscores_every_negative() -> None:
    predictions = [_scored(Label.YES, 0.9), _scored(Label.YES, 0.8), _scored(Label.NO, 0.1)]

    assert roc_auc(predictions) == 1.0


def test_auc_is_a_half_when_the_score_carries_no_signal() -> None:
    predictions = [_scored(Label.YES, 0.4), _scored(Label.NO, 0.4)]

    assert roc_auc(predictions) == 0.5


def test_auc_gives_half_credit_to_each_cross_class_tie() -> None:
    predictions = [
        _scored(Label.YES, 0.9),
        _scored(Label.YES, 0.8),
        _scored(Label.NO, 0.8),
        _scored(Label.NO, 0.1),
    ]

    assert roc_auc(predictions) == 0.875


def test_auc_does_not_move_when_every_verdict_is_the_same() -> None:
    # The argmax verdict can be degenerate while the ranking is still informative;
    # that difference is the whole point of reporting AUC beside F1.
    predictions = [_scored(Label.YES, 0.004), _scored(Label.NO, 0.001)]

    assert decide_at(predictions, 0.5).f1 == 0.0
    assert roc_auc(predictions) == 1.0


def test_auc_needs_both_classes_present() -> None:
    with pytest.raises(ValueError, match="one item of each class"):
        roc_auc([_scored(Label.YES, 0.9)])
