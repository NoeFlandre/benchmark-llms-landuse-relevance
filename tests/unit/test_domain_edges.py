"""Edge cases the mutation gate showed were untested: messages, boundaries, defaults."""

import re
from collections.abc import Sequence

import pytest

from landuse_relevance_bench.domain.dataset import BenchmarkItem
from landuse_relevance_bench.domain.engine import LabelScores, ScoringInput
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.orchestration import format_scores, score_all
from landuse_relevance_bench.domain.records import Prediction, RunMetadata
from landuse_relevance_bench.domain.roster import quantization_of
from landuse_relevance_bench.domain.scorers import scorer_for
from landuse_relevance_bench.domain.sharding import model_language_pairs, shard_pairs
from landuse_relevance_bench.domain.thresholds import (
    ScoreFormatError,
    decide_scores,
    parse_scores,
    roc_auc_scores,
    yes_scores,
)
from landuse_relevance_bench.domain.variants import repository_of


def _exact(message: str) -> str:
    return f"^{re.escape(message)}$"


def _metadata(**changes) -> dict:
    return {
        "model_id": "m/x",
        "language": "en",
        "model_revision": "r",
        "prompt_sha256": "p",
        "benchmark_sha256": "b",
        "max_new_tokens": 1,
        "batch_size": 1,
        "seed": 0,
        "decoding": "greedy",
        "dtype": "bfloat16",
        "started_at": "t",
        "duration_seconds": 1.0,
        **changes,
    }


class _EchoScorer:
    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        return [LabelScores({Label.YES: 0.8, Label.NO: 0.2}) for _ in inputs]


def _items(count: int) -> list[BenchmarkItem]:
    return [
        BenchmarkItem(
            item_id=f"{index:016x}",
            source_item_id=str(index),
            language="en",
            sentence=f"sentence {index}",
            label=Label.YES,
        )
        for index in range(count)
    ]


def test_a_label_score_needs_at_least_one_label() -> None:
    with pytest.raises(ValueError, match=_exact("a scored item needs at least one label score")):
        LabelScores({})


def test_scoring_accepts_a_batch_of_one_and_keeps_item_ids() -> None:
    predictions = score_all(_items(2), "{}", _EchoScorer(), batch_size=1)

    assert [p.item_id for p in predictions] == ["0" * 16, "0" * 15 + "1"]
    assert all(p.truncated is False for p in predictions)


def test_scores_are_rendered_in_label_order_whatever_the_insertion_order() -> None:
    rendered = format_scores(LabelScores({Label.YES: 0.25, Label.NO: 0.75}))

    assert rendered == "no=0.750000 yes=0.250000"


def test_blank_language_metadata_is_refused_with_its_reason() -> None:
    message = "run metadata requires a non-empty language; legacy records are archive-only"
    with pytest.raises(ValueError, match=_exact(message)):
        RunMetadata(**_metadata(language="   "))


def test_metadata_without_a_language_or_with_unknown_fields_is_refused() -> None:
    missing = "legacy/archive-only run metadata is missing required language"
    with pytest.raises(ValueError, match=_exact(missing)):
        RunMetadata.from_dict({k: v for k, v in _metadata().items() if k != "language"})
    with pytest.raises(ValueError, match=r"^invalid run metadata: .*unexpected"):
        RunMetadata.from_dict(_metadata(unexpected=1))


def test_an_unrostered_model_has_no_quantization() -> None:
    assert quantization_of("not/rostered") == ""


def test_an_unknown_scorer_is_named_in_the_error() -> None:
    with pytest.raises(KeyError, match="'x/y' is not in the scoring roster"):
        scorer_for("x/y")


def test_blank_models_and_languages_are_named_when_refused() -> None:
    with pytest.raises(ValueError, match=_exact("model values cannot be empty")):
        model_language_pairs([" "], ["en"])
    with pytest.raises(ValueError, match=_exact("language values cannot be empty")):
        model_language_pairs(["m/x"], [" "])


def test_duplicate_or_half_blank_shard_pairs_are_refused_with_reasons() -> None:
    with pytest.raises(ValueError, match=_exact("shard pair list contains duplicate pairs")):
        shard_pairs([("m", "en"), ("m", "en")], 0, 1)
    blank = "shard pairs require non-empty model and language values"
    with pytest.raises(ValueError, match=_exact(blank)):
        shard_pairs([("m", " ")], 0, 1)
    with pytest.raises(ValueError, match=_exact(blank)):
        shard_pairs([(" ", "en")], 0, 1)


def test_malformed_score_text_is_refused_with_the_offending_part() -> None:
    with pytest.raises(ScoreFormatError, match=_exact("not a label=score pair: 'yes'")):
        parse_scores("yes")
    with pytest.raises(ScoreFormatError, match=_exact("not a label=score pair: 'yes=1=2'")):
        parse_scores("yes=1=2")
    with pytest.raises(ScoreFormatError, match=_exact("unreadable score 'yes=abc'")):
        parse_scores("yes=abc")


def test_a_missing_yes_score_counts_as_zero() -> None:
    prediction = Prediction("0" * 16, Label.NO, Label.NO, raw_output="no=1.000000")

    assert yes_scores([prediction]) == (0.0,)


def test_a_score_exactly_at_the_threshold_is_a_yes() -> None:
    metrics = decide_scores([Label.YES], [0.5], 0.5)

    assert metrics.confusion.true_positive == 1


def test_mismatched_labels_and_scores_are_refused() -> None:
    with pytest.raises(ValueError, match=_exact("cannot re-decide an empty set of predictions")):
        decide_scores([], [], 0.5)
    with pytest.raises(ValueError, match="zip"):
        decide_scores([Label.YES, Label.NO], [0.5], 0.5)
    with pytest.raises(ValueError, match="zip"):
        roc_auc_scores([Label.YES, Label.NO], [0.5])
    with pytest.raises(ValueError, match=_exact("ROC AUC needs at least one item of each class")):
        roc_auc_scores([Label.YES], [0.5])


def test_only_the_first_separator_splits_a_variant_id() -> None:
    assert repository_of("org/repo@variant@extra") == "org/repo"
