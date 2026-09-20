"""Invariants that must hold for any input, not just the cases we thought of."""

from hypothesis import given
from hypothesis import strategies as st

from landuse_relevance_bench.domain.dataset import item_id_for
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.metrics import evaluate
from landuse_relevance_bench.domain.parsing import parse_label
from landuse_relevance_bench.domain.prompting import render_prompt
from landuse_relevance_bench.domain.records import Prediction

text = st.text(max_size=200)
labels = st.sampled_from([Label.YES, Label.NO])
outcomes = st.tuples(labels, st.one_of(labels, st.none()))


@given(template_head=text, sentence=text)
def test_rendering_always_places_the_sentence_in_the_prompt(
    template_head: str, sentence: str
) -> None:
    assert sentence in render_prompt(template_head + "{}", sentence)


@given(head=text.filter(lambda s: "{}" not in s), sentence=text)
def test_rendering_never_loses_the_template_around_the_sentence(head: str, sentence: str) -> None:
    """The first placeholder is the slot; anything around it survives verbatim."""
    assert render_prompt(head + "{}!", sentence) == f"{head}{sentence}!"


@given(raw=text)
def test_parsing_returns_a_label_or_nothing_and_never_raises(raw: str) -> None:
    assert parse_label(raw) in (Label.YES, Label.NO, None)


@given(prefix=text.filter(lambda s: not s or s[-1].isspace()), label=labels)
def test_a_trailing_verdict_is_recovered_when_nothing_earlier_decides(
    prefix: str, label: Label
) -> None:
    if parse_label(prefix) is None:
        assert parse_label(prefix + label.value) is label


@given(sentence=text)
def test_item_ids_are_stable_across_calls(sentence: str) -> None:
    assert item_id_for(sentence, "en") == item_id_for(sentence, "en")


@given(pairs=st.lists(outcomes, min_size=1, max_size=60))
def test_every_rate_stays_within_its_natural_bounds(pairs: list) -> None:
    metrics = evaluate(pairs)
    for value in (
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.balanced_accuracy,
        metrics.unparsed_rate,
    ):
        assert 0.0 <= value <= 1.0
    assert -1.0 <= metrics.matthews_corrcoef <= 1.0


@given(pairs=st.lists(outcomes, min_size=1, max_size=60))
def test_the_confusion_matrix_accounts_for_every_item_exactly_once(pairs: list) -> None:
    metrics = evaluate(pairs)
    assert metrics.confusion.total == len(pairs) == metrics.n_items


@given(expected=st.lists(labels, min_size=1, max_size=40))
def test_predicting_the_gold_label_scores_a_perfect_accuracy(expected: list[Label]) -> None:
    assert evaluate([(label, label) for label in expected]).accuracy == 1.0


@given(expected=st.lists(labels, min_size=1, max_size=40))
def test_failing_to_answer_at_all_scores_zero_accuracy(expected: list[Label]) -> None:
    metrics = evaluate([(label, None) for label in expected])
    assert metrics.accuracy == 0.0
    assert metrics.unparsed_rate == 1.0


@given(
    item_id=st.text(max_size=20), expected=labels, predicted=st.one_of(labels, st.none()), raw=text
)
def test_predictions_round_trip_through_their_dict_form(
    item_id: str, expected: Label, predicted: Label | None, raw: str
) -> None:
    prediction = Prediction(item_id=item_id, expected=expected, predicted=predicted, raw_output=raw)
    assert Prediction.from_dict(prediction.to_dict()) == prediction
