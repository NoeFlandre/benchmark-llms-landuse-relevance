import pytest

from factories import ScriptedGenerator
from landuse_relevance_bench.domain.dataset import build_item
from landuse_relevance_bench.domain.engine import Generation
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.orchestration import predict_all

TEMPLATE = "SENTENCE: {}"
ITEMS = (
    build_item({"sentence": "Dense forest covers the ridge.", "label": "yes"}),
    build_item({"sentence": "He was elected in 1974.", "label": "no"}),
)


def test_renders_one_prompt_per_item_in_order() -> None:
    generator = ScriptedGenerator(["yes", "no"])
    predict_all(ITEMS, TEMPLATE, generator)
    assert generator.prompts == [
        "SENTENCE: Dense forest covers the ridge.",
        "SENTENCE: He was elected in 1974.",
    ]


def test_pairs_each_raw_output_with_its_item_and_parses_it() -> None:
    predictions = predict_all(ITEMS, TEMPLATE, ScriptedGenerator(["Answer: yes", "no."]))
    assert [p.item_id for p in predictions] == [i.item_id for i in ITEMS]
    assert [p.expected for p in predictions] == [Label.YES, Label.NO]
    assert [p.predicted for p in predictions] == [Label.YES, Label.NO]
    assert [p.raw_output for p in predictions] == ["Answer: yes", "no."]


def test_keeps_unparsable_output_as_a_null_prediction() -> None:
    (prediction,) = predict_all(ITEMS[:1], TEMPLATE, ScriptedGenerator(["I am not sure"]))
    assert prediction.predicted is None
    assert prediction.raw_output == "I am not sure"


def test_batches_items_at_the_requested_size() -> None:
    generator = ScriptedGenerator(["yes", "no"])
    predict_all(ITEMS, TEMPLATE, generator, batch_size=1)
    assert generator.batch_sizes == [1, 1]


def test_rejects_a_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match=r"^batch_size must be at least 1, got 0$"):
        predict_all(ITEMS, TEMPLATE, ScriptedGenerator([]), batch_size=0)


def test_rejects_a_generator_that_returns_the_wrong_number_of_outputs() -> None:
    with pytest.raises(ValueError, match=r"^generator returned 1 outputs for 2 prompts$"):
        predict_all(ITEMS, TEMPLATE, ScriptedGenerator(["yes"]))


def test_an_empty_benchmark_produces_no_predictions() -> None:
    assert predict_all((), TEMPLATE, ScriptedGenerator([])) == ()


def test_a_truncated_generation_is_never_credited_with_a_verdict() -> None:
    """Hitting the token budget mid-sentence means the model never answered."""
    generator = ScriptedGenerator([Generation('Criteria for "yes": terrain', truncated=True)])
    (prediction,) = predict_all(ITEMS[:1], TEMPLATE, generator)
    assert prediction.predicted is None
    assert prediction.truncated is True


def test_a_finished_generation_is_parsed_normally() -> None:
    (prediction,) = predict_all(ITEMS[:1], TEMPLATE, ScriptedGenerator([Generation("yes")]))
    assert prediction.predicted is Label.YES
    assert prediction.truncated is False


def test_a_plain_string_from_a_generator_is_treated_as_finished() -> None:
    (prediction,) = predict_all(ITEMS[:1], TEMPLATE, ScriptedGenerator(["no"]))
    assert prediction.predicted is Label.NO
    assert prediction.truncated is False


def test_each_prediction_carries_its_batch_latency_and_token_counts() -> None:
    ticks = iter([10.0, 10.5, 20.0, 20.25])
    generator = ScriptedGenerator(
        [
            Generation("yes", generated_tokens=3, verify_steps=2),
            Generation("no", generated_tokens=4, accepted_drafts=5, proposed_drafts=8),
        ]
    )
    first, second = predict_all(ITEMS, TEMPLATE, generator, batch_size=1, clock=lambda: next(ticks))
    assert (first.latency_seconds, first.generated_tokens, first.verify_steps) == (0.5, 3, 2)
    assert (second.latency_seconds, second.accepted_drafts, second.proposed_drafts) == (0.25, 5, 8)


def test_per_generation_latency_overrides_the_whole_batch_wall_time() -> None:
    generator = ScriptedGenerator([Generation("yes", latency_seconds=0.125)])
    (prediction,) = predict_all(ITEMS[:1], TEMPLATE, generator, clock=lambda: 100.0)
    assert prediction.latency_seconds == 0.125
