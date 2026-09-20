from collections.abc import Sequence

import pytest

from landuse_relevance_bench.domain.dataset import build_item
from landuse_relevance_bench.domain.engine import Generation
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.orchestration import predict_all

TEMPLATE = "SENTENCE: {}"
ITEMS = (
    build_item(
        {
            "sentence": "Dense forest covers the ridge.",
            "label": "yes",
            "source_item_id": "source-1",
            "language": "en",
        }
    ),
    build_item(
        {
            "sentence": "He was elected in 1974.",
            "label": "no",
            "source_item_id": "source-2",
            "language": "en",
        }
    ),
)


class ScriptedGenerator:
    """Returns queued outputs and records the prompts and batch shapes it saw."""

    def __init__(self, outputs: Sequence[str | Generation]) -> None:
        self._outputs: list[str | Generation] = list(outputs)
        self.seen_prompts: list[str] = []
        self.batch_sizes: list[int] = []

    def generate(self, prompts: Sequence[str]) -> Sequence[str | Generation]:
        self.seen_prompts.extend(prompts)
        self.batch_sizes.append(len(prompts))
        taken, self._outputs = self._outputs[: len(prompts)], self._outputs[len(prompts) :]
        return taken


def test_renders_one_prompt_per_item_in_order() -> None:
    generator = ScriptedGenerator(["yes", "no"])
    predict_all(ITEMS, TEMPLATE, generator)
    assert generator.seen_prompts == [
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
