"""Turning a non-generative model's native scores into benchmark predictions."""

import pytest

from landuse_relevance_bench.domain.dataset import BenchmarkItem
from landuse_relevance_bench.domain.engine import LabelScores, ScoringInput
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.orchestration import score_all
from landuse_relevance_bench.domain.scorers import SCORER_ROSTER, scorer_for, scorer_ids


class _Scorer:
    def __init__(self, *scores: dict[Label, float]) -> None:
        self._scores = list(scores)
        self.batches: list[int] = []

    def score(self, prompts):
        self.batches.append(len(prompts))
        taken, self._scores = self._scores[: len(prompts)], self._scores[len(prompts) :]
        return [LabelScores(s) for s in taken]


def _items(count: int) -> list[BenchmarkItem]:
    return [
        BenchmarkItem(
            item_id=f"{i:016d}",
            source_item_id=f"src{i}",
            language="en",
            sentence=f"sentence {i}",
            label=Label.YES,
        )
        for i in range(count)
    ]


def test_the_verdict_is_the_highest_scoring_label() -> None:
    assert LabelScores({Label.YES: 0.9, Label.NO: 0.1}).verdict is Label.YES
    assert LabelScores({Label.YES: 0.2, Label.NO: 0.8}).verdict is Label.NO


def test_a_tie_resolves_the_same_way_every_time() -> None:
    tie = LabelScores({Label.YES: 0.5, Label.NO: 0.5})

    assert tie.verdict is LabelScores({Label.NO: 0.5, Label.YES: 0.5}).verdict is tie.verdict


def test_an_item_with_no_scores_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one label score"):
        LabelScores({})


def test_scoring_keeps_the_native_scores_so_another_rule_can_be_recomputed() -> None:
    scorer = _Scorer({Label.YES: 0.73, Label.NO: 0.27})

    (prediction,) = score_all(_items(1), "{}", scorer, batch_size=4)

    assert prediction.raw_output == "no=0.270000 yes=0.730000"
    assert prediction.predicted is Label.YES


def test_scoring_keeps_a_model_native_relevance_score() -> None:
    (prediction,) = score_all(_items(1), "{}", _NativeScoreScorer(1.25), batch_size=4)

    assert prediction.raw_output == "no=0.270000 yes=0.730000 native=1.250000"


def test_scoring_passes_both_rendered_prompt_and_original_sentence() -> None:
    scorer = _InputCaptureScorer()

    score_all(_items(1), "Question: {}", scorer)

    assert scorer.inputs == [ScoringInput(prompt="Question: sentence 0", sentence="sentence 0")]


def test_a_scored_prediction_is_never_truncated_or_unparsed() -> None:
    scorer = _Scorer({Label.YES: 0.1, Label.NO: 0.9}, {Label.YES: 0.6, Label.NO: 0.4})

    predictions = score_all(_items(2), "{}", scorer, batch_size=2)

    assert [p.truncated for p in predictions] == [False, False]
    assert all(p.predicted is not None for p in predictions)


def test_scoring_refuses_a_scorer_that_drops_items() -> None:
    with pytest.raises(ValueError, match="scorer returned 1 scores for 2 prompts"):
        score_all(_items(2), "{}", _Scorer({Label.YES: 1.0}), batch_size=2)


def test_the_scoring_roster_is_separate_from_the_generative_one() -> None:
    from landuse_relevance_bench.domain.roster import model_ids

    assert set(scorer_ids()).isdisjoint(model_ids())
    assert all(spec.decision_rule for spec in SCORER_ROSTER)
    assert scorer_for("Qwen/Qwen3-Reranker-0.6B").kind == "reranker"
    assert scorer_for("Alibaba-NLP/gte-multilingual-reranker-base").kind == "sequence-classifier"
    assert scorer_for("mixedbread-ai/mxbai-rerank-base-v2").kind == "reranker"


class _NativeScoreScorer:
    def __init__(self, native_score: float) -> None:
        self._native_score = native_score

    def score(self, inputs):
        return [
            LabelScores({Label.YES: 0.73, Label.NO: 0.27}, native_score=self._native_score)
            for _ in inputs
        ]


class _InputCaptureScorer:
    def __init__(self) -> None:
        self.inputs = []

    def score(self, inputs):
        self.inputs.extend(inputs)
        return [LabelScores({Label.YES: 1.0, Label.NO: 0.0}) for _ in inputs]
