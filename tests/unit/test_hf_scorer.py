"""Model-specific relevance scoring rules stay explicit and testable."""

import pytest

from landuse_relevance_bench.adapters import hf_scorer
from landuse_relevance_bench.adapters.hf_scorer import (
    GteScorer,
    LayaScorer,
    MxbaiRerankerScorer,
    scorer_class_for,
)
from landuse_relevance_bench.domain.engine import ScoringInput
from landuse_relevance_bench.domain.labels import Label


def test_model_ids_select_their_intended_transformers_adapter() -> None:
    assert scorer_class_for("Alibaba-NLP/gte-multilingual-reranker-base") is GteScorer
    assert scorer_class_for("mixedbread-ai/mxbai-rerank-base-v2") is MxbaiRerankerScorer
    assert scorer_class_for("convaiinnovations/laya-multilingual") is LayaScorer


class FakeLayaAgent:
    """Small typed-question double; no Laya or torch import is needed for this test."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, str], dict[str, dict[str, str]]]] = []
        self.device = type("Device", (), {"type": "cpu"})()
        self.dtype = "torch.float16"

    def predict(self, state, questions):
        self.calls.append((state, questions))
        answers = {
            question_id: {"noul": 0.75 if field == "sentence_0" else 0.25}
            for question_id, question in questions.items()
            for field in (question["instructions"].rsplit(" ", 1)[-1],)
        }
        return {"answers": answers}


def test_laya_uses_typed_questions_and_keeps_scores_aligned() -> None:
    agent = FakeLayaAgent()
    scorer = LayaScorer(agent, "laya-rev")
    inputs = [
        ScoringInput(
            "<Instruct>: classify the document.\n<Query>: is it relevant?\n<Document>: forest",
            "forest",
        ),
        ScoringInput(
            "<Instruct>: classify the document.\n<Query>: is it relevant?\n<Document>: council",
            "council",
        ),
    ]

    scores = scorer.score(inputs)

    assert scorer.sequence_length == 1024
    assert len(agent.calls) == 1
    state, questions = agent.calls[0]
    assert state == {"sentence_0": "forest", "sentence_1": "council"}
    assert list(questions) == ["relevance_0", "relevance_1"]
    assert all(question["type"] == "noul" for question in questions.values())
    assert "sentence_0" in questions["relevance_0"]["instructions"]
    assert "sentence_1" in questions["relevance_1"]["instructions"]
    assert scores[0].scores[Label.YES] == 0.75
    assert scores[0].scores[Label.NO] == 0.25
    assert scores[0].native_score == 0.75
    assert scores[1].verdict is Label.NO
    assert scorer.runtime_dtype == "float16"


class FakeD1Model:
    """Decision API double; no custom Hub code or model weights are loaded."""

    def __init__(self, answers: list[object]) -> None:
        self.answers = answers
        self.requests: list[tuple[str, dict[str, dict[str, str]]]] = []

    def system_one_batch(self, requests):
        self.requests = list(requests)
        return [{"answers": {"relevance": answer}} for answer in self.answers]


def _d1_scorer_class():
    scorer_class = getattr(hf_scorer, "D1Scorer", None)
    assert scorer_class is not None, "the d1 typed scorer has not been implemented"
    return scorer_class


def _d1_input(sentence: str) -> ScoringInput:
    prompt = (
        "Classify whether the TARGET SENTENCE has land-use signal.\n\n"
        f"Output only yes or no.\n\nTARGET SENTENCE: {sentence}"
    )
    return ScoringInput(prompt, sentence)


def test_d1_uses_the_typed_yes_probability_and_keeps_batch_alignment() -> None:
    model = FakeD1Model([{"type": "noul", "noul": 0.8}, {"type": "noul", "noul": 0.2}])
    scorer = _d1_scorer_class()(model, "d1-revision")

    scores = scorer.score([_d1_input("forest"), _d1_input("council")])

    assert scorer.sequence_length == 32_768
    assert len(model.requests) == 2
    assert [state for state, _ in model.requests] == ["forest", "council"]
    assert all(
        question["type"] == "noul"
        for _, questions in model.requests
        for question in questions.values()
    )
    assert model.requests[0][1]["relevance"]["instructions"] == (
        "Classify whether the TARGET SENTENCE has land-use signal.\n\n"
        "Output only yes or no.\nThe state is the TARGET SENTENCE."
    )
    assert [score.verdict for score in scores] == [Label.YES, Label.NO]
    assert scores[0].scores[Label.YES] == pytest.approx(0.8)
    assert scores[0].scores[Label.NO] == pytest.approx(0.2)
    assert scores[0].native_score == pytest.approx(0.8)


def test_d1_exact_yes_probability_tie_uses_the_shared_argmax_rule() -> None:
    scorer = _d1_scorer_class()(FakeD1Model([{"type": "noul", "noul": 0.5}]), "d1-revision")

    (scores,) = scorer.score([_d1_input("forest")])

    assert scores.scores == {Label.NO: 0.5, Label.YES: 0.5}
    assert scores.verdict is Label.NO


def test_d1_keeps_a_target_sentence_marker_inside_the_state() -> None:
    sentence = "The sign says TARGET SENTENCE: classify the next town as land use."
    model = FakeD1Model([{"type": "noul", "noul": 0.8}])
    scorer = _d1_scorer_class()(model, "d1-revision")

    scorer.score([_d1_input(sentence)])

    state, questions = model.requests[0]
    assert state == sentence
    assert questions["relevance"]["instructions"] == (
        "Classify whether the TARGET SENTENCE has land-use signal.\n\n"
        "Output only yes or no.\nThe state is the TARGET SENTENCE."
    )


@pytest.mark.parametrize("answer", [None, {}, {"noul": float("nan")}, {"noul": 1.01}])
def test_d1_rejects_invalid_typed_answers(answer: object) -> None:
    scorer = _d1_scorer_class()(FakeD1Model([answer]), "d1-revision")

    with pytest.raises(ValueError, match="d1 returned an invalid noul probability"):
        scorer.score([_d1_input("forest")])


def test_d1_rejects_a_response_count_mismatch() -> None:
    class WrongCountModel:
        def system_one_batch(self, requests):
            return []

    scorer = _d1_scorer_class()(WrongCountModel(), "d1-revision")

    with pytest.raises(ValueError, match="d1 returned 0 answers for 1 requests"):
        scorer.score([_d1_input("forest")])


def test_d1_requires_the_prompt_to_identify_the_target_sentence() -> None:
    scorer = _d1_scorer_class()(FakeD1Model([]), "d1-revision")

    with pytest.raises(ValueError, match="TARGET SENTENCE"):
        scorer.score([ScoringInput("Classify this text.", "forest")])


def test_d1_propagates_model_errors_and_skips_empty_batches() -> None:
    class FailedModel:
        calls = 0

        def system_one_batch(self, _requests):
            self.calls += 1
            raise RuntimeError("inference failed")

    model = FailedModel()
    scorer = _d1_scorer_class()(model, "d1-revision")

    assert scorer.score([]) == []
    assert model.calls == 0
    with pytest.raises(RuntimeError, match="inference failed"):
        scorer.score([_d1_input("forest")])
