"""Model-specific relevance scoring rules stay explicit and testable."""

from math import isclose

from landuse_relevance_bench.adapters.hf_scorer import (
    GteScorer,
    LayaScorer,
    MxbaiRerankerScorer,
    ScorerSettings,
    mxbai_normalize,
    scorer_class_for,
)
from landuse_relevance_bench.domain.engine import ScoringInput
from landuse_relevance_bench.domain.labels import Label


def test_model_ids_select_their_intended_transformers_adapter() -> None:
    assert scorer_class_for("Alibaba-NLP/gte-multilingual-reranker-base") is GteScorer
    assert scorer_class_for("mixedbread-ai/mxbai-rerank-base-v2") is MxbaiRerankerScorer
    assert scorer_class_for("convaiinnovations/laya-multilingual") is LayaScorer


def test_mxbai_native_logit_difference_uses_the_published_normalisation() -> None:
    assert isclose(mxbai_normalize(4.5), 0.5)
    assert mxbai_normalize(9.0) > 0.98
    assert mxbai_normalize(0.0) < 0.02


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
    scorer = LayaScorer(agent, ScorerSettings(dtype="bfloat16"), "laya-rev")
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
