"""Model-specific relevance scoring rules stay explicit and testable."""

from math import isclose

from landuse_relevance_bench.adapters.hf_scorer import (
    GteScorer,
    MxbaiRerankerScorer,
    mxbai_normalize,
    scorer_class_for,
)


def test_model_ids_select_their_intended_transformers_adapter() -> None:
    assert scorer_class_for("Alibaba-NLP/gte-multilingual-reranker-base") is GteScorer
    assert scorer_class_for("mixedbread-ai/mxbai-rerank-base-v2") is MxbaiRerankerScorer


def test_mxbai_native_logit_difference_uses_the_published_normalisation() -> None:
    assert isclose(mxbai_normalize(4.5), 0.5)
    assert mxbai_normalize(9.0) > 0.98
    assert mxbai_normalize(0.0) < 0.02
