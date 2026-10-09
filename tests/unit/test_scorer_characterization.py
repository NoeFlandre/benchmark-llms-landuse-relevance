"""Pin the exact verdict pairs of the logit-based scorers before and after refactors."""

from __future__ import annotations

import pytest
from tests.unit.test_hf_scorer_runtime import Model, Tokenizer, _inputs

from landuse_relevance_bench.adapters import hf_scorer
from landuse_relevance_bench.domain.labels import Label

torch = pytest.importorskip("torch")

NATIVE_LOGITS = (-3.5, 0.0, 0.7, 4.5, 9.25)


def _pairs(scores):
    return [(s.scores[Label.NO], s.scores[Label.YES], s.native_score) for s in scores]


def _sequence_logits():
    return torch.tensor([[value] for value in NATIVE_LOGITS], dtype=torch.float32)


def _causal_logits():
    logits = torch.zeros((len(NATIVE_LOGITS), 1, 16))
    logits[:, 0, 10] = torch.tensor(NATIVE_LOGITS)  # mxbai "1" token
    return logits


def test_gte_scores_are_the_sigmoid_pair_of_the_native_logit() -> None:
    scorer = hf_scorer.GteScorer(Tokenizer(), Model(_sequence_logits()))
    scores = scorer.score(_inputs(*"abcde"))
    for (no, yes, native), raw in zip(_pairs(scores), NATIVE_LOGITS, strict=True):
        expected = float(torch.sigmoid(torch.tensor(raw, dtype=torch.float32)))
        assert native == pytest.approx(raw)
        assert yes == pytest.approx(expected, abs=1e-7)
        assert no == pytest.approx(1.0 - expected, abs=1e-7)


def test_mxbai_scores_are_the_offset_sigmoid_pair_of_the_native_difference() -> None:
    scorer = hf_scorer.MxbaiRerankerScorer(Tokenizer(), Model(_causal_logits()))
    scores = scorer.score(_inputs(*"abcde"))
    for (no, yes, native), raw in zip(_pairs(scores), NATIVE_LOGITS, strict=True):
        expected = float(torch.sigmoid(torch.tensor(raw - 4.5, dtype=torch.float32)))
        assert native == pytest.approx(raw)
        assert yes == pytest.approx(expected, abs=1e-7)
        assert no == pytest.approx(1.0 - expected, abs=1e-7)


def test_every_rostered_scorer_resolves_to_an_adapter_and_nothing_else_is_registered() -> None:
    from landuse_relevance_bench.domain.scorers import SCORER_ROSTER

    ids = {spec.model_id for spec in SCORER_ROSTER}
    assert {spec.model_id: hf_scorer.scorer_class_for(spec.model_id) for spec in SCORER_ROSTER}
    assert set(hf_scorer._ADAPTERS) == ids
