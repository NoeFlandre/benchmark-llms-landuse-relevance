"""The GLiClass Ultra adapter follows the pinned upstream score contract."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from landuse_relevance_bench.adapters import hf_scorer
from landuse_relevance_bench.adapters.hf_scorer import (
    GliClassUltraScorer,
    ScorerSettings,
    scorer_class_for,
)
from landuse_relevance_bench.application import manifest_for
from landuse_relevance_bench.domain.engine import ScoringInput
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.scorers import (
    GLICLASS_ULTRA_ID,
    GLICLASS_ULTRA_NATIVE_LANGUAGES,
    GLICLASS_ULTRA_REVISION,
    scorer_for,
)

HYPOTHESIS = (
    "This sentence contains information useful for inferring the land use or land cover "
    "of the associated geographic place."
)
PROMPT = f"TARGET SENTENCE: forest\n\nHYPOTHESIS: {HYPOTHESIS}\n"


def test_ultra_is_pinned_and_runs_all_85_languages() -> None:
    spec = scorer_for(GLICLASS_ULTRA_ID)
    manifest = manifest_for(Path(__file__).parents[2] / "data" / "translations")

    assert len(GLICLASS_ULTRA_REVISION) == 40
    assert len(GLICLASS_ULTRA_NATIVE_LANGUAGES) == 20
    assert spec.total_parameters == 1_720_000_000
    assert spec.supported_languages is None
    assert spec.prompt == "data/prompt_zeroshot.txt"
    assert "native training coverage is 20 languages" in spec.note
    assert scorer_class_for(GLICLASS_ULTRA_ID) is GliClassUltraScorer
    assert manifest.revision == "d431a6c522b3c3a616e4f3d2814906c3011843e7"
    assert len(manifest.languages) == 85
    assert {manifest.files[language].rows for language in manifest.languages} == {300}


def test_ultra_uses_one_independent_label_score_and_the_inclusive_native_cutoff() -> None:
    def pipeline(texts: list[str], labels: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        assert labels == [HYPOTHESIS]
        assert kwargs == {"threshold": 0.0, "batch_size": len(texts)}
        return [
            [{"label": label, "score": score}]
            for label, score in zip(
                labels * len(texts),
                (0.5, 0.499999, 0.500001),
                strict=True,
            )
        ]

    scorer = GliClassUltraScorer(pipeline, "test-revision")
    inputs = [
        ScoringInput(PROMPT, "exact tie"),
        ScoringInput(PROMPT, "below"),
        ScoringInput(PROMPT, "above"),
    ]

    scores = scorer.score(inputs)

    assert [score.verdict for score in scores] == [Label.YES, Label.NO, Label.YES]
    assert [score.scores[Label.YES] for score in scores] == [0.5, 0.499999, 0.500001]
    assert [score.native_score for score in scores] == [0.5, 0.499999, 0.500001]


def test_ultra_loader_pins_model_and_tokenizer_to_the_same_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[dict[str, Any]] = []
    tokenizer_calls: list[dict[str, Any]] = []
    pipeline_calls: list[dict[str, Any]] = []

    class FakeModel:
        config = object()

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeModel":
            model_calls.append({"model_id": model_id, **kwargs})
            return cls()

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeTokenizer":
            tokenizer_calls.append({"model_id": model_id, **kwargs})
            return cls()

    class FakePipeline:
        def __init__(self, model: FakeModel, tokenizer: FakeTokenizer, **kwargs: Any) -> None:
            self.model = model
            pipeline_calls.append({"tokenizer": tokenizer, **kwargs})

    gliclass = ModuleType("gliclass")
    gliclass.GLiClassModel = FakeModel  # type: ignore[attr-defined]
    gliclass.ZeroShotClassificationPipeline = FakePipeline  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliclass", gliclass)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16"))
    monkeypatch.setattr(hf_scorer, "_torch_device", lambda: "cpu")
    monkeypatch.setattr(hf_scorer, "loaded_revision", lambda _model: "resolved-revision")

    scorer = GliClassUltraScorer.load(GLICLASS_ULTRA_ID, ScorerSettings(dtype="float16"))

    assert scorer.revision == GLICLASS_ULTRA_REVISION
    assert model_calls[0]["revision"] == GLICLASS_ULTRA_REVISION
    assert "trust_remote_code" not in model_calls[0]
    assert tokenizer_calls[0]["revision"] == GLICLASS_ULTRA_REVISION
    assert pipeline_calls[0]["classification_type"] == "multi-label"
    assert pipeline_calls[0]["max_length"] == 512
    assert pipeline_calls[0]["progress_bar"] is False
