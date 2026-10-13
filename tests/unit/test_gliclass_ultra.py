"""The GLiClass Ultra adapter follows the pinned upstream score contract."""

import csv
import hashlib
import json
import sys
from collections import Counter
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
from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.application import manifest_for
from landuse_relevance_bench.domain.dataset import BenchmarkItem
from landuse_relevance_bench.domain.engine import ScoringInput
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.orchestration import score_all
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


def _with_tokenizer_guard(pipeline: Any, token_count: int = 1) -> Any:
    def prepare_input(text: str, labels: list[str]) -> str:
        assert labels == [HYPOTHESIS]
        return f"<<LABEL>>{labels[0]}<<SEP>>{text}"

    def tokenizer(texts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        assert kwargs == {"truncation": False, "padding": False}
        return {"input_ids": [[0] * token_count for _ in texts]}

    pipeline.pipe = SimpleNamespace(prepare_input=prepare_input, tokenizer=tokenizer)
    return pipeline


def test_ultra_is_pinned_and_runs_all_85_languages() -> None:
    spec = scorer_for(GLICLASS_ULTRA_ID)
    manifest = manifest_for(Path(__file__).parents[2] / "data" / "translations")

    assert len(GLICLASS_ULTRA_REVISION) == 40
    assert GLICLASS_ULTRA_NATIVE_LANGUAGES == (
        "ar",
        "cs",
        "de",
        "es",
        "et",
        "fi",
        "fr",
        "he",
        "hi",
        "it",
        "lt",
        "lv",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "sv",
        "uk",
        "zh",
    )
    assert spec.total_parameters == 1_720_000_000
    assert spec.supported_languages is None
    assert spec.prompt == "data/prompt_zeroshot.txt"
    assert "native training coverage is 20 languages" in spec.note
    assert scorer_class_for(GLICLASS_ULTRA_ID) is GliClassUltraScorer
    request = RunRequest.for_model(
        GLICLASS_ULTRA_ID,
        language="en",
        benchmark_path=Path("data/translations/en.csv"),
        prompt_path=Path("data/prompt_zeroshot.txt"),
        output_dir=Path("results"),
    )
    assert request.revision == GLICLASS_ULTRA_REVISION
    assert manifest.revision == "d431a6c522b3c3a616e4f3d2814906c3011843e7"
    assert len(manifest.languages) == 85
    assert {manifest.files[language].rows for language in manifest.languages} == {300}


def test_recorded_ultra_token_lengths_cover_the_full_frozen_set() -> None:
    root = Path(__file__).parents[2]
    csv_path = root / "data" / "audits" / "gliclass-ultra-token-lengths.csv"
    metadata_path = root / "data" / "audits" / "gliclass-ultra-token-lengths.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert metadata["model_revision"] == GLICLASS_ULTRA_REVISION
    assert metadata["tokenizer_revision"] == GLICLASS_ULTRA_REVISION
    assert metadata["manifest_revision"] == "d431a6c522b3c3a616e4f3d2814906c3011843e7"
    assert metadata["row_count"] == len(rows) == 25_500
    assert metadata["language_count"] == 85
    assert metadata["rows_per_language"] == 300
    assert len({row["language"] for row in rows}) == 85
    assert set(Counter(row["language"] for row in rows).values()) == {300}
    assert metadata["minimum_token_count"] == min(int(row["token_count"]) for row in rows) == 29
    assert metadata["maximum_token_count"] == max(int(row["token_count"]) for row in rows) == 212
    assert metadata["over_limit_count"] == sum(row["over_limit"] == "True" for row in rows) == 0
    assert metadata["token_lengths_csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()


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

    scorer = GliClassUltraScorer(_with_tokenizer_guard(pipeline), "test-revision")
    inputs = [
        ScoringInput(PROMPT, "exact tie"),
        ScoringInput(PROMPT, "below"),
        ScoringInput(PROMPT, "above"),
    ]

    scores = scorer.score(inputs)

    assert [score.verdict for score in scores] == [Label.YES, Label.NO, Label.YES]
    assert [score.scores[Label.YES] for score in scores] == [0.5, 0.499999, 0.500001]
    assert [score.native_score for score in scores] == [0.5, 0.499999, 0.500001]


def test_ultra_rejects_inputs_that_would_be_truncated_before_scoring() -> None:
    pipeline_calls = 0

    def pipeline(texts: list[str], labels: list[str], **_kwargs: Any) -> list[list[dict[str, Any]]]:
        nonlocal pipeline_calls
        pipeline_calls += 1
        return [[{"label": labels[0], "score": 0.75}] for _ in texts]

    scorer = GliClassUltraScorer(_with_tokenizer_guard(pipeline, token_count=513), "test-revision")

    with pytest.raises(ValueError, match=r"would be truncated.*512"):
        scorer.score([ScoringInput(PROMPT, "a very long input")])

    assert pipeline_calls == 0


def test_ultra_allows_an_input_at_the_configured_sequence_limit() -> None:
    def pipeline(texts: list[str], labels: list[str], **_kwargs: Any) -> list[list[dict[str, Any]]]:
        return [[{"label": labels[0], "score": 0.75}] for _ in texts]

    scorer = GliClassUltraScorer(_with_tokenizer_guard(pipeline, token_count=512), "test-revision")

    (score,) = scorer.score([ScoringInput(PROMPT, "input at the configured limit")])

    assert score.verdict is Label.YES
    assert score.scores[Label.YES] == 0.75


def test_ultra_keeps_full_precision_scores_in_saved_predictions() -> None:
    yes_score = 0.5000001

    def pipeline(texts: list[str], labels: list[str], **_kwargs: Any) -> list[list[dict[str, Any]]]:
        return [[{"label": labels[0], "score": yes_score}] for _ in texts]

    item = BenchmarkItem(
        item_id="0123456789abcdef",
        source_item_id="source-1",
        language="en",
        sentence="A sentence about forests.",
        label=Label.YES,
    )
    template = f"TARGET SENTENCE: {{}}\n\nHYPOTHESIS: {HYPOTHESIS}\n"
    scorer = GliClassUltraScorer(_with_tokenizer_guard(pipeline), "revision")
    (prediction,) = score_all([item], template, scorer)
    restored = type(prediction).from_dict(json.loads(json.dumps(prediction.to_dict())))

    assert prediction.raw_output == "no=0.500000 yes=0.500000 native=0.500000"
    assert prediction.label_scores == {Label.NO: 1.0 - yes_score, Label.YES: yes_score}
    assert prediction.predicted is Label.YES
    assert restored.label_scores == prediction.label_scores


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
