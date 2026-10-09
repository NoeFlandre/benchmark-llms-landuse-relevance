"""CPU-only doubles exercise the optional Hugging Face scorer adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from landuse_relevance_bench.adapters import hf_scorer
from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.engine import ScoringInput
from landuse_relevance_bench.domain.labels import Label


def _torch() -> Any:
    return pytest.importorskip("torch")


class Tokenizer:
    eos_token = "<eos>"  # noqa: S105 - tokenizer sentinel, not a secret
    pad_token_id = None
    pad_token = None
    padding_side = "right"
    mask_token = "<mask>"  # noqa: S105 - tokenizer sentinel, not a secret
    mask_token_id = 9
    model_max_length = 4096
    _ids = {
        "yes": [1],
        " yes": [2],
        "Yes": [3],
        " Yes": [4],
        "no": [5],
        " no": [6],
        "No": [7],
        " No": [8],
        "1": [10],
        " 1": [10],
        "0": [11],
        " 0": [11],
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return self._ids.get(text, [20, 21])

    def __call__(self, *texts: Any, **_: Any) -> dict[str, Any]:
        torch = _torch()
        self.seen_texts = texts
        count = len(texts[0])
        rows = [
            [4, self.mask_token_id, 5] if self.mask_token in text else [4, 5, 6]
            for text in texts[0]
        ]
        return {"input_ids": torch.tensor(rows[:count], dtype=torch.long)}

    def apply_chat_template(self, messages: Any, **_: Any) -> str:
        return f"{messages[0]['content']}<think>"


class Model:
    device = "cpu"

    def __init__(self, logits: Any | None = None) -> None:
        self.config = SimpleNamespace(_commit_hash="loaded-revision")
        self.logits = logits
        self.evaluated = False

    def eval(self) -> None:
        self.evaluated = True

    def __call__(self, input_ids: Any, **_: Any) -> Any:
        torch = _torch()
        if self.logits is not None:
            return SimpleNamespace(logits=self.logits[: len(input_ids)])
        output = torch.zeros((len(input_ids), input_ids.shape[1], 16))
        output[:, :, 1] = 2.0
        output[:, :, 5] = 1.0
        output[:, :, 10] = 2.0
        output[:, :, 11] = 1.0
        return SimpleNamespace(logits=output)


def _inputs(*sentences: str) -> list[ScoringInput]:
    prompt = "Question\nHYPOTHESIS: land use"
    return [ScoringInput(prompt, sentence) for sentence in sentences]


def test_transformers_import_and_loader_helpers_are_lazy_and_revision_pinned(monkeypatch) -> None:
    torch = _torch()
    tokenizer = Tokenizer()
    model = Model()
    calls: list[tuple[str, dict[str, Any]]] = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Tokenizer:
            calls.append((model_id, kwargs))
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Model:
            calls.append((model_id, kwargs))
            return model

    module = SimpleNamespace(AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel)
    monkeypatch.setitem(sys.modules, "transformers", module)
    imported = hf_scorer._load_transformers()
    assert imported is module

    loaded_tokenizer = hf_scorer._left_padded_tokenizer("owner/model", "rev")
    settings = hf_scorer.ScorerSettings(dtype="float32")
    loaded_model = hf_scorer._pretrained("AutoModelForCausalLM", "owner/model", settings, "rev")

    assert loaded_tokenizer is tokenizer
    assert (tokenizer.padding_side, tokenizer.pad_token) == ("left", tokenizer.eos_token)
    assert loaded_model is model and model.evaluated
    assert calls == [
        ("owner/model", {"revision": "rev"}),
        (
            "owner/model",
            {
                "revision": "rev",
                "dtype": torch.float32,
                "device_map": "auto",
            },
        ),
    ]

    loaded = hf_scorer.RerankerScorer.load(
        "owner/model", hf_scorer.ScorerSettings(dtype="float32"), revision="rev"
    )
    assert type(loaded) is hf_scorer.RerankerScorer
    assert loaded.revision == "loaded-revision"


def test_transformers_scorer_adapters_score_and_handle_empty_batches() -> None:
    torch = _torch()
    tokenizer = Tokenizer()
    logits = torch.zeros((2, 3, 16))
    logits[:, -1, 1] = 4
    logits[:, -1, 5] = 1
    logits[:, -1, 10] = 8
    logits[:, -1, 11] = 1
    model = Model(logits)

    reranker = hf_scorer.RerankerScorer(tokenizer, model)
    mxbai = hf_scorer.MxbaiRerankerScorer(tokenizer, model)
    assert reranker.score([]) == []
    assert mxbai.score([]) == []
    assert reranker.score(_inputs("forest", "council"))[0].verdict is Label.YES
    assert mxbai.score(_inputs("forest", "council"))[0].verdict is Label.YES
    assert reranker.revision == "loaded-revision"

    single = Model(torch.tensor([[2.0], [0.5]]))
    binary = Model(torch.tensor([[1.0, 3.0], [4.0, 1.0]]))
    assert hf_scorer._sequence_relevance_logits(single.logits).tolist() == [2.0, 0.5]
    assert hf_scorer._sequence_relevance_logits(binary.logits).tolist() == [2.0, -3.0]
    gte = hf_scorer.GteScorer(tokenizer, binary)
    scores = gte.score(_inputs("forest", "council"))
    assert [score.verdict for score in scores] == [Label.YES, Label.NO]
    assert gte.score([]) == []
    with pytest.raises(ValueError, match="one or two"):
        hf_scorer._sequence_relevance_logits(torch.zeros((1, 3)))


def test_laya_groups_questions_and_rejects_invalid_probabilities() -> None:
    class Agent:
        dtype = None

        def __init__(self) -> None:
            self.groups: list[tuple[dict[str, str], dict[str, Any]]] = []

        def predict(self, state: dict[str, str], questions: dict[str, Any]) -> dict[str, Any]:
            self.groups.append((state, questions))
            return {"answers": {key: {"noul": 0.6} for key in questions}}

    agent = Agent()
    scorer = hf_scorer.LayaScorer(agent, "laya-rev")
    scores = scorer.score(_inputs("a", "b", "c", "d", "e"))
    assert scorer.score([]) == []
    assert len(scores) == 5 and len(agent.groups) == 2
    assert scorer.runtime_dtype == "sdk-default"
    assert hf_scorer._laya_instruction("rubric\n<Document>:", "sentence_0").endswith(
        "Evaluate the document in JSON field sentence_0"
    )
    with pytest.raises(ValueError, match="invalid noul"):
        hf_scorer._laya_probability({"noul": 1.1})


def test_zero_shot_and_gliclass_adapters_validate_labels_and_empty_inputs() -> None:
    class Pipeline:
        model = Model()

        def __init__(self, result: Any) -> None:
            self.result = result

        def __call__(self, *_: Any, **__: Any) -> Any:
            return self.result

    nli = hf_scorer.NliZeroShotScorer(
        Pipeline({"labels": ["land use"], "scores": [1.2]}), "nli-rev"
    )
    assert nli.score([]) == []
    assert nli.score(_inputs("forest"))[0].scores[Label.YES] == 1.0
    assert nli.peak_vram_bytes is None
    with pytest.raises(ValueError, match="share one hypothesis"):
        nli.score([ScoringInput("HYPOTHESIS: A", "a"), ScoringInput("HYPOTHESIS: B", "b")])

    one_label = "land use"
    glib = hf_scorer.GliClassScorer(Pipeline([[{"label": one_label, "score": 0.7}]]), "gli-rev")
    assert glib.score(_inputs("forest"))[0].scores[Label.YES] == pytest.approx(0.7)
    assert glib.score([]) == []
    with pytest.raises(ValueError, match="no score"):
        hf_scorer._label_probability([{"label": "other", "score": 0.1}], one_label)
    assert hf_scorer._probability_scores(-1).scores[Label.YES] == 0.0
    assert hf_scorer._probability_scores(2).scores[Label.YES] == 1.0


def test_gliner2_result_shapes_and_masked_prompt_token_validation() -> None:
    torch = _torch()

    class Extractor:
        def __init__(self, device: str) -> None:
            self.device = device

        def parameters(self) -> Any:
            return iter([SimpleNamespace(dtype=None)])

        def classify_text(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {"landuse": [{"label": "land use", "confidence": 0.8}]}

    extractor = Extractor("cpu")
    gliner = hf_scorer.Gliner2Scorer(extractor, "gliner-rev")
    assert gliner.score(_inputs("forest"))[0].scores[Label.YES] == pytest.approx(0.8)
    assert gliner.score([]) == []
    assert gliner.runtime_dtype == "sdk-default"
    assert (
        hf_scorer._gliner2_confidence({"label": "land use", "confidence": 0.4}, "land use") == 0.4
    )
    with pytest.raises(ValueError, match="no confidence"):
        hf_scorer._gliner2_confidence({"label": "other", "confidence": 0.4}, "land use")

    class MaskTokenizer(Tokenizer):
        pass

    mask_logits = torch.zeros((1, 3, 16))
    mask_logits[0, 1, 1] = 4
    mask_logits[0, 1, 5] = 1
    masked = hf_scorer.MaskedTokenScorer(MaskTokenizer(), Model(mask_logits))
    assert masked.score([ScoringInput("[MASK] forest", "forest")])[0].verdict is Label.YES
    with pytest.raises(ValueError, match="tokenize to exactly one"):

        class BadMaskTokenizer(MaskTokenizer):
            def __call__(self, *texts: Any, **kwargs: Any) -> dict[str, Any]:
                return {"input_ids": torch.tensor([[4, 5, 6]])}

        hf_scorer.MaskedTokenScorer(BadMaskTokenizer(), Model(mask_logits)).score(
            [ScoringInput("[MASK] forest", "forest")]
        )


def test_logprob_scorer_scores_without_generation_and_checks_token_spellings() -> None:
    torch = _torch()
    logits = torch.zeros((1, 3, 16))
    logits[0, -1, 1] = 3
    logits[0, -1, 5] = 1
    scorer = hf_scorer.CausalLogprobScorer(Tokenizer(), Model(logits))
    scores = scorer.score(_inputs("forest"))
    assert scores[0].verdict is Label.YES
    assert scorer.score([]) == []
    assert hf_scorer._torch_device() in {"cpu", "cuda:0"}

    class NoVerdictTokenizer(Tokenizer):
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return [40, 41]

    with pytest.raises(ValueError, match="no single-token spelling"):
        hf_scorer._verdict_token_ids(NoVerdictTokenizer(), "maybe")
    with pytest.raises(ValueError, match="not a single token"):
        hf_scorer._single_token_id(NoVerdictTokenizer(), "maybe")
    with pytest.raises(ValueError, match="one or two"):
        hf_scorer._sequence_relevance_logits(torch.zeros((1, 3)))


def test_scorer_selection_and_provider_reject_unknown_models(monkeypatch) -> None:
    from landuse_relevance_bench.adapters import hf_scorer as module

    original_selector = module.scorer_class_for

    class Loaded:
        revision = "resolved-rev"

        @classmethod
        def load(
            cls,
            _model_id: str,
            _settings: Any,
            revision: str | None = None,
        ):
            assert revision == "requested-rev"
            return cls()

    monkeypatch.setattr(module, "scorer_class_for", lambda _model_id: Loaded)
    request = RunRequest.for_run(
        "LiquidAI/LFM2.5-Encoder-350M",
        language="en",
        benchmark_path=Path("unused"),
        prompt_path=Path("unused"),
        output_dir=Path("unused"),
        revision="requested-rev",
    )
    scorer, revision = module.provide_scorer(request)
    assert type(scorer) is Loaded
    assert scorer.revision == "resolved-rev"
    assert revision == "requested-rev"
    monkeypatch.setattr(module, "scorer_class_for", original_selector)
    with pytest.raises(ValueError, match="no scoring adapter"):
        module.scorer_class_for("unknown/model")


def test_cuda_measurement_uses_device_and_parameter_fallback(monkeypatch) -> None:
    calls: list[tuple[str, Any]] = []
    cpu = SimpleNamespace(type="cpu")
    cuda = SimpleNamespace(type="cuda")
    available = False
    fake_cuda = SimpleNamespace(
        is_available=lambda: available,
        synchronize=lambda device: calls.append(("sync", device)),
        reset_peak_memory_stats=lambda device: calls.append(("reset", device)),
        max_memory_allocated=lambda device: 1234,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    class Measurement(hf_scorer._CudaMeasurement):
        def __init__(self, module: Any) -> None:
            self._model = module

    cpu_measurement = Measurement(
        SimpleNamespace(device=cpu, parameters=lambda: iter([SimpleNamespace(device=cpu)]))
    )
    assert cpu_measurement._measurement_device() is None
    cpu_measurement.begin_measurement()
    cpu_measurement.end_measurement()
    assert cpu_measurement.peak_vram_bytes is None

    available = True
    gpu_measurement = Measurement(SimpleNamespace(device=cuda))
    assert gpu_measurement._measurement_device() is cuda
    gpu_measurement.begin_measurement()
    gpu_measurement.end_measurement()
    assert gpu_measurement.peak_vram_bytes == 1234
    fallback = Measurement(
        SimpleNamespace(
            device=cpu,
            parameters=lambda: iter([SimpleNamespace(device=cuda)]),
        )
    )
    assert fallback._measurement_device() is cuda
    assert calls == [("sync", cuda), ("reset", cuda), ("sync", cuda)]


def test_all_optional_scorer_loaders_are_mockable_without_network(monkeypatch) -> None:
    torch = SimpleNamespace(
        float32="float32-dtype",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    tokenizer = Tokenizer()
    model = Model()
    auto_calls: list[tuple[str, dict[str, Any]]] = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Tokenizer:
            auto_calls.append(("tokenizer", {"model_id": model_id, **kwargs}))
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Model:
            auto_calls.append(("model", {"model_id": model_id, **kwargs}))
            loaded_model = Model()
            loaded_model.config._commit_hash = kwargs.get("revision")
            return loaded_model

    class Pipeline:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

        def __call__(self, *_: Any, **__: Any) -> list[Any]:
            return []

    transformers = SimpleNamespace(
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
        AutoModelForSequenceClassification=AutoModel,
        AutoModelForMaskedLM=AutoModel,
        pipeline=lambda *_args, **kwargs: Pipeline(**kwargs),
    )
    monkeypatch.setattr(hf_scorer, "_load_transformers", lambda: transformers)
    monkeypatch.setattr(hf_scorer, "_torch_device", lambda: "cpu")
    settings = hf_scorer.ScorerSettings(dtype="float32")

    gte = hf_scorer.GteScorer.load("owner/gte", settings, revision="gte-rev")
    masked = hf_scorer.MaskedTokenScorer.load("owner/masked", settings, revision="mask-rev")
    nli = hf_scorer.NliZeroShotScorer.load("owner/nli", settings, revision="nli-rev")
    assert gte.revision == "gte-rev"
    assert masked.revision == "mask-rev"
    assert nli.revision == "nli-rev"
    assert tokenizer.model_max_length == hf_scorer.ZEROSHOT_SEQUENCE_LENGTH

    gliclass = ModuleType("gliclass")
    gliclass.GLiClassModel = type(
        "GLiClassModel", (), {"from_pretrained": staticmethod(lambda *_a, **_k: model)}
    )
    gliclass.ZeroShotClassificationPipeline = lambda *args, **kwargs: Pipeline(
        model=args[0], **kwargs
    )
    monkeypatch.setitem(sys.modules, "gliclass", gliclass)
    gli = hf_scorer.GliClassScorer.load("owner/gli", settings, revision="gli-rev")
    assert gli.revision == "gli-rev"

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda model_id, **kwargs: f"/cache/{model_id.split('/')[-1]}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    auto_extractor = type(
        "AutoExtractor",
        (),
        {"from_pretrained": staticmethod(lambda path, **kwargs: model)},
    )
    gliner_module = ModuleType("gliner2")
    gliner_module.AutoExtractor = auto_extractor
    monkeypatch.setitem(sys.modules, "gliner2", gliner_module)
    model.evaluated = False
    gliner = hf_scorer.Gliner2Scorer.load("owner/gliner", settings, revision="gli-rev")
    assert gliner.revision == "gli-rev"
    assert model.evaluated  # Fresh flag: the GLiNER loader must put the model in eval mode.

    laya_agent = SimpleNamespace(dtype="torch.float16")
    monkeypatch.setattr(
        hf_scorer,
        "_load_laya",
        lambda: SimpleNamespace(load=lambda path, **kwargs: laya_agent),
    )
    laya = hf_scorer.LayaScorer.load("owner/laya", settings)
    assert laya.revision == "laya"
    assert laya.runtime_dtype == "float16"
    assert any(
        call[1].get("trust_remote_code") is True for call in auto_calls if call[0] == "model"
    )
