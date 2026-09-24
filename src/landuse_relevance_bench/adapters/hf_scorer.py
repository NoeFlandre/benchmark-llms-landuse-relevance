"""Scoring the benchmark with models that never generate text.

Every adapter returns a normalised yes-score per item and keeps the model's native
score for auditability. The families are next-token readers of causal LMs (Qwen3
rerankers, mxbai's binary ``1``/``0`` rule, and the log-probability read of a
generative model), GTE's sequence-classification logit, Laya's typed ``noul``
probability, and zero-shot models asked one shared hypothesis: NLI cross-encoders
through the ``zero-shot-classification`` pipeline, GLiClass and GLiNER2.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any

from landuse_relevance_bench.adapters.hf_scorer_prompt import reranker_input
from landuse_relevance_bench.adapters.pipeline import SCORING_SEQUENCE_LENGTH, RunRequest
from landuse_relevance_bench.domain.engine import LabelScorer, LabelScores, ScoringInput
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.scorers import MXBAI_LOGIT_OFFSET
from landuse_relevance_bench.domain.variants import repository_of

_SINGLE_LOGIT_COUNT = 1
_BINARY_LOGIT_COUNT = 2


def _load_transformers() -> Any:
    """Load the optional runtime without making the type checker index its registry."""
    module_name = "trans" + "formers"
    importer = __import__("builtins").__import__
    return importer(module_name)


@dataclass(frozen=True, slots=True)
class ScorerSettings:
    """What a scoring run needs; no decoding settings, because nothing is decoded."""

    dtype: str
    device_map: str = "auto"


class _CudaMeasurement:
    """Optional CUDA timing/memory hooks shared by every scorer.

    Scorers expose the torch module holding their weights as ``_model``; one whose SDK
    hides it (Laya) overrides :meth:`_measured_module`.
    """

    def _measured_module(self) -> Any:
        return getattr(self, "_model", None)

    def _measurement_device(self) -> Any:
        module = self._measured_module()
        device = getattr(module, "device", None)
        if getattr(device, "type", None) != "cuda":
            # Some SDK models (GLiClass) expose no ``device``; read it from their weights.
            parameters = getattr(module, "parameters", None)
            first = next(iter(parameters()), None) if callable(parameters) else None
            device = getattr(first, "device", None)
        return device if getattr(device, "type", None) == "cuda" else None

    def begin_measurement(self) -> None:
        import torch

        device = self._measurement_device()
        if device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def end_measurement(self) -> None:
        import torch

        device = self._measurement_device()
        if device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    @property
    def peak_vram_bytes(self) -> int | None:
        import torch

        device = self._measurement_device()
        if device is None or not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated(device))


def _left_padded_tokenizer(model_id: str, revision: str | None) -> Any:
    """A tokenizer that pads on the left, so the last position is every row's next token."""
    tokenizer: Any = _load_transformers().AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _pretrained(
    auto_class: str, model_id: str, settings: ScorerSettings, revision: str | None, **extra: Any
) -> Any:
    import torch

    model: Any = getattr(_load_transformers(), auto_class).from_pretrained(
        model_id,
        revision=revision,
        dtype=getattr(torch, settings.dtype),
        device_map=settings.device_map,
        **extra,
    )
    model.eval()
    return model


class _TransformersScorer(_CudaMeasurement):
    """A tokenizer and a Transformers model, with the checkpoint's resolved revision."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self._tokenizer = tokenizer
        self._model = model

    @property
    def revision(self) -> str:
        return str(getattr(self._model.config, "_commit_hash", "") or "")

    def _batch(self, *texts: Sequence[str], **options: Any) -> dict[str, Any]:
        batch = self._tokenizer(
            *texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SCORING_SEQUENCE_LENGTH,
            **options,
        )
        return {k: v.to(self._model.device) for k, v in batch.items()}

    def _last_token_logits(self, texts: Sequence[str]) -> Any:
        """Next-token logits for each already-templated text, in one forward pass."""
        import torch

        batch = self._batch(texts, add_special_tokens=False)
        with torch.inference_mode():
            return self._model(**batch).logits[:, -1, :]


class _CausalScorer(_TransformersScorer):
    """A causal LM read at its next token; subclasses choose the turn and the rule."""

    @classmethod
    def load(cls, model_id: str, settings: ScorerSettings, revision: str | None = None) -> Any:
        return cls(
            _left_padded_tokenizer(model_id, revision),
            _pretrained("AutoModelForCausalLM", model_id, settings, revision),
        )


class RerankerScorer(_CausalScorer):
    """Reads a verdict from the yes/no logits of a reranker's next token."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        super().__init__(tokenizer, model)
        self._yes_id, self._no_id = (
            _single_token_id(tokenizer, "yes"),
            _single_token_id(tokenizer, "no"),
        )

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        import torch

        if not inputs:
            return []
        logits = self._last_token_logits([reranker_input(item.prompt) for item in inputs])
        pair = torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=-1)
        probabilities = torch.softmax(pair.float(), dim=-1)
        return [
            LabelScores(
                {Label.NO: float(row[0]), Label.YES: float(row[1])},
                native_score=float(row[1] - row[0]),
            )
            for row in probabilities
        ]


class GteScorer(_TransformersScorer):
    """Score query/document pairs with GTE's sequence-classification logit."""

    @classmethod
    def load(
        cls, model_id: str, settings: ScorerSettings, revision: str | None = None
    ) -> "GteScorer":
        tokenizer: Any = _load_transformers().AutoTokenizer.from_pretrained(
            model_id, revision=revision
        )
        model = _pretrained(
            "AutoModelForSequenceClassification",
            model_id,
            settings,
            revision,
            trust_remote_code=True,
        )
        return cls(tokenizer, model)

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        import torch

        if not inputs:
            return []
        batch = self._batch([item.prompt for item in inputs], [item.sentence for item in inputs])
        with torch.inference_mode():
            logits = self._model(**batch).logits
        native = _sequence_relevance_logits(logits)
        probabilities = torch.sigmoid(native.float())
        return [
            LabelScores(
                {Label.NO: float(1.0 - probability), Label.YES: float(probability)},
                native_score=float(raw),
            )
            for raw, probability in zip(native, probabilities, strict=True)
        ]


class MxbaiRerankerScorer(_CausalScorer):
    """Apply mxbai-rerank-v2's official binary next-token scoring rule."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        super().__init__(tokenizer, model)
        self._yes_id, self._no_id = (
            _single_token_id(tokenizer, "1"),
            _single_token_id(tokenizer, "0"),
        )

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        import torch

        if not inputs:
            return []
        logits = self._last_token_logits([self._as_chat(item) for item in inputs])
        native = logits[:, self._yes_id].float() - logits[:, self._no_id].float()
        probabilities = torch.sigmoid(native - MXBAI_LOGIT_OFFSET)
        return [
            LabelScores(
                {Label.NO: float(1.0 - probability), Label.YES: float(probability)},
                native_score=float(raw),
            )
            for raw, probability in zip(native, probabilities, strict=True)
        ]

    @staticmethod
    def _as_chat(item: ScoringInput) -> str:
        return (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
            "assistant.<|im_end|>\n<|im_start|>user\n"
            f"query: {item.prompt}\n"
            f"document: {item.sentence}\n"
            "You are a search relevance expert who evaluates how well documents match search "
            "queries. For each query-document pair, carefully analyze the semantic relationship "
            "between them, then provide your binary relevance judgment (0 for not relevant, 1 "
            "for relevant).\nRelevance:<|im_end|>\n<|im_start|>assistant\n"
        )


LAYA_SEQUENCE_LENGTH = 1024
LAYA_GROUP_SIZE = 4
LAYA_MODEL_FILES = (
    "rl_agent_config.json",
    "model.safetensors",
    "tokenizer/*",
    "encoder/*",
)


def _load_laya() -> Any:
    """Load the optional Laya SDK without making the base CLI import it."""
    importer = __import__("builtins").__import__
    return importer("laya")


class LayaScorer(_CudaMeasurement):
    """Score one sentence per typed Laya ``noul`` question.

    Laya's public API batches questions sharing one state. A small state group is
    therefore used here, with explicit JSON field names in each question so the
    sentence-to-score mapping remains deterministic.
    """

    sequence_length = LAYA_SEQUENCE_LENGTH

    def __init__(self, agent: Any, revision: str) -> None:
        self._agent = agent
        self._revision = revision

    def _measured_module(self) -> Any:
        return self._agent

    @classmethod
    def load(
        cls, model_id: str, settings: ScorerSettings, revision: str | None = None
    ) -> "LayaScorer":
        import torch
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=list(LAYA_MODEL_FILES),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = _load_laya().load(local_path, device=device)
        resolved_revision = revision or Path(local_path).name
        del settings  # Laya's SDK selects its own dtype, recorded as runtime_dtype
        return cls(agent, resolved_revision)

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def runtime_dtype(self) -> str:
        """Expose the dtype selected by Laya's device-aware runtime."""
        value = getattr(self._agent, "dtype", None)
        if value is None:
            return "sdk-default"
        return str(value).removeprefix("torch.")

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        if not inputs:
            return []
        scores: list[LabelScores] = []
        for start in range(0, len(inputs), LAYA_GROUP_SIZE):
            group = inputs[start : start + LAYA_GROUP_SIZE]
            state = {f"sentence_{index}": item.sentence for index, item in enumerate(group)}
            questions = {
                f"relevance_{index}": {
                    "type": "noul",
                    "instructions": _laya_instruction(item.prompt, f"sentence_{index}"),
                }
                for index, item in enumerate(group)
            }
            response = self._agent.predict(state, questions)
            answers = response["answers"]
            for index in range(len(group)):
                probability = _laya_probability(answers[f"relevance_{index}"])
                scores.append(
                    LabelScores(
                        {Label.NO: 1.0 - probability, Label.YES: probability},
                        native_score=probability,
                    )
                )
        return scores


def _laya_instruction(prompt: str, field: str) -> str:
    """Keep the shared rubric while pointing Laya at one JSON state field."""
    document_marker = "<Document>:"
    rubric = prompt.rsplit(document_marker, 1)[0].strip()
    return f"{rubric}\nEvaluate the document in JSON field {field}"


def _laya_probability(answer: Any) -> float:
    """Read and validate Laya's rounded probability for the true ``noul`` option."""
    probability = float(answer["noul"])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Laya returned an invalid noul probability: {probability}")
    return probability


ZEROSHOT_HYPOTHESIS_MARKER = "HYPOTHESIS:"
ZEROSHOT_SEQUENCE_LENGTH = 512


def batch_hypothesis(inputs: Sequence[ScoringInput]) -> str:
    """The one hypothesis a zero-shot batch is scored against, refusing a mixed batch."""
    hypotheses = {zeroshot_hypothesis(item.prompt) for item in inputs}
    if len(hypotheses) != 1:
        raise ValueError(f"a zero-shot batch must share one hypothesis, got {len(hypotheses)}")
    return hypotheses.pop()


def zeroshot_hypothesis(prompt: str) -> str:
    """Read the shared hypothesis out of the rendered zero-shot prompt.

    The hypothesis lives in the prompt file, not in code, so its sha256 is pinned in
    every result exactly like an LLM prompt's.
    """
    _, marker, hypothesis = prompt.rpartition(ZEROSHOT_HYPOTHESIS_MARKER)
    hypothesis = hypothesis.strip()
    if not marker or not hypothesis:
        raise ValueError(f"zero-shot prompt has no {ZEROSHOT_HYPOTHESIS_MARKER!r} line")
    return hypothesis


def _probability_scores(probability: float, native: float | None = None) -> LabelScores:
    probability = min(max(float(probability), 0.0), 1.0)
    return LabelScores(
        {Label.NO: 1.0 - probability, Label.YES: probability},
        native_score=probability if native is None else native,
    )


def _torch_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class NliZeroShotScorer(_CudaMeasurement):
    """Entailment probability of the land-use hypothesis, via the zero-shot pipeline.

    This is the checkpoints' documented interface: ``pipeline("zero-shot-classification")``
    with the sentence as premise. A single candidate label is scored as entailment
    against contradiction (``not_entailment`` for two-way models), which is what the
    pipeline does for ``multi_label=True``.
    """

    sequence_length = ZEROSHOT_SEQUENCE_LENGTH

    def __init__(self, pipeline: Any, revision: str) -> None:
        self._pipeline = pipeline
        self._model = pipeline.model
        self._revision = revision

    @classmethod
    def load(
        cls, model_id: str, settings: ScorerSettings, revision: str | None = None
    ) -> "NliZeroShotScorer":
        import torch

        transformers = _load_transformers()
        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
        tokenizer.model_max_length = min(tokenizer.model_max_length, ZEROSHOT_SEQUENCE_LENGTH)
        model: Any = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, settings.dtype)
        )
        pipeline = transformers.pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=_torch_device(),
        )
        resolved = str(getattr(model.config, "_commit_hash", "") or "")
        return cls(pipeline, revision or resolved)

    @property
    def revision(self) -> str:
        return self._revision

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        if not inputs:
            return []
        hypothesis = batch_hypothesis(inputs)
        results = self._pipeline(
            [item.sentence for item in inputs],
            candidate_labels=[hypothesis],
            hypothesis_template="{}",
            multi_label=True,
            batch_size=len(inputs),
            truncation=True,
        )
        if isinstance(results, dict):
            results = [results]
        return [_probability_scores(result["scores"][0]) for result in results]


class GliClassScorer(_CudaMeasurement):
    """GLiClass's sigmoid score for the single land-use label."""

    sequence_length = ZEROSHOT_SEQUENCE_LENGTH

    def __init__(self, pipeline: Any, revision: str) -> None:
        self._pipeline = pipeline
        self._model = getattr(pipeline, "model", None)
        self._revision = revision

    @classmethod
    def load(
        cls, model_id: str, settings: ScorerSettings, revision: str | None = None
    ) -> "GliClassScorer":
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        transformers = _load_transformers()

        model = GLiClassModel.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, settings.dtype)
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
        # multi-label scores each label independently, so one label keeps its own
        # probability instead of being normalised to 1 against nothing.
        pipeline = ZeroShotClassificationPipeline(
            model,
            tokenizer,
            classification_type="multi-label",
            device=_torch_device(),
            max_length=ZEROSHOT_SEQUENCE_LENGTH,
            progress_bar=False,
        )
        resolved = str(getattr(model.config, "_commit_hash", "") or "")
        return cls(pipeline, revision or resolved)

    @property
    def revision(self) -> str:
        return self._revision

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        if not inputs:
            return []
        hypothesis = batch_hypothesis(inputs)
        results = self._pipeline(
            [item.sentence for item in inputs],
            [hypothesis],
            threshold=0.0,
            batch_size=len(inputs),
        )
        return [_label_probability(result, hypothesis) for result in results]


def _label_probability(result: Any, label: str) -> LabelScores:
    for entry in result:
        if entry["label"] == label:
            return _probability_scores(entry["score"])
    raise ValueError(f"GLiClass returned no score for the label {label!r}")


class Gliner2Scorer(_CudaMeasurement):
    """GLiNER2.5's classification probability for the single land-use label."""

    TASK = "landuse"
    # classify_text takes one text per call, so the batch is scored item by item.
    effective_batch_size = 1
    # classify_text applies the checkpoint's own window; the adapter sets no cap.
    sequence_length = None

    def __init__(self, model: Any, revision: str) -> None:
        self._model = model
        self._revision = revision

    @classmethod
    def load(
        cls, model_id: str, settings: ScorerSettings, revision: str | None = None
    ) -> "Gliner2Scorer":
        from huggingface_hub import snapshot_download

        # Imported by name: gliner2 lives in its own environment (it pins Transformers 4).
        auto_extractor = __import__("builtins").__import__("gliner2").AutoExtractor

        del settings  # the SDK picks its own runtime dtype, recorded as runtime_dtype
        local_path = snapshot_download(model_id, revision=revision)
        model = auto_extractor.from_pretrained(
            local_path, map_location=_torch_device().split(":")[0]
        )
        model.eval()
        return cls(model, revision or Path(local_path).name)

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def runtime_dtype(self) -> str:
        dtype = getattr(next(self._model.parameters()), "dtype", None)
        return "sdk-default" if dtype is None else str(dtype).removeprefix("torch.")

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        if not inputs:
            return []
        hypothesis = batch_hypothesis(inputs)
        tasks = {self.TASK: {"labels": [hypothesis], "multi_label": True, "cls_threshold": 0.0}}
        scores = []
        for item in inputs:
            result = self._model.classify_text(item.sentence, tasks, include_confidence=True)
            scores.append(_probability_scores(_gliner2_confidence(result[self.TASK], hypothesis)))
        return scores


def _gliner2_confidence(task_result: Any, label: str) -> float:
    """Read one label's confidence from GLiNER2's classification output shapes."""
    entries = task_result if isinstance(task_result, list) else [task_result]
    for entry in entries:
        if isinstance(entry, dict) and entry.get("label") == label:
            return float(entry["confidence"])
    raise ValueError(f"GLiNER2 returned no confidence for the label {label!r}")


class CausalLogprobScorer(_CausalScorer):
    """Reads a generative model's verdict from one forward pass, without decoding.

    The input is the exact chat turn the generative run used (same prompt, same
    template), with the model's opening ``<think>`` closed empty so that the next
    token is the verdict itself. ``P(yes)`` is renormalised over the two verdict
    tokens; the native score is ``log P(yes) - log P(no)`` over the full vocabulary.
    """

    THINK_CLOSE = "</think>"

    def __init__(self, tokenizer: Any, model: Any) -> None:
        super().__init__(tokenizer, model)
        self._yes_ids = _verdict_token_ids(tokenizer, "yes")
        self._no_ids = _verdict_token_ids(tokenizer, "no")

    def as_chat(self, prompt: str) -> str:
        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if text.rstrip().endswith("<think>"):
            text = text.rstrip() + self.THINK_CLOSE
        return text

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        import torch

        if not inputs:
            return []
        logits = self._last_token_logits([self.as_chat(item.prompt) for item in inputs]).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        log_yes = torch.logsumexp(log_probs[:, self._yes_ids], dim=-1)
        log_no = torch.logsumexp(log_probs[:, self._no_ids], dim=-1)
        p_yes = torch.sigmoid(log_yes - log_no)
        return [
            _probability_scores(float(p), native=float(ly - ln))
            for p, ly, ln in zip(p_yes, log_yes, log_no, strict=True)
        ]


def _verdict_token_ids(tokenizer: Any, word: str) -> list[int]:
    """Single-token spellings of a verdict (case and leading-space variants)."""
    ids: set[int] = set()
    for candidate in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError(f"{word!r} has no single-token spelling; log-probabilities undefined")
    return sorted(ids)


def _single_token_id(tokenizer: Any, word: str) -> int:
    """The id of ``word`` as the model would emit it, refusing an ambiguous encoding."""
    for candidate in (word, f" {word}"):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(f"{word!r} is not a single token for this tokenizer; scores would be wrong")


def _sequence_relevance_logits(logits: Any) -> Any:
    """Return one positive-class logit for either one- or two-logit checkpoints."""
    if logits.shape[-1] == _SINGLE_LOGIT_COUNT:
        return logits.view(-1)
    if logits.shape[-1] == _BINARY_LOGIT_COUNT:
        return logits[:, 1] - logits[:, 0]
    raise ValueError(f"expected one or two sequence-classification logits, got {logits.shape}")


def mxbai_normalize(native_score: float) -> float:
    """Match mxbai-rerank-v2's estimated-max sigmoid normalisation."""
    return 1.0 / (1.0 + exp(-(native_score - MXBAI_LOGIT_OFFSET)))


def scorer_class_for(model_id: str) -> type[Any]:
    """Select the adapter matching a rostered scoring checkpoint."""
    adapters: dict[str, type[Any]] = {
        "Alibaba-NLP/gte-multilingual-reranker-base": GteScorer,
        "mixedbread-ai/mxbai-rerank-base-v2": MxbaiRerankerScorer,
        "convaiinnovations/laya-multilingual": LayaScorer,
        "Qwen/Qwen3-Reranker-0.6B": RerankerScorer,
        "Qwen/Qwen3-Reranker-4B": RerankerScorer,
        "LiquidAI/LFM2.5-2.6B@logprob": CausalLogprobScorer,
        "knowledgator/gliclass-multilang-mini": GliClassScorer,
        "MoritzLaurer/bge-m3-zeroshot-v2.0": NliZeroShotScorer,
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": NliZeroShotScorer,
        "BalaRajesh1/mmbert-small-nli": NliZeroShotScorer,
        "fastino/gliner2.5-multi-v1": Gliner2Scorer,
    }
    try:
        return adapters[model_id]
    except KeyError as exc:
        raise ValueError(f"no scoring adapter is registered for {model_id!r}") from exc


def provide_scorer(request: RunRequest) -> tuple[LabelScorer, str]:
    """The default scorer provider used by the CLI."""
    settings = ScorerSettings(dtype=request.dtype)
    scorer = scorer_class_for(request.model_id).load(
        repository_of(request.model_id), settings, revision=request.revision
    )
    return scorer, request.revision or scorer.revision
