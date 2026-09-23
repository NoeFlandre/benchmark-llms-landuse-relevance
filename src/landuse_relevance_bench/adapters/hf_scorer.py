"""Scoring the benchmark with models that never generate text.

The roster contains four scoring families: Qwen's yes/no next-token rerankers,
GTE's sequence-classification logit, mxbai's binary ``1``/``0`` next-token
logit difference, and Laya's typed ``noul`` decision probability. Each adapter
returns the same normalised relevance score and also keeps its native score for
auditability.
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
    """Optional CUDA timing/memory hooks shared by all Transformer scorers."""

    def _measurement_device(self) -> Any:
        agent = getattr(self, "_agent", None)
        device = getattr(agent, "device", None)
        if getattr(device, "type", None) == "cuda":
            return device
        model = getattr(self, "_model", None)
        if model is None:
            model = getattr(getattr(self, "_pipeline", None), "model", None)
        device = getattr(model, "device", None)
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


class RerankerScorer(_CudaMeasurement):
    """Reads a verdict from the yes/no logits of a reranker's next token."""

    def __init__(self, tokenizer: Any, model: Any, settings: ScorerSettings) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._settings = settings
        self._yes_id, self._no_id = (
            _single_token_id(tokenizer, "yes"),
            _single_token_id(tokenizer, "no"),
        )

    @classmethod
    def load(
        cls, model_id: str, settings: ScorerSettings, revision: str | None = None
    ) -> "RerankerScorer":
        import torch

        transformers = _load_transformers()

        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model: Any = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
        )
        model.eval()
        return cls(tokenizer, model, settings)

    @property
    def revision(self) -> str:
        return str(getattr(self._model.config, "_commit_hash", "") or "")

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        import torch

        if not inputs:
            return []
        texts = [self._as_chat(item.prompt) for item in inputs]
        batch = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SCORING_SEQUENCE_LENGTH,
            add_special_tokens=False,
        )
        batch = {k: v.to(self._model.device) for k, v in batch.items()}
        with torch.inference_mode():
            logits = self._model(**batch).logits[:, -1, :]
        pair = torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=-1)
        probabilities = torch.softmax(pair.float(), dim=-1)
        return [
            LabelScores(
                {Label.NO: float(row[0]), Label.YES: float(row[1])},
                native_score=float(row[1] - row[0]),
            )
            for row in probabilities
        ]

    def _as_chat(self, prompt: str) -> str:
        return reranker_input(prompt)


class GteScorer(_CudaMeasurement):
    """Score query/document pairs with GTE's sequence-classification logit."""

    def __init__(self, tokenizer: Any, model: Any, settings: ScorerSettings) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._settings = settings

    @classmethod
    def load(
        cls, model_id: str, settings: ScorerSettings, revision: str | None = None
    ) -> "GteScorer":
        import torch

        transformers = _load_transformers()

        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
        model: Any = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
            trust_remote_code=True,
        )
        model.eval()
        return cls(tokenizer, model, settings)

    @property
    def revision(self) -> str:
        return str(getattr(self._model.config, "_commit_hash", "") or "")

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        import torch

        if not inputs:
            return []
        batch = self._tokenizer(
            [item.prompt for item in inputs],
            [item.sentence for item in inputs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SCORING_SEQUENCE_LENGTH,
        )
        batch = {k: v.to(self._model.device) for k, v in batch.items()}
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


class MxbaiRerankerScorer(_CudaMeasurement):
    """Apply mxbai-rerank-v2's official binary next-token scoring rule."""

    def __init__(self, tokenizer: Any, model: Any, settings: ScorerSettings) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._settings = settings
        self._yes_id, self._no_id = (
            _single_token_id(tokenizer, "1"),
            _single_token_id(tokenizer, "0"),
        )

    @classmethod
    def load(
        cls, model_id: str, settings: ScorerSettings, revision: str | None = None
    ) -> "MxbaiRerankerScorer":
        import torch

        transformers = _load_transformers()

        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model: Any = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
        )
        model.eval()
        return cls(tokenizer, model, settings)

    @property
    def revision(self) -> str:
        return str(getattr(self._model.config, "_commit_hash", "") or "")

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        import torch

        if not inputs:
            return []
        texts = [self._as_chat(item) for item in inputs]
        batch = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SCORING_SEQUENCE_LENGTH,
            add_special_tokens=False,
        )
        batch = {k: v.to(self._model.device) for k, v in batch.items()}
        with torch.inference_mode():
            logits = self._model(**batch).logits[:, -1, :]
        native = logits[:, self._yes_id].float() - logits[:, self._no_id].float()
        probabilities = torch.sigmoid(native - 4.5)
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

    def __init__(self, agent: Any, settings: ScorerSettings, revision: str) -> None:
        self._agent = agent
        self._settings = settings
        self._revision = revision

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
        return cls(agent, settings, resolved_revision)

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


class GliClassScorer(_CudaMeasurement):
    """Reads a verdict from a zero-shot classifier's score for each label."""

    _LABELS = ("yes", "no")

    def __init__(self, pipeline: Any, revision: str) -> None:
        self._pipeline = pipeline
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
        pipeline = ZeroShotClassificationPipeline(
            model, tokenizer, classification_type="single-label", device="cuda:0"
        )
        resolved = str(getattr(model.config, "_commit_hash", "") or "")
        return cls(pipeline, revision or resolved)

    @property
    def revision(self) -> str:
        return self._revision

    def score(self, inputs: Sequence[ScoringInput]) -> list[LabelScores]:
        if not inputs:
            return []
        results = self._pipeline(
            [item.prompt for item in inputs], list(self._LABELS), threshold=0.0
        )
        return [_as_label_scores(result) for result in results]


def _as_label_scores(result: Any) -> LabelScores:
    scores = {Label(entry["label"]): float(entry["score"]) for entry in result}
    missing = {Label.YES, Label.NO} - set(scores)
    for label in missing:
        scores[label] = 0.0
    return LabelScores(scores)


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
    return 1.0 / (1.0 + exp(-(native_score - 4.5)))


def scorer_class_for(model_id: str) -> type[Any]:
    """Select the adapter matching a rostered scoring checkpoint."""
    adapters: dict[str, type[Any]] = {
        "Alibaba-NLP/gte-multilingual-reranker-base": GteScorer,
        "mixedbread-ai/mxbai-rerank-base-v2": MxbaiRerankerScorer,
        "convaiinnovations/laya-multilingual": LayaScorer,
        "Qwen/Qwen3-Reranker-0.6B": RerankerScorer,
        "Qwen/Qwen3-Reranker-4B": RerankerScorer,
    }
    try:
        return adapters[model_id]
    except KeyError as exc:
        raise ValueError(f"no scoring adapter is registered for {model_id!r}") from exc


def provide_scorer(request: RunRequest) -> tuple[LabelScorer, str]:
    """The default scorer provider used by the CLI."""
    settings = ScorerSettings(dtype=request.dtype)
    scorer = scorer_class_for(request.model_id).load(
        request.model_id, settings, revision=request.revision
    )
    return scorer, request.revision or scorer.revision
