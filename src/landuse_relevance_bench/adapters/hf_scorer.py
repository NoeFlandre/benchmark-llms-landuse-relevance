"""Scoring the benchmark with models that never generate text.

A reranker is a causal LM whose verdict lives in the logits of the next token, not
in anything it writes: reading the `yes` and `no` logits and normalising them gives
the model's own answer without decoding a single token. A zero-shot classifier
returns a score per candidate label directly.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from landuse_relevance_bench.adapters.hf_scorer_prompt import RERANKER_SYSTEM
from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.engine import LabelScorer, LabelScores
from landuse_relevance_bench.domain.labels import Label


@dataclass(frozen=True, slots=True)
class ScorerSettings:
    """What a scoring run needs; no decoding settings, because nothing is decoded."""

    dtype: str
    device_map: str = "auto"


class RerankerScorer:
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer: Any = AutoTokenizer.from_pretrained(model_id, revision=revision)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model: Any = AutoModelForCausalLM.from_pretrained(
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

    def score(self, prompts: Sequence[str]) -> list[LabelScores]:
        import torch

        if not prompts:
            return []
        texts = [self._as_chat(p) for p in prompts]
        batch = self._tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(self._model.device) for k, v in batch.items()}
        with torch.inference_mode():
            logits = self._model(**batch).logits[:, -1, :]
        pair = torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=-1)
        probabilities = torch.softmax(pair.float(), dim=-1)
        return [
            LabelScores({Label.NO: float(row[0]), Label.YES: float(row[1])})
            for row in probabilities
        ]

    def _as_chat(self, prompt: str) -> str:
        template = getattr(self._tokenizer, "chat_template", None)
        if not template:
            return f"{RERANKER_SYSTEM}\n\n{prompt}"
        return self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": RERANKER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )


class GliClassScorer:
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

        # ty: ignore[unresolved-import]  - optional "scoring" extra, absent from the dev env
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        model = GLiClassModel.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, settings.dtype)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        pipeline = ZeroShotClassificationPipeline(
            model, tokenizer, classification_type="single-label", device="cuda:0"
        )
        resolved = str(getattr(model.config, "_commit_hash", "") or "")
        return cls(pipeline, revision or resolved)

    @property
    def revision(self) -> str:
        return self._revision

    def score(self, prompts: Sequence[str]) -> list[LabelScores]:
        if not prompts:
            return []
        results = self._pipeline(list(prompts), list(self._LABELS), threshold=0.0)
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


def provide_scorer(request: RunRequest) -> tuple[LabelScorer, str]:
    """The default scorer provider used by the CLI."""
    settings = ScorerSettings(dtype=request.dtype)
    scorer = RerankerScorer.load(request.model_id, settings, revision=request.revision)
    return scorer, request.revision or scorer.revision
