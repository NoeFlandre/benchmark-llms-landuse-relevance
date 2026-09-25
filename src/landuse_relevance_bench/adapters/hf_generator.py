"""A deterministic, batched text generator backed by Hugging Face Transformers.

Kept behind :class:`~landuse_relevance_bench.domain.engine.TextGenerator` so the
benchmark itself never imports a model runtime.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.engine import Generation


def is_truncated(
    token_ids: Sequence[int], max_new_tokens: int, stop_token_ids: Collection[int]
) -> bool:
    """Whether a completion used its whole budget without ever stopping on its own.

    A stop token anywhere in the row means the model finished: rows shorter than the
    batch's longest are padded past it, so the final id is padding, not the answer.
    """
    if not token_ids:
        return False
    return len(token_ids) >= max_new_tokens and not any(t in stop_token_ids for t in token_ids)


def generated_length(token_ids: Sequence[int], stop_token_ids: Collection[int]) -> int:
    """Tokens the model actually produced: up to and including its first stop token.

    Anything after that is padding added to match the batch's longest row.
    """
    for position, token in enumerate(token_ids):
        if token in stop_token_ids:
            return position + 1
    return len(token_ids)


def stop_token_ids(generation_config: Any, tokenizer: Any) -> frozenset[int]:
    """The ids that end a completion: the generation config's EOS, else the tokenizer's."""
    raw = (
        generation_config.eos_token_id if generation_config is not None else tokenizer.eos_token_id
    )
    ids = raw if isinstance(raw, list) else [raw]
    return frozenset(i for i in ids if i is not None)


def user_turn(prompt: str, *, vision: bool) -> list[dict[str, Any]]:
    """A single user message; a vision-language template expects typed content parts."""
    content: Any = [{"type": "text", "text": prompt}] if vision else prompt
    return [{"role": "user", "content": content}]


@dataclass(frozen=True, slots=True)
class GeneratorSettings:
    """Decoding settings; greedy throughout, so a run replays exactly."""

    max_new_tokens: int
    dtype: str
    seed: int
    device_map: str = "auto"


class TransformersGenerator:
    """Completes prompts with a chat-templated, greedy, left-padded batch decode."""

    def __init__(self, tokenizer: Any, model: Any, settings: GeneratorSettings) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._settings = settings

    @classmethod
    def load(
        cls, model_id: str, settings: GeneratorSettings, revision: str | None = None
    ) -> "TransformersGenerator":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(settings.seed)
        # The Transformers stubs model a tokenizer as a union that includes None, so the
        # attributes below are all "unresolved" to a type checker. It is a tokenizer.
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
    def _stop_token_ids(self) -> frozenset[int]:
        return stop_token_ids(self._model.generation_config, self._tokenizer)

    @property
    def revision(self) -> str:
        """The resolved weights commit, so a result names the exact artefact used."""
        return str(getattr(self._model.config, "_commit_hash", "") or "")

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        import torch

        if not prompts:
            return []
        texts = [self._as_chat(p) for p in prompts]
        batch = self._tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(self._model.device) for k, v in batch.items()}
        with torch.inference_mode():
            generated = self._model.generate(
                **batch,
                max_new_tokens=self._settings.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        completions = generated[:, batch["input_ids"].shape[1] :]
        return [self._as_generation(row) for row in completions]

    def _as_generation(self, completion: Any) -> Generation:
        ids = completion.tolist()
        return Generation(
            text=self._tokenizer.decode(completion, skip_special_tokens=True).strip(),
            truncated=is_truncated(ids, self._settings.max_new_tokens, self._stop_token_ids),
            generated_tokens=generated_length(ids, self._stop_token_ids),
        )

    def _as_chat(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            user_turn(prompt, vision=False),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


class VisionLanguageGenerator(TransformersGenerator):
    """A vision-language model prompted with text only.

    Loaded through ``AutoProcessor`` and ``AutoModelForImageTextToText`` as its model
    card prescribes; the processor's chat template renders a text-only user turn and
    its tokenizer does the batched, left-padded encoding.
    """

    def __init__(self, processor: Any, model: Any, settings: GeneratorSettings) -> None:
        super().__init__(processor.tokenizer, model, settings)
        self._processor = processor

    @classmethod
    def load(
        cls, model_id: str, settings: GeneratorSettings, revision: str | None = None
    ) -> "VisionLanguageGenerator":
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

        set_seed(settings.seed)
        processor: Any = AutoProcessor.from_pretrained(model_id, revision=revision)
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model: Any = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
        )
        model.eval()
        return cls(processor, model, settings)

    def _as_chat(self, prompt: str) -> str:
        return self._processor.apply_chat_template(
            user_turn(prompt, vision=True),
            tokenize=False,
            add_generation_prompt=True,
        )


def _settings(request: RunRequest) -> GeneratorSettings:
    return GeneratorSettings(
        max_new_tokens=request.max_new_tokens, dtype=request.dtype, seed=request.seed
    )


def provide(request: RunRequest) -> tuple[TransformersGenerator, str]:
    """The Transformers generator provider; vision-language models load their own way."""
    loader = VisionLanguageGenerator if request.vision else TransformersGenerator
    generator = loader.load(
        request.model_id,
        _settings(request),
        revision=request.revision,
    )
    return generator, request.revision or generator.revision
