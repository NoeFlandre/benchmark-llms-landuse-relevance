"""A deterministic, batched text generator backed by Hugging Face Transformers.

Kept behind :class:`~landuse_relevance_bench.domain.engine.TextGenerator` so the
benchmark itself never imports a model runtime.
"""

import gc
from collections.abc import Collection, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.adapters.revision import resolve_revision
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


def chat_template_kwargs(*, vision: bool) -> dict[str, Any]:
    """Extra chat-template options: text models are asked not to emit a thinking block."""
    return {} if vision else {"enable_thinking": False}


@dataclass(frozen=True, slots=True)
class GeneratorSettings:
    """Decoding settings; greedy throughout, so a run replays exactly."""

    max_new_tokens: int
    dtype: str
    seed: int
    device_map: str = "auto"
    continuous_batching: bool = False


def _prepare_tokenizer(tokenizer: Any) -> None:
    """Left-pad for batched decoding, padding with EOS when no pad token is defined."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def _load_model(
    auto_cls: Any, model_id: str, settings: GeneratorSettings, revision: str | None
) -> Any:
    import torch

    model: Any = auto_cls.from_pretrained(
        model_id,
        revision=revision,
        dtype=getattr(torch, settings.dtype),
        device_map=settings.device_map,
    )
    model.eval()
    return model


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
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(settings.seed)
        # The Transformers stubs model a tokenizer as a union that includes None, so the
        # attributes below are all "unresolved" to a type checker. It is a tokenizer.
        tokenizer: Any = AutoTokenizer.from_pretrained(model_id, revision=revision)
        _prepare_tokenizer(tokenizer)
        model = _load_model(AutoModelForCausalLM, model_id, settings, revision)
        return cls(tokenizer, model, settings)

    @property
    def _stop_token_ids(self) -> frozenset[int]:
        return stop_token_ids(self._require_model().generation_config, self._tokenizer)

    @property
    def revision(self) -> str:
        """The resolved weights commit, so a result names the exact artefact used."""
        return str(getattr(self._require_model().config, "_commit_hash", "") or "")

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        if not prompts:
            return []
        if self._settings.continuous_batching:
            return self._generate_continuously(prompts)
        import torch

        model = self._require_model()
        texts = [self._as_chat(p) for p in prompts]
        batch = self._tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.inference_mode():
            generated = model.generate(
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

    def _generate_continuously(self, prompts: Sequence[str]) -> list[Generation]:
        model = self._require_model()
        texts = [self._as_chat(prompt) for prompt in prompts]
        inputs = [self._tokenizer.encode(text, add_special_tokens=False) for text in texts]
        generation_config = deepcopy(model.generation_config)
        generation_config.max_new_tokens = self._settings.max_new_tokens
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.pad_token_id = self._tokenizer.pad_token_id
        outputs = model.generate_batch(
            inputs=inputs,
            generation_config=generation_config,
            progress_bar=False,
            warmup=True,
        )
        if len(outputs) != len(prompts):
            raise ValueError(
                f"continuous batching returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        generations = []
        for output in outputs.values():
            if output.error is not None:
                raise RuntimeError(
                    f"continuous batch request {output.request_id} failed: {output.error}"
                )
            _start, end = output.lifespan
            latency = end - output.created_time if end >= output.created_time else None
            generations.append(
                self._as_generation(output.generated_tokens, latency_seconds=latency)
            )
        return generations

    def close(self) -> None:
        """Release model references and return unused CUDA cache blocks to the driver."""
        self._model = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_model(self) -> Any:
        model = self._model
        if model is None:
            raise RuntimeError("generator is closed; load a model before generating")
        return model

    def _as_generation(
        self, completion: Any, *, latency_seconds: float | None = None
    ) -> Generation:
        ids = (
            completion.tolist()
            if callable(getattr(completion, "tolist", None))
            else list(completion)
        )
        return Generation(
            text=self._tokenizer.decode(ids, skip_special_tokens=True).strip(),
            truncated=is_truncated(ids, self._settings.max_new_tokens, self._stop_token_ids),
            generated_tokens=generated_length(ids, self._stop_token_ids),
            latency_seconds=latency_seconds,
        )

    def _as_chat(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            user_turn(prompt, vision=False),
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs(vision=False),
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
        from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

        set_seed(settings.seed)
        processor: Any = AutoProcessor.from_pretrained(model_id, revision=revision)
        _prepare_tokenizer(processor.tokenizer)
        model = _load_model(AutoModelForImageTextToText, model_id, settings, revision)
        return cls(processor, model, settings)

    def _as_chat(self, prompt: str) -> str:
        return self._processor.apply_chat_template(
            user_turn(prompt, vision=True),
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs(vision=True),
        )


def _settings(request: RunRequest) -> GeneratorSettings:
    return GeneratorSettings(
        max_new_tokens=request.max_new_tokens,
        dtype=request.dtype,
        seed=request.seed,
        continuous_batching=request.continuous_batching,
    )


def provide(request: RunRequest) -> tuple[TransformersGenerator, str]:
    """The Transformers generator provider; vision-language models load their own way."""
    if request.continuous_batching and request.vision:
        raise ValueError(
            "continuous batching does not support the vision-language Transformers path"
        )
    loader = VisionLanguageGenerator if request.vision else TransformersGenerator
    generator = loader.load(
        request.model_id,
        _settings(request),
        revision=request.revision,
    )
    return generator, resolve_revision(request.model_id, request.revision, generator.revision)
