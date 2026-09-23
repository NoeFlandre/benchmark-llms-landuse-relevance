"""A deterministic, batched text generator backed by Hugging Face Transformers.

Kept behind :class:`~landuse_relevance_bench.domain.engine.TextGenerator` so the
benchmark itself never imports a model runtime.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.engine import Generation
from landuse_relevance_bench.domain.roster import quantization_of


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


def _auto_class(model_id: str, revision: str | None) -> Any:
    """Pick the loader a checkpoint declares, not the one most checkpoints happen to use.

    Some rostered checkpoints are vision-language models whose architecture is a
    conditional-generation class (Qwen3.5, for example). ``AutoModelForCausalLM``
    cannot load those, so read the config and choose. The benchmark only ever sends
    text, so a vision-language model is prompted through its text tower.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    architectures = getattr(config, "architectures", None) or []
    vision = getattr(config, "vision_config", None) is not None
    if vision or any(name.endswith("ForConditionalGeneration") for name in architectures):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


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
        from transformers import AutoTokenizer, set_seed

        set_seed(settings.seed)
        # The Transformers stubs model a tokenizer as a union that includes None, so the
        # attributes below are all "unresolved" to a type checker. It is a tokenizer.
        tokenizer: Any = AutoTokenizer.from_pretrained(model_id, revision=revision)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model: Any = _auto_class(model_id, revision).from_pretrained(
            model_id,
            revision=revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
        )
        model.eval()
        return cls(tokenizer, model, settings)

    @property
    def _stop_token_ids(self) -> frozenset[int]:
        config = self._model.generation_config
        raw = config.eos_token_id if config is not None else self._tokenizer.eos_token_id
        ids = raw if isinstance(raw, list) else [raw]
        return frozenset(i for i in ids if i is not None)

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
        return Generation(
            text=self._tokenizer.decode(completion, skip_special_tokens=True).strip(),
            truncated=is_truncated(
                completion.tolist(), self._settings.max_new_tokens, self._stop_token_ids
            ),
        )

    def _as_chat(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


def provide(request: RunRequest) -> tuple[Any, str]:
    """The default generator provider used by the CLI.

    A rostered GGUF quant is dispatched to llama.cpp; everything else runs here.
    """
    if quantization_of(request.model_id):
        from landuse_relevance_bench.adapters.llama_generator import provide as provide_gguf

        return provide_gguf(request)
    generator = TransformersGenerator.load(
        request.model_id,
        GeneratorSettings(
            max_new_tokens=request.max_new_tokens, dtype=request.dtype, seed=request.seed
        ),
        revision=request.revision,
    )
    return generator, request.revision or generator.revision
