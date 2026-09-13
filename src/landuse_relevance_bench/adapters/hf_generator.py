"""A deterministic, batched text generator backed by Hugging Face Transformers.

Kept behind :class:`~landuse_relevance_bench.domain.engine.TextGenerator` so the
benchmark itself never imports a model runtime.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from landuse_relevance_bench.adapters.pipeline import RunRequest


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
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
        )
        model.eval()
        return cls(tokenizer, model, settings)

    @property
    def revision(self) -> str:
        """The resolved weights commit, so a result names the exact artefact used."""
        return str(getattr(self._model.config, "_commit_hash", "") or "")

    def generate(self, prompts: Sequence[str]) -> list[str]:
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
        return [
            self._tokenizer.decode(row, skip_special_tokens=True).strip() for row in completions
        ]

    def _as_chat(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


def provide(request: RunRequest) -> tuple[TransformersGenerator, str]:
    """The default generator provider used by the CLI."""
    generator = TransformersGenerator.load(
        request.model_id,
        GeneratorSettings(
            max_new_tokens=request.max_new_tokens, dtype=request.dtype, seed=request.seed
        ),
        revision=request.revision,
    )
    return generator, request.revision or generator.revision
