"""A deterministic text generator for GGUF quants, backed by llama.cpp.

The Transformers generator cannot run a GGUF quant without dequantizing it, which
would measure a bf16 model with rounding noise rather than quantized inference. This
adapter keeps the quantized kernels, and otherwise mirrors the Transformers path: the
same prompt in a chat turn rendered from the checkpoint's own template with thinking
disabled, greedy decoding, the same token budget, and truncation reported when that
budget runs out.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.engine import Generation
from landuse_relevance_bench.domain.roster import spec_for

LLAMA_CONTEXT_LENGTH = 8192
_LENGTH_STOP = "length"


@dataclass(frozen=True, slots=True)
class LlamaSettings:
    max_new_tokens: int
    seed: int
    context_length: int = LLAMA_CONTEXT_LENGTH


def render_chat(template: str, prompt: str, *, bos_token: str = "", eos_token: str = "") -> str:
    """Render the GGUF's embedded Jinja chat template exactly as Transformers would."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    cast(dict[str, Any], environment.globals)["raise_exception"] = raise_exception
    return environment.from_string(template).render(
        messages=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        bos_token=bos_token,
        eos_token=eos_token,
    )


class LlamaCppGenerator:
    """Completes prompts one at a time with greedy llama.cpp decoding on the GPU."""

    def __init__(self, llama: Any, settings: LlamaSettings, revision: str) -> None:
        self._llama = llama
        self._settings = settings
        self._revision = revision
        metadata = getattr(llama, "metadata", {}) or {}
        self._template = metadata.get("tokenizer.chat_template", "")
        if not self._template:
            raise ValueError("GGUF carries no tokenizer.chat_template; cannot build the turn")
        self._bos = _token_text(llama, metadata.get("tokenizer.ggml.bos_token_id"))
        self._eos = _token_text(llama, metadata.get("tokenizer.ggml.eos_token_id"))

    @classmethod
    def load(
        cls,
        repository: str,
        weights_file: str,
        settings: LlamaSettings,
        revision: str | None = None,
    ) -> "LlamaCppGenerator":
        from huggingface_hub import HfApi, hf_hub_download

        # Imported by name: llama.cpp is installed only in the GGUF job's environment.
        llama_class = __import__("builtins").__import__("llama_cpp").Llama

        resolved = revision or str(HfApi().model_info(repository).sha or "")
        path = hf_hub_download(repository, weights_file, revision=resolved or None)
        llama = llama_class(
            model_path=path,
            n_gpu_layers=-1,
            n_ctx=settings.context_length,
            seed=settings.seed,
            logits_all=False,
            verbose=False,
        )
        return cls(llama, settings, resolved)

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def runtime_dtype(self) -> str:
        """No torch dtype applies; the precision is the recorded quantization label."""
        return "gguf"

    def as_chat(self, prompt: str) -> str:
        return render_chat(self._template, prompt, bos_token=self._bos, eos_token=self._eos)

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        return [self._generate_one(prompt) for prompt in prompts]

    def _generate_one(self, prompt: str) -> Generation:
        self._llama.reset()
        completion = self._llama.create_completion(
            prompt=self.as_chat(prompt),
            max_tokens=self._settings.max_new_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repeat_penalty=1.0,
            seed=self._settings.seed,
        )
        choice = completion["choices"][0]
        return Generation(
            text=str(choice["text"]).strip(),
            truncated=choice.get("finish_reason") == _LENGTH_STOP,
        )


def _token_text(llama: Any, token_id: Any) -> str:
    if token_id is None:
        return ""
    try:
        return llama.detokenize([int(token_id)], special=True).decode("utf-8", "replace")
    except (TypeError, ValueError):
        return ""


def provide(request: RunRequest) -> tuple[LlamaCppGenerator, str]:
    """Load the quant a rostered GGUF model names."""
    spec = spec_for(request.model_id)
    generator = LlamaCppGenerator.load(
        spec.repository,
        spec.weights_file,
        LlamaSettings(max_new_tokens=request.max_new_tokens, seed=request.seed),
        revision=request.revision,
    )
    return generator, generator.revision
