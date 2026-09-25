"""Greedy generation through SGLang's offline engine, optionally with a speculative draft.

This is the runtime the LFM2.5-VL-3B-DSpark model card prescribes: the target is
launched with the draft attached (``speculative_algorithm="DSPARK"``), and the same
engine without the ``speculative_*`` arguments is the like-for-like baseline. SGLang
is an optional extra imported only when a model is loaded. See ADR-0007.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from landuse_relevance_bench.adapters.hf_generator import user_turn
from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.engine import Generation


class Engine(Protocol):
    """The slice of ``sglang.Engine`` this project depends on."""

    def generate(self, *, input_ids: list[list[int]], sampling_params: dict[str, Any]) -> Any: ...
    def shutdown(self) -> None: ...


def engine_arguments(request: RunRequest) -> dict[str, Any]:
    """``sglang.Engine`` keyword arguments for ``request``, mirroring the card's CLI flags."""
    arguments: dict[str, Any] = {
        "model_path": request.model_id,
        "dtype": request.dtype,
        "random_seed": request.seed,
        **dict(request.speculative),
    }
    if request.revision:
        arguments["revision"] = request.revision
    if request.draft_model_id:
        arguments["speculative_draft_model_path"] = request.draft_model_id
        if request.draft_revision:
            arguments["speculative_draft_model_revision"] = request.draft_revision
    return arguments


def as_generation(output: Mapping[str, Any]) -> Generation:
    """Read one SGLang completion: its text, whether it hit the budget, its token counts.

    ``spec_verify_ct`` counts target verification passes and, like the draft-token
    counts, is only reported when a draft is attached; generated tokens per pass is
    the mean accept length SGLang itself reports as ``spec_accept_length``.
    """
    meta = output.get("meta_info", {})
    finish = meta.get("finish_reason") or {}
    steps = meta.get("spec_verify_ct")
    return Generation(
        text=str(output["text"]).strip(),
        truncated=finish.get("type") == "length",
        generated_tokens=meta.get("completion_tokens"),
        verify_steps=int(steps) if steps else None,
        accepted_drafts=meta.get("spec_num_correct_drafts"),
        proposed_drafts=meta.get("spec_num_proposed_drafts"),
    )


class SGLangGenerator:
    """Completes chat-templated prompts with greedy decoding on an SGLang engine."""

    def __init__(
        self, engine: Engine, encode: Callable[[str], list[int]], max_new_tokens: int
    ) -> None:
        self._engine = engine
        self._encode = encode
        self._max_new_tokens = max_new_tokens

    @classmethod
    def load(cls, request: RunRequest) -> "SGLangGenerator":
        import sglang  # ty: ignore[unresolved-import]  # the `speculative` extra

        encode = _chat_encoder(request)
        engine = sglang.Engine(**engine_arguments(request))
        return cls(engine, encode, request.max_new_tokens)

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        if not prompts:
            return []
        outputs = self._engine.generate(
            input_ids=[self._encode(p) for p in prompts],
            sampling_params={"temperature": 0.0, "max_new_tokens": self._max_new_tokens},
        )
        return [as_generation(o) for o in outputs]

    def close(self) -> None:
        """Stop the engine's scheduler processes and free the GPU for the next run."""
        self._engine.shutdown()


def _chat_encoder(request: RunRequest) -> Callable[[str], list[int]]:
    """The same chat template and encoding the Transformers runs use.

    Token ids go to the engine directly, so SGLang cannot add a second BOS to a
    template that already starts with one.
    """
    from transformers import AutoProcessor, AutoTokenizer

    loader: Any = AutoProcessor if request.vision else AutoTokenizer
    template: Any = loader.from_pretrained(request.model_id, revision=request.revision)
    tokenizer: Any = template.tokenizer if request.vision else template
    extra = {} if request.vision else {"enable_thinking": False}

    def encode(prompt: str) -> list[int]:
        text = template.apply_chat_template(
            user_turn(prompt, vision=request.vision),
            tokenize=False,
            add_generation_prompt=True,
            **extra,
        )
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    return encode


def provide(request: RunRequest) -> tuple[SGLangGenerator, str]:
    return SGLangGenerator.load(request), request.revision or ""
