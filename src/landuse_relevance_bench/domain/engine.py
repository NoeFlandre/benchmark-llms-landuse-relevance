"""The single seam between the benchmark and any model runtime."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Generation:
    """One completion, and whether it ran out of budget before finishing.

    A truncated generation carries no verdict even when a ``yes``/``no`` appears in
    it — the model was still mid-thought.
    """

    text: str
    truncated: bool = False
    #: Tokens the model produced for this completion, when the runtime can count them.
    generated_tokens: int | None = None
    #: Target verification passes under speculative decoding; ``None`` without a draft.
    verify_steps: int | None = None
    #: Draft tokens the target accepted, and the draft tokens proposed, when reported.
    accepted_drafts: int | None = None
    proposed_drafts: int | None = None

    @classmethod
    def of(cls, output: "str | Generation") -> "Generation":
        """Accept a bare string from a generator that cannot report truncation."""
        return output if isinstance(output, Generation) else cls(output)


class TextGenerator(Protocol):
    """Deterministically completes a batch of prompts, one output per prompt."""

    def generate(self, prompts: Sequence[str]) -> Sequence[Generation | str]: ...
