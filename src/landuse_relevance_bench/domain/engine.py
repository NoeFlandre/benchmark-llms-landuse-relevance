"""The single seam between the benchmark and any model runtime."""

from collections.abc import Sequence
from typing import Protocol


class TextGenerator(Protocol):
    """Deterministically completes a batch of prompts, one output per prompt."""

    def generate(self, prompts: Sequence[str]) -> Sequence[str]: ...
