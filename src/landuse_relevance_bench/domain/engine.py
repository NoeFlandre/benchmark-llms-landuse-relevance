"""The single seam between the benchmark and any model runtime."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from landuse_relevance_bench.domain.labels import Label


@dataclass(frozen=True, slots=True)
class Generation:
    """One completion, and whether it ran out of budget before finishing.

    A truncated generation carries no verdict even when a ``yes``/``no`` appears in
    it — the model was still mid-thought.
    """

    text: str
    truncated: bool = False

    @classmethod
    def of(cls, output: "str | Generation") -> "Generation":
        """Accept a bare string from a generator that cannot report truncation."""
        return output if isinstance(output, Generation) else cls(output)


class TextGenerator(Protocol):
    """Deterministically completes a batch of prompts, one output per prompt."""

    def generate(self, prompts: Sequence[str]) -> Sequence[Generation | str]: ...


@dataclass(frozen=True, slots=True)
class LabelScores:
    """What a non-generative model assigns to each label for one item.

    The scores are the model's own output, kept verbatim so a different decision
    rule can be recomputed later without re-running anything — the same reason a
    generative run keeps its raw text.
    """

    scores: Mapping[Label, float]

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("a scored item needs at least one label score")

    @property
    def verdict(self) -> Label:
        """The highest-scoring label, ties broken by label order for determinism."""
        return max(sorted(self.scores, key=lambda label: label.value), key=self.scores.__getitem__)


class LabelScorer(Protocol):
    """Scores a batch of rendered prompts against the labels, without generating."""

    def score(self, prompts: Sequence[str]) -> Sequence[LabelScores]: ...
