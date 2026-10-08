"""The single seam between the benchmark and any model runtime."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from landuse_relevance_bench.domain.labels import Label


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
    #: The runtime's per-request end-to-end latency, when available.
    latency_seconds: float | None = None

    @classmethod
    def of(cls, output: "str | Generation") -> "Generation":
        """Accept a bare string from a generator that cannot report truncation."""
        return output if isinstance(output, Generation) else cls(output)


class TextGenerator(Protocol):
    """Deterministically completes a batch of prompts, one output per prompt."""

    def generate(self, prompts: Sequence[str]) -> Sequence[Generation | str]: ...


# The optional capabilities below are small, separate protocols: a runtime narrows to the
# ones it actually provides, and a type checker verifies the names the adapters declare.


@runtime_checkable
class Closable(Protocol):
    """A generator that holds resources the run must release."""

    def close(self) -> None: ...


@runtime_checkable
class DraftRevisionReporting(Protocol):
    """A generator that reports the draft-model revision it ran with, if any."""

    @property
    def draft_revision(self) -> str | None: ...


@runtime_checkable
class BatchSizeReporting(Protocol):
    """A generator that runs fewer items per call than the requested batch size."""

    @property
    def effective_batch_size(self) -> int: ...


@runtime_checkable
class RuntimeDtypeReporting(Protocol):
    """A model that reports the dtype its runtime selected."""

    @property
    def runtime_dtype(self) -> object: ...


@dataclass(frozen=True, slots=True)
class ScoringInput:
    """The rendered judgement plus the original sentence being judged."""

    prompt: str
    sentence: str


@dataclass(frozen=True, slots=True)
class LabelScores:
    """What a non-generative model assigns to each label for one item.

    The scores are the model's own output, kept verbatim so a different decision
    rule can be recomputed later without re-running anything — the same reason a
    generative run keeps its raw text.
    """

    scores: Mapping[Label, float]
    native_score: float | None = None

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("a scored item needs at least one label score")

    @property
    def verdict(self) -> Label:
        """The highest-scoring label, ties broken by label order for determinism."""
        return max(sorted(self.scores, key=lambda label: label.value), key=self.scores.__getitem__)


class LabelScorer(Protocol):
    """Scores a batch of rendered prompts against the labels, without generating."""

    def score(self, inputs: Sequence[ScoringInput]) -> Sequence[LabelScores]: ...


@runtime_checkable
class MeasurementStart(Protocol):
    """A scorer that wants to know when a timed scoring pass begins."""

    def begin_measurement(self) -> None: ...


@runtime_checkable
class MeasurementEnd(Protocol):
    """A scorer that wants to know when a timed scoring pass ends."""

    def end_measurement(self) -> None: ...


@runtime_checkable
class PeakVRAMReporting(Protocol):
    """A scorer that reports the peak device memory of its last measured pass."""

    @property
    def peak_vram_bytes(self) -> int | None: ...


@runtime_checkable
class SequenceLengthReporting(Protocol):
    """A scorer that reports the input cap it truncates at; ``None`` when its SDK decides."""

    @property
    def sequence_length(self) -> int | None: ...
