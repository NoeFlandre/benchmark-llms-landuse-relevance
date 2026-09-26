"""Pure selection and comparability rules for benchmark runs."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from landuse_relevance_bench.domain.records import RunResult
from landuse_relevance_bench.domain.roster import TRANSFORMERS, spec_for

_COMPARISON_FIELDS = ("benchmark_sha256", "prompt_sha256", "max_new_tokens", "decoding")


@dataclass(frozen=True, slots=True)
class ComparabilityDifference:
    """One run setting that differs from the named reference run."""

    run_name: str
    field: str
    expected: object
    actual: object


@dataclass(frozen=True, slots=True)
class ComparabilityReport:
    """Whether a group of runs share the settings needed for score comparison."""

    reference_name: str | None
    differences: tuple[ComparabilityDifference, ...]

    @property
    def comparable(self) -> bool:
        return not self.differences

    @property
    def summary(self) -> str:
        if self.comparable:
            return "all runs use the same benchmark, prompt, token budget, and decoding"
        reference = self.reference_name or "reference"
        details = "; ".join(
            f"{difference.run_name}.{difference.field}={difference.actual!r} "
            f"(expected {difference.expected!r} from {reference})"
            for difference in self.differences
        )
        return f"runs are not comparable: {details}"


def check_comparable(results: Sequence[RunResult]) -> ComparabilityReport:
    """Compare every run with a deterministic reference on core experiment settings."""
    if not results:
        return ComparabilityReport(reference_name=None, differences=())
    reference = min(results, key=lambda result: result.metadata.name)
    differences = tuple(
        ComparabilityDifference(
            run_name=result.metadata.name,
            field=field,
            expected=getattr(reference.metadata, field),
            actual=getattr(result.metadata, field),
        )
        for result in sorted(results, key=lambda item: item.metadata.name)
        if result is not reference
        for field in _COMPARISON_FIELDS
        if getattr(result.metadata, field) != getattr(reference.metadata, field)
    )
    return ComparabilityReport(reference_name=reference.metadata.name, differences=differences)


def _runtime_of(name: str) -> str:
    try:
        return spec_for(name).runtime
    except KeyError:
        return TRANSFORMERS


def select_run_names(
    names: Sequence[str], *, only: str | None = None, runtimes: Sequence[str] | None = None
) -> list[str]:
    """Select run names by regex and roster runtime, preserving roster order."""
    try:
        pattern = re.compile(only) if only is not None else None
    except re.error as exc:
        raise ValueError(f"invalid --only regex {only!r}: {exc}") from exc
    return [
        name
        for name in names
        if (pattern is None or pattern.search(name))
        and (not runtimes or _runtime_of(name) in runtimes)
    ]
