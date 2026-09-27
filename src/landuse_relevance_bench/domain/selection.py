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
    return _check_fields(results, _COMPARISON_FIELDS)


def _check_fields(results: Sequence[RunResult], fields: Sequence[str]) -> ComparabilityReport:
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
        for field in fields
        if getattr(result.metadata, field) != getattr(reference.metadata, field)
    )
    return ComparabilityReport(reference_name=reference.metadata.name, differences=differences)


def collection_comparability_errors(results: Sequence[RunResult]) -> tuple[str, ...]:
    """Check that one report does not combine incompatible benchmark settings.

    Generative methods share one prompt and token budget. Scoring methods may use
    different prompts from each other, but a single named scorer must use the same
    settings across its language runs.
    """
    if not results:
        return ()
    errors = []
    benchmarks_by_language: dict[str, set[str]] = {}
    for result in results:
        benchmarks_by_language.setdefault(result.metadata.language, set()).add(
            result.metadata.benchmark_sha256
        )
    for language, benchmarks in sorted(benchmarks_by_language.items()):
        if len(benchmarks) > 1:
            errors.append(
                f"runs are not comparable: benchmark_sha256 differs for language {language}"
            )

    comparable_fields = tuple(field for field in _COMPARISON_FIELDS if field != "benchmark_sha256")
    generative = [result for result in results if result.metadata.is_generative]
    report = _check_fields(generative, comparable_fields)
    if report.differences:
        errors.append(report.summary)

    scoring_by_name: dict[str, list[RunResult]] = {}
    for result in results:
        if not result.metadata.is_generative:
            scoring_by_name.setdefault(result.metadata.name, []).append(result)
    for name, scorer_runs in sorted(scoring_by_name.items()):
        report = _check_fields(scorer_runs, comparable_fields)
        if report.differences:
            errors.append(f"{name}: {report.summary}")
    return tuple(errors)


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
