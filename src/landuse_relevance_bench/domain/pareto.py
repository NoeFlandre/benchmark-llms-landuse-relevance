"""Balanced accuracy against model size: the Pareto frontier of the benchmark."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

from landuse_relevance_bench.domain.roster import ROSTER
from landuse_relevance_bench.domain.scorers import SCORER_ROSTER

# A runtime variant serves a checkpoint through another engine or with a draft model
# attached. It has the checkpoint's size, so it would only repeat the same point.
_RUNTIME_VARIANT = re.compile(r"@sglang|\+DSpark")


@dataclass(frozen=True, slots=True)
class ModelPoint:
    """One model's macro metrics, with its total parameter count when the roster has it."""

    model_id: str
    parameters: int | None
    generative: bool
    language_count: int
    n_items_total: int
    balanced_accuracy: float
    f1: float
    mcc: float


@cache
def _parameter_counts() -> dict[str, int]:
    counts = {spec.name: spec.total_parameters for spec in ROSTER}
    counts.update({spec.model_id: spec.total_parameters for spec in SCORER_ROSTER})
    return counts


def parameters_of(model_id: str) -> int | None:
    """The total parameter count the roster records for ``model_id``, if any."""
    return _parameter_counts().get(model_id)


def is_runtime_variant(model_id: str) -> bool:
    """Whether ``model_id`` names an alternative runtime of a checkpoint, not a checkpoint."""
    return _RUNTIME_VARIANT.search(model_id) is not None


def plotted_points(points: Iterable[ModelPoint], language_total: int) -> tuple[ModelPoint, ...]:
    """The points that compare fairly: every language, one checkpoint each, a known size."""
    return tuple(
        point
        for point in points
        if point.parameters is not None
        and point.language_count == language_total
        and not is_runtime_variant(point.model_id)
    )


def pareto_frontier(points: Iterable[ModelPoint]) -> tuple[ModelPoint, ...]:
    """The points that no smaller model matches on balanced accuracy, smallest first.

    Points are read from the smallest up. A point joins the frontier when it beats every
    point read before it. Points without a parameter count are skipped.
    """
    sized = [point for point in points if point.parameters is not None]
    sized.sort(key=lambda point: (point.parameters or 0, -point.balanced_accuracy, point.model_id))
    frontier: list[ModelPoint] = []
    best = float("-inf")
    for point in sized:
        if point.balanced_accuracy > best:
            frontier.append(point)
            best = point.balanced_accuracy
    return tuple(frontier)
