"""Dependency-free uncertainty summaries for small paired classification runs."""

import random
from collections.abc import Sequence
from math import comb, sqrt
from typing import Literal

from landuse_relevance_bench.domain.metrics import Outcome, evaluate
from landuse_relevance_bench.domain.records import Prediction

Interval = tuple[float, float]
_Z_95 = 1.959963984540054
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000


def wilson_interval(successes: int, total: int, *, z: float = _Z_95) -> Interval | None:
    """The two-sided Wilson score interval for a binomial proportion."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must describe a valid binomial count")
    if total == 0:
        return None
    proportion = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = (proportion + z2 / (2 * total)) / denominator
    radius = z * sqrt(proportion * (1 - proportion) / total + z2 / (4 * total * total))
    return (max(0.0, centre - radius / denominator), min(1.0, centre + radius / denominator))


def bootstrap_interval(
    outcomes: Sequence[Outcome],
    metric: Literal["f1", "matthews_corrcoef"],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> Interval:
    """Percentile 95% bootstrap interval over items, using a reproducible seed."""
    if not outcomes:
        raise ValueError("cannot bootstrap an empty set of outcomes")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    rng = random.Random(seed)  # noqa: S311 -- seeded resampling does not need cryptographic entropy
    size = len(outcomes)
    scores = sorted(
        getattr(evaluate([outcomes[rng.randrange(size)] for _ in range(size)]), metric)
        for _ in range(resamples)
    )
    return (_quantile(scores, 0.025), _quantile(scores, 0.975))


def exact_mcnemar_p_value(first_only_correct: int, second_only_correct: int) -> float:
    """Two-sided exact McNemar p-value from the discordant-pair binomial tail."""
    if first_only_correct < 0 or second_only_correct < 0:
        raise ValueError("discordant pair counts cannot be negative")
    discordant = first_only_correct + second_only_correct
    if discordant == 0:
        return 1.0
    smaller = min(first_only_correct, second_only_correct)
    tail = sum(comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_mcnemar_p_value(
    predictions: Sequence[Prediction], baseline: Sequence[Prediction]
) -> float | None:
    """Compare correctness only when the same complete item set is present."""
    ours = {prediction.item_id: prediction for prediction in predictions}
    theirs = {prediction.item_id: prediction for prediction in baseline}
    if not _same_nonempty_item_set(ours, theirs):
        return None
    if not _matching_expected_labels(ours, theirs):
        return None
    first_only, second_only = _discordant_correctness(ours, theirs)
    return exact_mcnemar_p_value(first_only, second_only)


def _same_nonempty_item_set(ours: dict[str, Prediction], theirs: dict[str, Prediction]) -> bool:
    return bool(ours) and ours.keys() == theirs.keys()


def _matching_expected_labels(ours: dict[str, Prediction], theirs: dict[str, Prediction]) -> bool:
    return all(ours[item_id].expected == theirs[item_id].expected for item_id in ours)


def _discordant_correctness(
    ours: dict[str, Prediction], theirs: dict[str, Prediction]
) -> tuple[int, int]:
    first_only = second_only = 0
    for item_id, prediction in ours.items():
        reference = theirs[item_id]
        ours_correct = prediction.predicted is prediction.expected
        theirs_correct = reference.predicted is reference.expected
        first_only += ours_correct and not theirs_correct
        second_only += theirs_correct and not ours_correct
    return first_only, second_only


def interval_text(interval: Interval | None) -> str | None:
    """Render a compact interval consistently in CSV and Markdown tables."""
    return None if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
