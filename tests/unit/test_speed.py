from dataclasses import dataclass

import pytest

from landuse_relevance_bench.domain.speed import percentile, summarise_speed


@dataclass
class Timed:
    latency_seconds: float | None = None
    generated_tokens: int | None = None
    verify_steps: int | None = None
    accepted_drafts: int | None = None
    proposed_drafts: int | None = None


@pytest.mark.parametrize(
    ("fraction", "expected"), [(0.0, 1.0), (0.5, 2.5), (0.95, 3.85), (1.0, 4.0)]
)
def test_percentile_interpolates_between_ranks(fraction: float, expected: float) -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], fraction) == pytest.approx(expected)


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"^percentile of an empty sequence$"):
        percentile([], 0.5)


def test_a_fully_recorded_run_reports_every_figure() -> None:
    speed = summarise_speed(
        [Timed(1.0, 30, 10, 12, 80), Timed(3.0, 10, 5, 3, 40)], wall_seconds=4.0
    )
    assert speed.sentences_per_second == 0.5
    assert speed.latency_mean_seconds == 2.0
    assert speed.latency_p50_seconds == 2.0
    assert speed.latency_p95_seconds == pytest.approx(2.9)
    assert speed.generated_tokens == 40
    assert speed.output_tokens_per_second == 10.0
    assert speed.mean_accept_length == pytest.approx(40 / 15)
    assert speed.draft_accept_rate == pytest.approx(15 / 120)


def test_a_figure_missing_from_any_prediction_is_not_reported() -> None:
    speed = summarise_speed([Timed(1.0, 30), Timed(1.0, None)], wall_seconds=2.0)
    assert speed.generated_tokens is None
    assert speed.output_tokens_per_second is None
    assert speed.mean_accept_length is None
    assert speed.latency_p95_seconds == 1.0


def test_a_run_without_timings_still_reports_wall_time_throughput() -> None:
    speed = summarise_speed([Timed(), Timed()], wall_seconds=0.0)
    assert speed.n_items == 2
    assert speed.sentences_per_second is None
    assert speed.latency_mean_seconds is None


def test_a_single_verification_pass_still_yields_an_accept_length() -> None:
    speed = summarise_speed([Timed(1.0, 7, 1, 6, 8)], wall_seconds=1.0)
    assert speed.mean_accept_length == 7.0
