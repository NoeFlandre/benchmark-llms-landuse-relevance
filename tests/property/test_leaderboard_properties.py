"""Leaderboard shape and ordering invariants."""

from hypothesis import given
from hypothesis import strategies as st
from strategies import comparable_leaderboards

from landuse_relevance_bench.adapters import results_store
from landuse_relevance_bench.adapters.results_store import leaderboard_rows, rounded


@given(results=comparable_leaderboards())
def test_leaderboard_has_one_descending_f1_row_per_input_and_ignores_input_order(
    results, monkeypatch
) -> None:
    monkeypatch.setattr(
        results_store,
        "bootstrap_interval",
        lambda *_args, **_kwargs: (0.0, 0.0),
    )
    rows = leaderboard_rows(results)
    reversed_rows = leaderboard_rows(list(reversed(results)))
    assert [row["model_id"] for row in rows] == [row["model_id"] for row in reversed_rows]
    assert len(rows) == len(results)
    assert [row["f1"] for row in rows] == sorted((row["f1"] for row in rows), reverse=True)


@given(
    value=st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.none()),
    digits=st.integers(min_value=0, max_value=8),
)
def test_rounding_is_idempotent(value: float | None, digits: int) -> None:
    once = rounded(value, digits)
    assert rounded(once, digits) == once
