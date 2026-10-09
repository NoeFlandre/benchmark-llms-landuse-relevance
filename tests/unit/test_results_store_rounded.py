"""``rounded`` keeps the requested digits as a float and passes unavailable metrics through."""

from landuse_relevance_bench.adapters.results_store import rounded


def test_rounded_keeps_the_requested_digits_as_a_float() -> None:
    value = rounded(1.23456, 2)

    assert value == 1.23
    assert isinstance(value, float)


def test_rounded_keeps_an_unavailable_measurement_as_none() -> None:
    assert rounded(None, 2) is None
