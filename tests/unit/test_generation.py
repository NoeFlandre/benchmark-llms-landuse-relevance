from landuse_relevance_bench.domain.engine import Generation


def test_a_generation_defaults_to_having_finished() -> None:
    assert Generation("yes").truncated is False


def test_a_generation_can_report_that_it_hit_the_token_budget() -> None:
    assert Generation("1. Analyze the request", truncated=True).truncated is True
