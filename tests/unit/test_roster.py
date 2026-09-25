import pytest

from landuse_relevance_bench.domain.roster import ROSTER, model_ids, spec_for


def test_every_model_id_is_a_namespaced_hub_repository() -> None:
    assert all(spec.model_id.count("/") == 1 for spec in ROSTER)


def test_model_ids_are_unique() -> None:
    assert len(set(model_ids())) == len(ROSTER)


def test_every_model_is_small_enough_to_be_a_little_llm() -> None:
    assert all(spec.total_parameters < 10_000_000_000 for spec in ROSTER)


def test_lookup_returns_the_matching_spec() -> None:
    assert spec_for("LiquidAI/LFM2.5-350M").total_parameters == 354_483_968


def test_lookup_of_an_unknown_model_fails() -> None:
    with pytest.raises(KeyError, match=r"'nobody/nothing' is not in the benchmark roster"):
        spec_for("nobody/nothing")
