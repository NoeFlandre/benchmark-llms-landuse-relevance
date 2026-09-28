import pytest

from landuse_relevance_bench.domain.sharding import (
    PairStatus,
    model_language_pairs,
    pair_statuses,
    shard_pairs,
)


def test_model_language_pairs_are_sorted_and_unique() -> None:
    assert model_language_pairs(["b/model", "a/model"], ["fr", "en"]) == (
        ("a/model", "en"),
        ("a/model", "fr"),
        ("b/model", "en"),
        ("b/model", "fr"),
    )


def test_shards_are_disjoint_and_complete() -> None:
    pairs = model_language_pairs(["a/model", "b/model"], ["en", "fr", "de"])
    partitions = [set(shard_pairs(pairs, index, 3)) for index in range(3)]

    assert set.union(*partitions) == set(pairs)
    assert sum(len(partition) for partition in partitions) == len(pairs)
    assert all(
        partitions[left].isdisjoint(partitions[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


@pytest.mark.parametrize(
    ("shard_index", "shard_count"),
    [(-1, 2), (2, 2), (0, 0), (0, -1)],
)
def test_invalid_shard_values_are_rejected(shard_index: int, shard_count: int) -> None:
    with pytest.raises(ValueError, match="shard"):
        shard_pairs((("model", "en"),), shard_index, shard_count)


def test_pair_statuses_are_deterministic() -> None:
    pairs = (("a/model", "en"), ("a/model", "fr"))

    assert pair_statuses(pairs, {pairs[0]}) == (
        PairStatus("a/model", "en", "complete"),
        PairStatus("a/model", "fr", "pending"),
    )
