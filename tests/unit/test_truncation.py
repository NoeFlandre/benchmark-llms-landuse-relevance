"""Deciding whether a completion ran out of budget, independent of any tensor."""

from landuse_relevance_bench.adapters.hf_generator import is_truncated

STOP = frozenset({2})


def test_a_completion_that_reached_a_stop_token_has_finished() -> None:
    assert is_truncated([7, 8, 2], max_new_tokens=3, stop_token_ids=STOP) is False


def test_a_completion_that_used_the_whole_budget_without_stopping_is_truncated() -> None:
    assert is_truncated([7, 8, 9], max_new_tokens=3, stop_token_ids=STOP) is True


def test_a_short_completion_has_finished_even_without_a_stop_token() -> None:
    assert is_truncated([7], max_new_tokens=3, stop_token_ids=STOP) is False


def test_batch_padding_past_a_stop_token_is_not_mistaken_for_truncation() -> None:
    """Short rows are padded to the longest row in the batch, so the last id is padding."""
    assert is_truncated([7, 2, 0, 0], max_new_tokens=4, stop_token_ids=STOP) is False


def test_a_model_with_several_stop_tokens_is_honoured() -> None:
    assert is_truncated([7, 5], max_new_tokens=2, stop_token_ids=frozenset({2, 5})) is False


def test_an_empty_completion_has_not_been_truncated() -> None:
    assert is_truncated([], max_new_tokens=8, stop_token_ids=STOP) is False
