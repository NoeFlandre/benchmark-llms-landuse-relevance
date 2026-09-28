"""The exact turn a reranker is given, built by hand rather than by a chat template."""

from landuse_relevance_bench.adapters.hf_scorer_prompt import (
    RERANKER_SYSTEM,
    reranker_input,
)
from landuse_relevance_bench.domain.prompting import render_prompt

TEMPLATE = "<Instruct>: judge it\n<Query>: is it relevant?\n<Document>: {}\n"


def test_the_rendered_body_survives_into_the_model_input() -> None:
    sentence = "The village is ringed by terraced farmland."
    text = reranker_input(render_prompt(TEMPLATE, sentence))

    assert sentence in text
    assert "<Instruct>: judge it" in text
    assert "<Query>: is it relevant?" in text


def test_two_different_sentences_produce_two_different_inputs() -> None:
    first = reranker_input(render_prompt(TEMPLATE, "dense pine forest"))
    second = reranker_input(render_prompt(TEMPLATE, "a treaty signed in 1815"))

    assert first != second


def test_the_turn_carries_the_system_message_and_a_closed_thinking_block() -> None:
    text = reranker_input(render_prompt(TEMPLATE, "anything"))

    assert RERANKER_SYSTEM in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
