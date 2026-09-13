import pytest

from landuse_relevance_bench.domain.prompting import MissingPlaceholderError, render_prompt

TEMPLATE = "Classify.\n\nTARGET SENTENCE: {}"


def test_substitutes_the_sentence_for_the_placeholder() -> None:
    assert render_prompt(TEMPLATE, "A road crosses the plain.") == (
        "Classify.\n\nTARGET SENTENCE: A road crosses the plain."
    )


def test_rejects_a_template_without_a_placeholder() -> None:
    with pytest.raises(MissingPlaceholderError, match=r"no '\{\}' placeholder"):
        render_prompt("No placeholder here", "sentence")


def test_substitutes_only_the_first_placeholder() -> None:
    assert render_prompt("{} and {}", "x") == "x and {}"


def test_braces_in_the_sentence_are_not_interpreted() -> None:
    assert render_prompt(TEMPLATE, "{not_a_field} {}") == (
        "Classify.\n\nTARGET SENTENCE: {not_a_field} {}"
    )
