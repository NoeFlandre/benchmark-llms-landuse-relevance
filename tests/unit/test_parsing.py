import pytest

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.parsing import parse_label, parse_with_mode


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("yes", Label.YES),
        ("no", Label.NO),
        ("no\n", Label.NO),
        ("YES", Label.YES),
        ("  No  ", Label.NO),
        ("yes.", Label.YES),
        ("**no**", Label.NO),
        ("Answer: yes", Label.YES),
        ("<think>hmm</think>yes", Label.YES),
        ('"yes"', Label.YES),
    ],
)
def test_parses_known_shapes(raw: str, expected: Label) -> None:
    assert parse_label(raw) is expected


@pytest.mark.parametrize("raw", ["", "   ", "maybe", "I cannot answer", "yesno", "nonsense"])
def test_returns_none_when_no_decision_token(raw: str) -> None:
    assert parse_label(raw) is None


def test_the_verdict_is_the_last_decision_token_not_the_first() -> None:
    """A reasoning trace restates both options before committing to one."""
    reasoning = (
        'Criteria for "yes": vegetation, terrain, buildings.\n'
        'Criteria for "no": history, administration.\n'
        "The sentence describes an election.\n"
        "Answer: no"
    )
    assert parse_label(reasoning) is Label.NO


def test_does_not_match_a_token_inside_a_word() -> None:
    assert parse_label("nostalgia yesterday") is None
    assert parse_label("yes, not nonsense") is Label.YES


def test_trailing_prose_after_the_verdict_does_not_flip_it() -> None:
    assert parse_label("no\nExplanation: none") is Label.NO


def test_leading_answer_wins_before_later_mentions_of_the_other_label() -> None:
    parsed = parse_with_mode("Yes — the prompt mentions no, but the answer is yes.")
    assert (parsed.label, parsed.mode) == (Label.YES, "leading")


def test_reasoning_first_output_uses_the_last_label_as_a_fallback() -> None:
    parsed = parse_with_mode("Criteria for yes: woods. Criteria for no: history. Answer: no")
    assert (parsed.label, parsed.mode) == (Label.NO, "last")


def test_a_single_answer_with_wrapping_punctuation_is_exact() -> None:
    parsed = parse_with_mode(" **NO** ")
    assert (parsed.label, parsed.mode) == (Label.NO, "exact")


def test_unparsed_output_has_no_parse_mode() -> None:
    parsed = parse_with_mode("perhaps")
    assert (parsed.label, parsed.mode) == (None, None)
