import pytest

from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.parsing import parse_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("yes", Label.YES),
        ("no", Label.NO),
        ("YES", Label.YES),
        ("  No  ", Label.NO),
        ("yes.", Label.YES),
        ("**no**", Label.NO),
        ("Answer: yes", Label.YES),
        ("no\nExplanation: none", Label.NO),
        ("<think>hmm</think>yes", Label.YES),
        ('"yes"', Label.YES),
    ],
)
def test_parses_known_shapes(raw: str, expected: Label) -> None:
    assert parse_label(raw) is expected


@pytest.mark.parametrize("raw", ["", "   ", "maybe", "I cannot answer", "yesno", "nonsense"])
def test_returns_none_when_no_decision_token(raw: str) -> None:
    assert parse_label(raw) is None


def test_takes_the_first_decision_token_when_several_appear() -> None:
    assert parse_label("yes, definitely not no") is Label.YES
    assert parse_label("no, it is not yes") is Label.NO


def test_does_not_match_token_inside_a_word() -> None:
    assert parse_label("nostalgia yesterday") is None
