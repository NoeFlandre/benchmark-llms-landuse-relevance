import pytest

from landuse_relevance_bench.domain.labels import Label


def test_label_values_are_lowercase_tokens() -> None:
    assert Label.YES.value == "yes"
    assert Label.NO.value == "no"


def test_label_is_constructible_from_token() -> None:
    assert Label("yes") is Label.YES
    assert Label("no") is Label.NO


def test_label_rejects_unknown_token() -> None:
    with pytest.raises(ValueError):
        Label("maybe")


def test_label_stringifies_as_the_bare_token() -> None:
    assert f"{Label.YES}" == "yes"
    assert str(Label.NO) == "no"
