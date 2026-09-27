from landuse_relevance_bench.domain.labels import Label


def test_label_stringifies_as_the_bare_token() -> None:
    assert f"{Label.YES}" == "yes"
    assert str(Label.NO) == "no"
