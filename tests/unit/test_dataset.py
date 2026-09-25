import pytest

from landuse_relevance_bench.domain.dataset import BenchmarkItem, InvalidRowError, build_item

ROW = {
    "sentence": "The island is covered in dense rainforest.",
    "label": "yes",
    "polygon_name": "Rabi",
    "region": "fiji",
    "source": "wikipedia",
}


def test_builds_an_item_from_a_well_formed_row() -> None:
    item = build_item(ROW)
    assert isinstance(item, BenchmarkItem)
    assert item.sentence == ROW["sentence"]
    assert item.label.value == "yes"
    assert item.region == "fiji"
    assert item.polygon_name == "Rabi"
    assert item.source == "wikipedia"


def test_item_id_is_deterministic_and_content_addressed() -> None:
    assert build_item(ROW).item_id == build_item(dict(ROW)).item_id
    other = build_item({**ROW, "sentence": "A different sentence."})
    assert other.item_id != build_item(ROW).item_id


@pytest.mark.parametrize("field", ["sentence", "label"])
def test_rejects_a_row_missing_a_required_field(field: str) -> None:
    row = {k: v for k, v in ROW.items() if k != field}
    with pytest.raises(
        InvalidRowError, match=rf"^benchmark row is missing the required field '{field}'$"
    ):
        build_item(row)


def test_rejects_a_blank_sentence() -> None:
    with pytest.raises(InvalidRowError, match=r"^benchmark row has a blank sentence$"):
        build_item({**ROW, "sentence": "   "})


def test_rejects_an_unknown_label() -> None:
    with pytest.raises(InvalidRowError, match=r"^unknown label 'maybe'; expected 'yes' or 'no'$"):
        build_item({**ROW, "label": "maybe"})


def test_optional_fields_default_to_empty() -> None:
    item = build_item({"sentence": "Trees.", "label": "no"})
    assert item.polygon_name == ""
    assert item.region == ""
    assert item.source == ""
