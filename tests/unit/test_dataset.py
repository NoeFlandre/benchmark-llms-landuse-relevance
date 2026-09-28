import pytest

from landuse_relevance_bench.domain.dataset import (
    BenchmarkItem,
    InvalidRowError,
    build_item,
    item_id_for,
)

ROW = {
    "sentence": "The island is covered in dense rainforest.",
    "label": "yes",
    "polygon_name": "Rabi",
    "region": "fiji",
    "source": "wikipedia",
    "source_item_id": "source-42",
    "language": "en",
}


def test_builds_an_item_from_a_well_formed_row() -> None:
    item = build_item(ROW)
    assert isinstance(item, BenchmarkItem)
    assert item.sentence == ROW["sentence"]
    assert item.label.value == "yes"
    assert item.region == "fiji"
    assert item.polygon_name == "Rabi"
    assert item.source == "wikipedia"


def test_item_id_is_deterministic_and_source_addressed() -> None:
    assert build_item(ROW).item_id == build_item(dict(ROW)).item_id
    other = build_item({**ROW, "source_item_id": "source-43"})
    assert other.item_id != build_item(ROW).item_id
    assert build_item(ROW).item_id == item_id_for("source-42", "en")


def test_same_source_item_gets_different_item_ids_per_language() -> None:
    french = build_item({**ROW, "language": "fr", "sentence": "La forêt est dense."})
    german = build_item({**ROW, "language": "de", "sentence": "Der Wald ist dicht."})

    assert french.source_item_id == german.source_item_id == "source-42"
    assert french.item_id != german.item_id


def test_build_item_keeps_source_identity_and_language() -> None:
    item = build_item(ROW)

    assert item.source_item_id == "source-42"
    assert item.language == "en"


def test_item_id_is_a_short_hex_digest() -> None:
    item_id = build_item(ROW).item_id
    assert len(item_id) == 16
    assert all(c in "0123456789abcdef" for c in item_id)


@pytest.mark.parametrize("field", ["sentence", "label", "source_item_id", "language"])
def test_rejects_a_row_missing_a_required_field(field: str) -> None:
    row = {k: v for k, v in ROW.items() if k != field}
    with pytest.raises(
        InvalidRowError, match=rf"^benchmark row is missing the required field '{field}'$"
    ):
        build_item(row)


def test_rejects_a_blank_sentence() -> None:
    with pytest.raises(InvalidRowError, match=r"^benchmark row has a blank sentence$"):
        build_item({**ROW, "sentence": "   "})


@pytest.mark.parametrize("field", ["source_item_id", "language"])
def test_rejects_a_blank_identity_field(field: str) -> None:
    with pytest.raises(InvalidRowError, match=rf"^benchmark row has a blank {field}$"):
        build_item({**ROW, field: "   "})


def test_rejects_an_unknown_label() -> None:
    with pytest.raises(InvalidRowError, match=r"^unknown label 'maybe'; expected 'yes' or 'no'$"):
        build_item({**ROW, "label": "maybe"})


def test_optional_fields_default_to_empty() -> None:
    item = build_item(
        {"sentence": "Trees.", "label": "no", "source_item_id": "source-1", "language": "en"}
    )
    assert item.polygon_name == ""
    assert item.region == ""
    assert item.source == ""
