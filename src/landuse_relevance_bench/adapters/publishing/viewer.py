"""The multilingual benchmark exported as one CSV that the Hub can preview."""

from collections.abc import Sequence
from pathlib import Path


def write_viewer_dataset(
    data_root: Path,
    output: Path,
    *,
    languages: Sequence[str] | None = None,
) -> Path:
    """Export the validated multilingual benchmark as one Hub-viewable CSV."""
    import csv

    from landuse_relevance_bench.adapters.translations import (
        load_language_benchmark,
        load_manifest,
    )

    viewer_columns = (
        "item_id",
        "source_item_id",
        "language",
        "sentence",
        "label",
        "polygon_name",
        "h3_cell",
        "latitude",
        "longitude",
        "source",
        "region",
        "source_url",
    )
    manifest = load_manifest(data_root)
    selected = tuple(sorted(set(languages or manifest.languages)))
    unknown = sorted(set(selected) - set(manifest.languages))
    if unknown:
        raise ValueError(f"unknown viewer language(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("viewer dataset requires at least one language")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(viewer_columns), lineterminator="\n")
        writer.writeheader()
        for language in selected:
            items = load_language_benchmark(data_root, language)
            source_path = data_root / manifest.files[language].path
            with source_path.open(newline="", encoding="utf-8") as source_handle:
                rows = list(csv.DictReader(source_handle))
            if len(rows) != len(items):
                raise ValueError(
                    f"viewer export row mismatch for {language}: {len(rows)} != {len(items)}"
                )
            for item, source_row in zip(items, rows, strict=True):
                row = {column: source_row.get(column, "") or "" for column in viewer_columns}
                row.update(
                    item_id=item.item_id,
                    source_item_id=item.source_item_id,
                    language=item.language,
                    sentence=item.sentence,
                    label=item.label.value,
                )
                writer.writerow(row)
    return output
