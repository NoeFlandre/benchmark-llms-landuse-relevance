"""Driving a text generator across the benchmark."""

from collections.abc import Iterator, Sequence

from landuse_relevance_bench.domain.dataset import BenchmarkItem
from landuse_relevance_bench.domain.engine import TextGenerator
from landuse_relevance_bench.domain.parsing import parse_label
from landuse_relevance_bench.domain.prompting import render_prompt
from landuse_relevance_bench.domain.records import Prediction

DEFAULT_BATCH_SIZE = 16


def predict_all(
    items: Sequence[BenchmarkItem],
    template: str,
    generator: TextGenerator,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[Prediction, ...]:
    """Run every item through ``generator`` in order and parse each verdict."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    predictions: list[Prediction] = []
    for batch in _batched(items, batch_size):
        outputs = list(generator.generate([render_prompt(template, i.sentence) for i in batch]))
        if len(outputs) != len(batch):
            raise ValueError(f"generator returned {len(outputs)} outputs for {len(batch)} prompts")
        predictions.extend(
            Prediction(
                item_id=item.item_id,
                expected=item.label,
                predicted=parse_label(output),
                raw_output=output,
            )
            for item, output in zip(batch, outputs, strict=True)
        )
    return tuple(predictions)


def _batched(items: Sequence[BenchmarkItem], size: int) -> Iterator[Sequence[BenchmarkItem]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
