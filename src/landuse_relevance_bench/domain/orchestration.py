"""Driving a text generator, or a non-generative scorer, across the benchmark."""

import time
from collections.abc import Callable, Iterator, Sequence

from landuse_relevance_bench.domain.dataset import BenchmarkItem
from landuse_relevance_bench.domain.engine import (
    Generation,
    LabelScorer,
    LabelScores,
    ScoringInput,
    TextGenerator,
)
from landuse_relevance_bench.domain.parsing import parse_with_mode
from landuse_relevance_bench.domain.prompting import render_prompt
from landuse_relevance_bench.domain.records import Prediction

DEFAULT_BATCH_SIZE = 16


def predict_all(
    items: Sequence[BenchmarkItem],
    template: str,
    generator: TextGenerator,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Prediction, ...]:
    """Run every item through ``generator`` and retain parse and speed evidence."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    predictions: list[Prediction] = []
    for batch in _batched(items, batch_size):
        prompts = [render_prompt(template, item.sentence) for item in batch]
        started = clock()
        outputs = list(generator.generate(prompts))
        latency = clock() - started
        if len(outputs) != len(batch):
            raise ValueError(f"generator returned {len(outputs)} outputs for {len(batch)} prompts")
        predictions.extend(
            _predict(item, Generation.of(output), latency)
            for item, output in zip(batch, outputs, strict=True)
        )
    return tuple(predictions)


def score_all(
    items: Sequence[BenchmarkItem],
    template: str,
    scorer: LabelScorer,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[Prediction, ...]:
    """Score every item in order, taking each verdict from the model's own scores.

    A scoring model cannot leave a verdict unparsed or run out of budget, so no
    prediction here is ever truncated or None. That is a property of the method,
    not a perfect score, and the published card says so.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    predictions: list[Prediction] = []
    for batch in _batched(items, batch_size):
        scored = list(
            scorer.score(
                [ScoringInput(render_prompt(template, i.sentence), i.sentence) for i in batch]
            )
        )
        if len(scored) != len(batch):
            raise ValueError(f"scorer returned {len(scored)} scores for {len(batch)} prompts")
        predictions.extend(
            _scored_prediction(item, scored[index]) for index, item in enumerate(batch)
        )
    return tuple(predictions)


def _scored_prediction(item: BenchmarkItem, scores: LabelScores) -> Prediction:
    return Prediction(
        item_id=item.item_id,
        expected=item.label,
        predicted=scores.verdict,
        raw_output=format_scores(scores),
    )


def format_scores(scores: LabelScores) -> str:
    """Render native scores as stable ``label=score`` text, ordered by label.

    Kept as text rather than a serialised structure so the domain stays free of
    serialisation libraries, and readable enough that another decision rule can be
    recomputed from a published result without re-running the model.
    """
    rendered = " ".join(
        f"{label.value}={scores.scores[label]:.6f}"
        for label in sorted(scores.scores)  # StrEnum: sorts by the token itself
    )
    if scores.native_score is not None:
        rendered += f" native={scores.native_score:.6f}"
    return rendered


def _predict(item: BenchmarkItem, generation: Generation, batch_latency: float) -> Prediction:
    parsed = parse_with_mode(generation.text) if not generation.truncated else None
    return Prediction(
        item_id=item.item_id,
        expected=item.label,
        predicted=None if parsed is None else parsed.label,
        raw_output=generation.text,
        truncated=generation.truncated,
        parse_mode=None if parsed is None else parsed.mode,
        latency_seconds=(
            batch_latency if generation.latency_seconds is None else generation.latency_seconds
        ),
        generated_tokens=generation.generated_tokens,
        verify_steps=generation.verify_steps,
        accepted_drafts=generation.accepted_drafts,
        proposed_drafts=generation.proposed_drafts,
    )


def _batched(items: Sequence[BenchmarkItem], size: int) -> Iterator[Sequence[BenchmarkItem]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
