"""Byte-for-byte golden cards, so card composition can be refactored mechanically.

Regenerate a golden file only for an intended change to the published card.
"""

from dataclasses import replace
from pathlib import Path

from tests.unit.test_hf_publish import (
    PROMPT,
    PROMPT_SHA256,
    SCORER_PROMPT,
    _result_with_outcomes,
    _scoring_result,
)

from landuse_relevance_bench.adapters.hf_publish import CardOptions, dataset_card
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import read_runs
from landuse_relevance_bench.domain.labels import Label
from landuse_relevance_bench.domain.records import RunResult

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden"


def _assert_golden(card: str, name: str) -> None:
    assert card == (GOLDEN / name).read_text(encoding="utf-8")


def test_the_card_of_the_committed_archive_is_pinned() -> None:
    card = dataset_card(
        read_runs(ROOT / "results", recursive=True),
        benchmark_name="benchmark.csv",
        prompt_text=load_prompt(ROOT / "data" / "prompt.txt"),
    )
    _assert_golden(card, "dataset_card_archive.md")


def _mixed_card() -> str:
    outcomes = (
        (Label.YES, Label.YES),
        (Label.YES, Label.NO),
        (Label.NO, Label.NO),
        (Label.NO, Label.YES),
    )
    return dataset_card(
        [
            _result_with_outcomes("gen/one", outcomes),
            _result_with_outcomes("gen/two", outcomes[:2] + outcomes[2:][::-1]),
            _scoring_result("score/best", (0.9, 0.8, 0.2, 0.1), vram_gib=1),
            _scoring_result("score/second", (0.9, 0.4, 0.7, 0.2), vram_gib=2),
        ],
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        options=CardOptions(scorer_prompt_text=SCORER_PROMPT, viewer_file="data/viewer.csv"),
    )


def test_the_card_with_generative_and_scoring_runs_is_pinned() -> None:
    _assert_golden(_mixed_card(), "dataset_card_mixed.md")


OUTCOMES = (
    (Label.YES, Label.YES),
    (Label.YES, Label.NO),
    (Label.NO, Label.NO),
    (Label.NO, Label.YES),
)
LANGUAGES = ("en", "fr")


def _per_language(result: RunResult, **metadata: object) -> list[RunResult]:
    return [
        replace(result, metadata=replace(result.metadata, language=language, **metadata))
        for language in LANGUAGES
    ]


def _timed(result: RunResult, latency_seconds: float) -> RunResult:
    return replace(
        result,
        predictions=tuple(
            replace(prediction, latency_seconds=latency_seconds, generated_tokens=4)
            for prediction in result.predictions
        ),
    )


def _full_card() -> str:
    """A card that exercises every optional section the renderers can emit."""
    llm = "LiquidAI/LFM2.5-2.6B"
    vl = "LiquidAI/LFM2.5-VL-3B"
    results: list[RunResult] = [
        *_per_language(_result_with_outcomes(llm, OUTCOMES), duration_seconds=3600.0),
        *(
            _timed(run, 0.5)
            for run in _per_language(
                _result_with_outcomes(vl, OUTCOMES),
                run_id=f"{vl}@sglang-throughput-b16",
                runtime="sglang",
                device_name="NVIDIA A100",
            )
        ),
        *(
            _timed(run, 0.25)
            for run in _per_language(
                _result_with_outcomes(vl, OUTCOMES),
                run_id=f"{vl}+DSpark-throughput-b16",
                runtime="sglang",
                device_name="NVIDIA A100",
                draft_model_id="LiquidAI/LFM2.5-350M",
            )
        ),
        *_per_language(
            _result_with_outcomes("unsloth/Big-GGUF@IQ2", OUTCOMES),
            dtype="gguf",
            quantization="IQ2",
        ),
        *_per_language(
            _scoring_result(f"{llm}@logprob", (0.9, 0.8, 0.2, 0.1), vram_gib=1),
            prompt_sha256=PROMPT_SHA256,
            duration_seconds=36.0,
            device_name="NVIDIA A100",
        ),
        *_per_language(
            _scoring_result("LiquidAI/LFM2.5-Encoder-350M", (0.9, 0.4, 0.7, 0.2), vram_gib=2),
        ),
    ]
    timing = _per_language(
        _result_with_outcomes(llm, OUTCOMES),
        duration_seconds=700.0,
        device_name="NVIDIA A100",
    )
    return dataset_card(
        results,
        benchmark_name="benchmark.csv",
        prompt_text=PROMPT,
        options=CardOptions(
            scorer_prompt_text=SCORER_PROMPT, timing_results=timing, viewer_file="data/viewer.csv"
        ),
    )


def test_the_card_with_every_optional_section_is_pinned() -> None:
    _assert_golden(_full_card(), "dataset_card_full.md")
