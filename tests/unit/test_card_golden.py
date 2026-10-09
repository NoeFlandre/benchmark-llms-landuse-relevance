"""Byte-for-byte golden cards, so card composition can be refactored mechanically.

Regenerate a golden file only for an intended change to the published card.
"""

from pathlib import Path

from tests.unit.test_hf_publish import (
    PROMPT,
    SCORER_PROMPT,
    _result_with_outcomes,
    _scoring_result,
)

from landuse_relevance_bench.adapters.hf_publish import dataset_card
from landuse_relevance_bench.adapters.prompt_file import load_prompt
from landuse_relevance_bench.adapters.results_store import read_runs
from landuse_relevance_bench.domain.labels import Label

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
        scorer_prompt_text=SCORER_PROMPT,
        viewer_file="data/viewer.csv",
    )


def test_the_card_with_generative_and_scoring_runs_is_pinned() -> None:
    _assert_golden(_mixed_card(), "dataset_card_mixed.md")
