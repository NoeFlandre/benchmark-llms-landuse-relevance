from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROMPT_TEMPLATE = "Classify.\n\nOutput only yes or no.\n\nTARGET SENTENCE: {}"
CSV_TEXT = (
    "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url,"
    "source_item_id,language\n"
    '"Dense mangrove forest lines the lagoon.",yes,Rabi,839116fffffffff,-16.4,-122.1,'
    "wikipedia,fiji,https://example.org/a,source-1,en\n"
    '"The council was dissolved in 1974.",no,Cruzen Island,83f35efffffffff,-74.7,-140.3,'
    "website,antarctica,https://example.org/b,source-2,en\n"
)


@pytest.fixture
def prompt_path(tmp_path: Path) -> Path:
    path = tmp_path / "prompt.txt"
    path.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    return path


@pytest.fixture
def benchmark_path(tmp_path: Path) -> Path:
    path = tmp_path / "benchmark.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def real_benchmark_path() -> Path:
    return PROJECT_ROOT / "data" / "translations" / "en" / "v3-final-en.csv"


@pytest.fixture
def real_prompt_path() -> Path:
    return PROJECT_ROOT / "data" / "prompt.txt"
