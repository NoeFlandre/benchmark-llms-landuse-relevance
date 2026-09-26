from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROMPT_TEMPLATE = "Classify.\n\nOutput only yes or no.\n\nTARGET SENTENCE: {}"
CSV_TEXT = (
    "sentence,label,polygon_name,h3_cell,latitude,longitude,source,region,source_url\n"
    '"Dense mangrove forest lines the lagoon.",yes,Rabi,839116fffffffff,-16.4,-122.1,'
    "wikipedia,fiji,https://example.org/a\n"
    '"The council was dissolved in 1974.",no,Cruzen Island,83f35efffffffff,-74.7,-140.3,'
    "website,antarctica,https://example.org/b\n"
)


@pytest.fixture(autouse=True)
def _plain_cli_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Typer/Rich error output uncoloured and unwrapped regardless of the host.

    On GitHub Actions Rich forces colour and wraps at 80 columns, which splits
    messages that tests assert on.
    """
    for var in ("GITHUB_ACTIONS", "FORCE_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")


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
    return PROJECT_ROOT / "data" / "benchmark.csv"


@pytest.fixture
def real_prompt_path() -> Path:
    return PROJECT_ROOT / "data" / "prompt.txt"
