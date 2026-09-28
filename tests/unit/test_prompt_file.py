from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.prompt_file import PromptFileError, load_prompt


def test_loads_the_template_verbatim(prompt_path: Path) -> None:
    assert load_prompt(prompt_path).endswith("TARGET SENTENCE: {}")


def test_rejects_a_template_without_a_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "p.txt"
    path.write_text("no placeholder", encoding="utf-8")
    with pytest.raises(PromptFileError):
        load_prompt(path)


def test_rejects_a_blank_template(tmp_path: Path) -> None:
    path = tmp_path / "p.txt"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(PromptFileError):
        load_prompt(path)


def test_loads_the_real_project_prompt(real_prompt_path: Path) -> None:
    template = load_prompt(real_prompt_path)
    assert "TARGET SENTENCE: {}" in template


def test_multilingual_prompt_policy_is_documented() -> None:
    adr = Path(__file__).parents[2] / "docs" / "adr" / "0006-multilingual-prompt-language.md"
    text = adr.read_text(encoding="utf-8").lower()

    assert "english" in text
    assert "yes" in text and "no" in text
    assert "unparsed" in text
    assert "raw_output" in text
    assert "recomput" in text
