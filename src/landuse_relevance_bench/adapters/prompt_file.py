"""Loading the prompt template from disk."""

from pathlib import Path

from landuse_relevance_bench.domain.prompting import PLACEHOLDER


class PromptFileError(ValueError):
    """Raised when a prompt file cannot serve as a benchmark template."""


def load_prompt(path: Path) -> str:
    """Read ``path`` verbatim, checking it can actually host a target sentence."""
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise PromptFileError(f"prompt file {path} is empty")
    if PLACEHOLDER not in template:
        raise PromptFileError(
            f"prompt file {path} contains no {PLACEHOLDER!r} placeholder for the target sentence"
        )
    return template
