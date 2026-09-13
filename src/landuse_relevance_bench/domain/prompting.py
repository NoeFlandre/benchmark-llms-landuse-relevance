"""Rendering of the benchmark prompt template."""

PLACEHOLDER = "{}"


class MissingPlaceholderError(ValueError):
    """Raised when a prompt template has no slot for the target sentence."""


def render_prompt(template: str, sentence: str) -> str:
    """Insert ``sentence`` at the first ``{}`` of ``template``.

    Plain substitution, not :meth:`str.format`, so braces inside the sentence are
    left untouched.
    """
    if PLACEHOLDER not in template:
        raise MissingPlaceholderError(
            f"prompt template contains no {PLACEHOLDER!r} placeholder for the target sentence"
        )
    return template.replace(PLACEHOLDER, sentence, 1)
