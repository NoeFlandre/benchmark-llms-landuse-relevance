"""Recovery of a verdict from the raw text a model generated."""

import re
from dataclasses import dataclass
from typing import Literal

from landuse_relevance_bench.domain.labels import Label

_DECISION = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_EXACT = re.compile(r"^\W*(yes|no)\W*$", re.IGNORECASE)
_LEADING = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)

ParseMode = Literal["exact", "leading", "last"]


@dataclass(frozen=True, slots=True)
class ParsedLabel:
    """A recovered label and the rule that selected it."""

    label: Label | None
    mode: ParseMode | None


def parse_with_mode(raw: str) -> ParsedLabel:
    """Prefer an exact answer, then a leading answer, then the last label token.

    The leading-answer rule handles outputs such as ``yes, because the prompt mentions
    no``. Reasoning-first outputs that do not begin with a verdict retain the historical
    last-token fallback. ``mode`` records which rule supplied the label.
    """
    for pattern, mode in ((_EXACT, "exact"), (_LEADING, "leading")):
        match = pattern.search(raw)
        if match:
            return ParsedLabel(Label(match.group(1).lower()), mode)  # type: ignore[arg-type]
    matches = _DECISION.findall(raw)
    if matches:
        return ParsedLabel(Label(matches[-1].lower()), "last")
    return ParsedLabel(None, None)


def parse_label(raw: str) -> Label | None:
    """Return the label recovered by :func:`parse_with_mode`, or ``None``."""
    return parse_with_mode(raw).label
