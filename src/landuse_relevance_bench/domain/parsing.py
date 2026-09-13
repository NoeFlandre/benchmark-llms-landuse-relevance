"""Recovery of a verdict from the raw text a model generated."""

import re

from landuse_relevance_bench.domain.labels import Label

_DECISION = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_label(raw: str) -> Label | None:
    """Return the first standalone ``yes``/``no`` token in ``raw``, or ``None``.

    ``None`` means the model did not follow the output contract; callers must
    score that as a failure rather than silently defaulting to a class.
    """
    match = _DECISION.search(raw)
    if match is None:
        return None
    return Label(match.group(1).lower())
