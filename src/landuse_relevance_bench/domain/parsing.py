"""Recovery of a verdict from the raw text a model generated."""

import re

from landuse_relevance_bench.domain.labels import Label

_DECISION = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_label(raw: str) -> Label | None:
    """Return the last standalone ``yes``/``no`` token in ``raw``, or ``None``.

    The *last* rather than the first: a model that answers directly emits one token
    either way, while a model that reasons first restates both options — ``Criteria
    for "yes": ...`` — before committing. Reading from the front scores that
    restatement as the verdict.

    ``None`` means the model did not follow the output contract; callers must score
    that as a failure rather than silently defaulting to a class.
    """
    matches = _DECISION.findall(raw)
    if not matches:
        return None
    return Label(matches[-1].lower())
