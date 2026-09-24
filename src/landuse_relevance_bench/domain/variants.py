"""Roster ids that name a variant of a Hub checkpoint (``repo@variant``).

Two methods on one checkpoint (a log-probability read of a generative model, a GGUF
quant) need distinct roster ids so their result files never collide; the part before
the separator is always the loadable Hub repository.
"""

VARIANT_SEPARATOR = "@"


def repository_of(model_id: str) -> str:
    """Strip a ``@variant`` suffix, leaving the loadable Hub repository id."""
    return model_id.split(VARIANT_SEPARATOR, 1)[0]
