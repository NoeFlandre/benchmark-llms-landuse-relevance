"""Resolving the exact weights commit a run used, so a result is reproducible."""

import logging

logger = logging.getLogger(__name__)


def hub_commit(model_id: str) -> str:
    """The commit the Hub currently serves for ``model_id``, or ``""`` if unreachable."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha or ""
    except (ImportError, OSError, ValueError) as exc:
        logger.debug("could not resolve %s on the Hub: %s", model_id, exc)
        return ""


def resolve_revision(model_id: str, *candidates: str | None) -> str:
    """The first known revision among ``candidates``, else the Hub's current commit.

    An unresolved revision is recorded as ``""`` with a warning, so a result that cannot
    be reproduced exactly is visible when it is written, not when it is re-run.
    """
    for candidate in candidates:
        if candidate:
            return candidate
    revision = hub_commit(model_id)
    if not revision:
        logger.warning(
            "could not determine the weights revision of %s; pass --revision to pin it",
            model_id,
        )
    return revision
