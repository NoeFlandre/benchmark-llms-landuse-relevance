"""Resolving the exact weights commit a run used, so a result is reproducible."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)


def is_immutable_revision(revision: str | None) -> bool:
    """Whether ``revision`` is a full Hub commit id that cannot move."""
    return revision is not None and _COMMIT_SHA.fullmatch(revision) is not None


def hub_commit(model_id: str, revision: str | None = None) -> str:
    """Resolve a model ref to its commit id, or return ``""`` if the Hub is unreachable."""
    try:
        from huggingface_hub import HfApi

        info = (
            HfApi().model_info(model_id, revision=revision)
            if revision
            else HfApi().model_info(model_id)
        )
        commit = str(info.sha or "")
        return commit.lower() if is_immutable_revision(commit) else ""
    except Exception as exc:  # noqa: BLE001 - Hub/network failures must not abort an optional pin lookup.
        logger.debug("could not resolve %s at %s on the Hub: %s", model_id, revision, exc)
        return ""


def resolve_revision(model_id: str, *candidates: str | None) -> str:
    """Resolve a requested ref, or the current default, to an immutable commit id.

    Full commit ids need no network request. Branches, tags, short hashes, and an unset
    revision are looked up on the Hub. If that lookup fails, return ``""`` so callers can
    refuse checkpoint reuse rather than trust a mutable or unknown ref.
    """
    for candidate in candidates:
        if candidate is not None and is_immutable_revision(candidate):
            return candidate.lower()
    requested = next((candidate for candidate in candidates if candidate), None)
    revision = hub_commit(model_id, requested)
    if not is_immutable_revision(revision):
        logger.warning("could not determine an immutable weights revision of %s", model_id)
        return ""
    return revision.lower()


def loaded_revision(model: Any) -> str:
    """The commit hash recorded on a loaded model's config, or ``""`` if it has none."""
    return str(getattr(model.config, "_commit_hash", "") or "")
