"""Content digests used to pin inputs to their runs.

The text digest is pure and lives in the domain; it is re-exported here for adapters.
"""

import hashlib
from pathlib import Path

from landuse_relevance_bench.domain.digests import sha256_of_text

__all__ = ["sha256_of_file", "sha256_of_text"]

_CHUNK = 1 << 20


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_bytes(value: bytes) -> str:
    """Hash an in-memory input snapshot without reopening its source path."""
    return hashlib.sha256(value).hexdigest()
