"""Text digests that pin a run's inputs to the exact text they were computed from."""

import hashlib


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
