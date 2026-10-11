import hashlib
from pathlib import Path

from landuse_relevance_bench.adapters import hashing
from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text
from landuse_relevance_bench.domain import digests


def test_file_digest_matches_the_digest_of_its_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_of_file(path) == sha256_of_text("abc")


def test_text_digest_is_the_sha256_of_its_utf8_bytes() -> None:
    assert sha256_of_text("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert sha256_of_text("é") == hashlib.sha256("é".encode()).hexdigest()


def test_the_adapter_re_exports_the_domain_text_digest() -> None:
    assert hashing.sha256_of_text is digests.sha256_of_text
