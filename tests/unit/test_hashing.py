from pathlib import Path

from landuse_relevance_bench.adapters.hashing import sha256_of_file, sha256_of_text


def test_file_digest_matches_the_digest_of_its_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_of_file(path) == sha256_of_text("abc")
