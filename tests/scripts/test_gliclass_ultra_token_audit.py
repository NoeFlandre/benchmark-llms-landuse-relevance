import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from landuse_relevance_bench.adapters.translations import TranslationManifest
from landuse_relevance_bench.domain.dataset import BenchmarkItem
from landuse_relevance_bench.domain.labels import Label
from scripts import audit_gliclass_ultra_lengths as audit
from scripts.audit_gliclass_ultra_lengths import _input_formatter, _records_for_batch


def test_audit_lengths_writes_exact_rows_and_refuses_truncation_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"architecture_type": "uni-encoder", "prompt_first": True}),
        encoding="utf-8",
    )
    manifest = TranslationManifest(
        dataset="NoeFlandre/landuse-sentence-relevance-golden-human-set",
        revision=audit.EXPECTED_MANIFEST_REVISION,
        split="train",
        languages=("en",),
        row_count=2,
        source_item_ids=("source-short", "source-long"),
        files={},
        whole_set_sha256="whole-set-sha",
    )
    items = (
        BenchmarkItem("item-short", "source-short", "en", "sentence-within", Label.YES),
        BenchmarkItem("item-long", "source-long", "en", "sentence-over", Label.NO),
    )

    class FakeTokenizer:
        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
            assert kwargs == {"truncation": False, "padding": False}
            lengths = [512 if text.endswith("sentence-within") else 513 for text in texts]
            return {"input_ids": [[0] * length for length in lengths]}

    tokenizer = FakeTokenizer()
    hub_calls: list[dict[str, str]] = []

    def download_config(model_id: str, *, filename: str, revision: str) -> str:
        hub_calls.append({"model_id": model_id, "filename": filename, "revision": revision})
        return str(config_path)

    def load_tokenizer(model_id: str, **kwargs: object) -> FakeTokenizer:
        assert model_id == audit.GLICLASS_ULTRA_ID
        assert kwargs == {
            "revision": audit.GLICLASS_ULTRA_REVISION,
            "trust_remote_code": False,
        }
        return tokenizer

    monkeypatch.setattr(audit, "EXPECTED_LANGUAGE_COUNT", 1)
    monkeypatch.setattr(audit, "EXPECTED_ROWS_PER_LANGUAGE", 2)
    monkeypatch.setattr(audit, "hf_hub_download", download_config)
    monkeypatch.setattr(audit.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(audit, "load_manifest", lambda _root: manifest)
    monkeypatch.setattr(audit, "load_language_benchmark", lambda _root, _lang: items)

    csv_path = tmp_path / "audit.csv"
    metadata_path = tmp_path / "audit.json"
    metadata = audit.audit_lengths(
        output=csv_path,
        metadata_output=metadata_path,
        root=tmp_path,
    )
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert hub_calls == [
        {
            "model_id": audit.GLICLASS_ULTRA_ID,
            "filename": "config.json",
            "revision": audit.GLICLASS_ULTRA_REVISION,
        }
    ]
    assert metadata["row_count"] == 2
    assert metadata["minimum_token_count"] == 512
    assert metadata["maximum_token_count"] == 513
    assert metadata["over_limit_count"] == 1
    assert [(row["item_id"], row["token_count"], row["over_limit"]) for row in rows] == [
        ("item-short", "512", "False"),
        ("item-long", "513", "True"),
    ]
    assert metadata_path.is_file()


def test_audit_uses_the_pinned_uni_encoder_label_and_sentence_format() -> None:
    formatter, architecture, prompt_first = _input_formatter(
        {"architecture_type": "uni-encoder", "prompt_first": True}
    )

    formatted = formatter.prepare_input("bonjour", ["land use hypothesis"])

    assert architecture == "uni-encoder"
    assert prompt_first is True
    assert formatted == "<<LABEL>>land use hypothesis<<SEP>>bonjour"


def test_token_audit_records_lengths_and_flags_only_overflow() -> None:
    class FakeTokenizer:
        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
            assert kwargs == {"truncation": False, "padding": False}
            return {"input_ids": [[0] * int(text) for text in texts]}

    rows = [
        ("en", SimpleNamespace(item_id="short", source_item_id="source-1"), "512"),
        ("fr", SimpleNamespace(item_id="long", source_item_id="source-1"), "513"),
    ]

    records = _records_for_batch(FakeTokenizer(), rows)

    assert [record.token_count for record in records] == [512, 513]
    assert [record.over_limit for record in records] == [False, True]
    assert [(record.language, record.item_id) for record in records] == [
        ("en", "short"),
        ("fr", "long"),
    ]
