"""Record GLiClass Ultra token lengths without loading model weights."""

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, TypedDict, cast

from gliclass.pipeline import UniEncoderZeroShotClassificationPipeline
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from landuse_relevance_bench.adapters.hf_scorer import zeroshot_hypothesis
from landuse_relevance_bench.adapters.translations import (
    TranslationManifest,
    load_language_benchmark,
    load_manifest,
)
from landuse_relevance_bench.domain.dataset import BenchmarkItem
from landuse_relevance_bench.domain.scorers import GLICLASS_ULTRA_ID, GLICLASS_ULTRA_REVISION

EXPECTED_MANIFEST_REVISION = "d431a6c522b3c3a616e4f3d2814906c3011843e7"
EXPECTED_LANGUAGE_COUNT = 85
EXPECTED_ROWS_PER_LANGUAGE = 300
PROMPT_REPO_PATH = "data/prompt_zeroshot.txt"
MAX_SEQUENCE_LENGTH = 512
TOKENIZER_BATCH_SIZE = 512
INPUT_COLUMNS = ("language", "item_id", "source_item_id", "token_count", "over_limit")
LABEL_MARKER = "<<LABEL>>"
SEPARATOR_MARKER = "<<SEP>>"


class TokenBatch(Protocol):
    def __getitem__(self, key: str) -> Sequence[Sequence[int]]: ...


class Tokenizer(Protocol):
    def __call__(self, texts: Sequence[str], *, truncation: bool, padding: bool) -> TokenBatch: ...


class PromptConfig(Protocol):
    prompt_first: bool


class PipelineModel(Protocol):
    config: PromptConfig


class InputFormatter(Protocol):
    label_token: str
    sep_token: str
    model: PipelineModel

    def prepare_input(
        self,
        text: str,
        labels: list[str],
        examples: object | None = None,
        prompt: str | None = None,
    ) -> str: ...


class AuditMetadata(TypedDict):
    model_id: str
    model_revision: str
    tokenizer_revision: str
    manifest_dataset: str
    manifest_revision: str
    manifest_whole_set_sha256: str
    language_count: int
    rows_per_language: int
    tokenizer_class: str
    architecture_type: str
    prompt_first: bool
    prompt_path: str
    prompt_sha256: str
    hypothesis: str
    max_sequence_length: int
    row_count: int
    minimum_token_count: int
    maximum_token_count: int
    over_limit_count: int
    token_lengths_csv: str
    token_lengths_csv_sha256: str


@dataclass(frozen=True, slots=True)
class TokenLengthRecord:
    language: str
    item_id: str
    source_item_id: str
    token_count: int
    over_limit: bool


def _model_config() -> dict[str, object]:
    path = hf_hub_download(
        GLICLASS_ULTRA_ID,
        filename="config.json",
        revision=GLICLASS_ULTRA_REVISION,
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("the pinned GLiClass Ultra config is not a JSON object")
    return payload


def _input_formatter(config: Mapping[str, object]) -> tuple[InputFormatter, str, bool]:
    architecture_type = config.get("architecture_type")
    prompt_first = config.get("prompt_first")
    if (
        not isinstance(architecture_type, str)
        or architecture_type != "uni-encoder"
        or not isinstance(prompt_first, bool)
    ):
        raise ValueError(
            "the pinned Ultra tokenizer audit expects a uni-encoder config with prompt_first"
        )
    formatter = cast(InputFormatter, object.__new__(UniEncoderZeroShotClassificationPipeline))
    formatter.label_token = LABEL_MARKER
    formatter.sep_token = SEPARATOR_MARKER
    formatter.model = cast(
        PipelineModel,
        SimpleNamespace(config=SimpleNamespace(prompt_first=prompt_first)),
    )
    return formatter, architecture_type, prompt_first


def _formatted_inputs(
    root: Path,
    languages: Sequence[str],
    formatter: InputFormatter,
    hypothesis: str,
) -> Iterable[tuple[str, BenchmarkItem, str]]:
    for language in languages:
        for item in load_language_benchmark(root, language):
            formatted = formatter.prepare_input(item.sentence, [hypothesis])
            yield language, item, formatted


def _token_lengths(tokenizer: Tokenizer, formatted_inputs: Sequence[str]) -> list[int]:
    lengths = []
    for offset in range(0, len(formatted_inputs), TOKENIZER_BATCH_SIZE):
        batch = formatted_inputs[offset : offset + TOKENIZER_BATCH_SIZE]
        encoded = tokenizer(batch, truncation=False, padding=False)
        token_ids = encoded["input_ids"]
        if len(token_ids) != len(batch):
            raise ValueError("tokenizer returned a different number of rows than it received")
        lengths.extend(len(sequence) for sequence in token_ids)
    return lengths


def _write_report(
    records: Sequence[TokenLengthRecord],
    output: Path,
    metadata_output: Path,
    context: "AuditContext",
) -> AuditMetadata:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    csv_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    over_limit_count = sum(record.over_limit for record in records)
    metadata: AuditMetadata = {
        "model_id": GLICLASS_ULTRA_ID,
        "model_revision": GLICLASS_ULTRA_REVISION,
        "tokenizer_revision": GLICLASS_ULTRA_REVISION,
        "manifest_dataset": context.manifest.dataset,
        "manifest_revision": context.manifest.revision,
        "manifest_whole_set_sha256": context.manifest.whole_set_sha256,
        "language_count": len(context.manifest.languages),
        "rows_per_language": context.manifest.row_count,
        "tokenizer_class": type(context.tokenizer).__name__,
        "architecture_type": context.architecture_type,
        "prompt_first": context.prompt_first,
        "prompt_path": PROMPT_REPO_PATH,
        "prompt_sha256": hashlib.sha256(context.prompt_path.read_bytes()).hexdigest(),
        "hypothesis": context.hypothesis,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "row_count": len(records),
        "minimum_token_count": min(record.token_count for record in records),
        "maximum_token_count": max(record.token_count for record in records),
        "over_limit_count": over_limit_count,
        "token_lengths_csv": output.name,
        "token_lengths_csv_sha256": csv_sha256,
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


@dataclass(frozen=True, slots=True)
class AuditContext:
    manifest: TranslationManifest
    tokenizer: Tokenizer
    architecture_type: str
    prompt_first: bool
    prompt_path: Path
    hypothesis: str


def audit_lengths(*, output: Path, metadata_output: Path, root: Path) -> AuditMetadata:
    manifest = load_manifest(root)
    if manifest.revision != EXPECTED_MANIFEST_REVISION:
        raise ValueError(
            f"translation manifest revision is {manifest.revision}; "
            f"expected {EXPECTED_MANIFEST_REVISION}"
        )
    if (
        len(manifest.languages) != EXPECTED_LANGUAGE_COUNT
        or manifest.row_count != EXPECTED_ROWS_PER_LANGUAGE
    ):
        raise ValueError("the Ultra audit requires the frozen 85-language by 300-row inventory")

    config = _model_config()
    formatter, architecture_type, prompt_first = _input_formatter(config)
    tokenizer = cast(
        Tokenizer,
        AutoTokenizer.from_pretrained(
            GLICLASS_ULTRA_ID,
            revision=GLICLASS_ULTRA_REVISION,
            trust_remote_code=False,
        ),
    )
    project_root = Path(__file__).resolve().parents[1]
    prompt_path = project_root / PROMPT_REPO_PATH
    hypothesis = zeroshot_hypothesis(prompt_path.read_text(encoding="utf-8"))

    records: list[TokenLengthRecord] = []
    batch_rows: list[tuple[str, BenchmarkItem, str]] = []
    for row in _formatted_inputs(root, manifest.languages, formatter, hypothesis):
        batch_rows.append(row)
        if len(batch_rows) == TOKENIZER_BATCH_SIZE:
            records.extend(_records_for_batch(tokenizer, batch_rows))
            batch_rows.clear()
    if batch_rows:
        records.extend(_records_for_batch(tokenizer, batch_rows))

    expected_rows = len(manifest.languages) * manifest.row_count
    if len(records) != expected_rows:
        raise ValueError(f"tokenizer audit recorded {len(records)} rows; expected {expected_rows}")
    return _write_report(
        records,
        output,
        metadata_output,
        AuditContext(
            manifest,
            tokenizer,
            architecture_type,
            prompt_first,
            prompt_path,
            hypothesis,
        ),
    )


def _records_for_batch(
    tokenizer: Tokenizer, rows: Sequence[tuple[str, BenchmarkItem, str]]
) -> list[TokenLengthRecord]:
    lengths = _token_lengths(tokenizer, [formatted for _, _, formatted in rows])
    return [
        TokenLengthRecord(
            language=language,
            item_id=item.item_id,
            source_item_id=item.source_item_id,
            token_count=length,
            over_limit=length > MAX_SEQUENCE_LENGTH,
        )
        for (language, item, _), length in zip(rows, lengths, strict=True)
    ]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "audits" / "gliclass-ultra-token-lengths.csv",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_root / "data" / "audits" / "gliclass-ultra-token-lengths.json",
    )
    parser.add_argument("--data-root", type=Path, default=project_root / "data" / "translations")
    args = parser.parse_args()

    metadata = audit_lengths(
        output=args.output,
        metadata_output=args.metadata_output,
        root=args.data_root,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if int(metadata["over_limit_count"]) > 0:
        print("Overlength inputs were recorded; refusing a 512-token GLiClass Ultra run.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
