"""Vendor the fixed multilingual benchmark from the Hugging Face Dataset Viewer."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from landuse_relevance_bench.adapters.hashing import sha256_of_file
from landuse_relevance_bench.adapters.translations import (
    LANGUAGE_NEUTRAL_COLUMNS,
    LANGUAGE_PATTERN,
    TranslationDataError,
    TranslationFile,
    source_item_ids_for_rows,
    whole_set_sha256,
)

DATASET_VIEWER_BASE = "https://datasets-server.huggingface.co"
HUB_API_BASE = "https://huggingface.co/api/datasets"
DEFAULT_DATASET = "NoeFlandre/landuse-sentence-relevance-golden-human-set"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT = Path("data/translations")
PAGE_LENGTH = 100
EXPECTED_LANGUAGE_COUNT = 85
REQUEST_DELAY_SECONDS = 1.5
MAX_FETCH_ATTEMPTS = 7
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}
TRANSLATION_PATH_PART_COUNT = 4
UPSTREAM_COLUMNS = ("sentence", *LANGUAGE_NEUTRAL_COLUMNS)
VENDORED_COLUMNS = (*UPSTREAM_COLUMNS, "source_item_id", "language")
JsonFetcher = Callable[[str], object]
ContentFetcher = Callable[[str], bytes]
_request_state = {"last_request_at": 0.0}


def fetch_json(url: str) -> object:
    """Fetch one public JSON endpoint with a stable user agent."""
    try:
        return json.loads(fetch_bytes(url).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TranslationDataError(f"could not decode JSON from {url}: {exc}") from exc


def fetch_bytes(url: str) -> bytes:
    """Fetch one public resource with throttling and transient-error retries."""
    for attempt in range(MAX_FETCH_ATTEMPTS):
        elapsed = time.monotonic() - _request_state["last_request_at"]
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)
        request = Request(url, headers={"User-Agent": "landuse-relevance-bench-v3/1.0"})
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read()
            _request_state["last_request_at"] = time.monotonic()
            return payload
        except HTTPError as exc:
            _request_state["last_request_at"] = time.monotonic()
            if exc.code not in RETRYABLE_HTTP_STATUS or attempt == MAX_FETCH_ATTEMPTS - 1:
                raise TranslationDataError(f"could not fetch JSON from {url}: {exc}") from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                delay = 0.0
            time.sleep(max(2.0, min(30.0, delay or 2**attempt)))
        except (URLError, TimeoutError, OSError) as exc:
            raise TranslationDataError(f"could not fetch resource from {url}: {exc}") from exc
    raise AssertionError("fetch loop returned without a result")


def vendor_dataset(  # noqa: PLR0913
    *,
    dataset: str,
    split: str,
    output: Path,
    expected_row_count: int = 300,
    expected_language_count: int = EXPECTED_LANGUAGE_COUNT,
    source: str = "hub",
    fetcher: JsonFetcher = fetch_json,
    content_fetcher: ContentFetcher = fetch_bytes,
    force: bool = False,
) -> None:
    """Fetch, validate, and atomically install every language split."""
    if expected_row_count <= 0:
        raise TranslationDataError("expected_row_count must be positive")
    if expected_language_count <= 0:
        raise TranslationDataError("expected_language_count must be positive")
    if source == "viewer":
        _vendor_viewer_dataset(
            dataset=dataset,
            split=split,
            output=output,
            expected_row_count=expected_row_count,
            expected_language_count=expected_language_count,
            fetcher=fetcher,
            force=force,
        )
        return
    if source == "hub":
        _vendor_hub_dataset(
            dataset=dataset,
            split=split,
            output=output,
            expected_row_count=expected_row_count,
            expected_language_count=expected_language_count,
            fetcher=fetcher,
            content_fetcher=content_fetcher,
            force=force,
        )
        return
    raise TranslationDataError(f"unknown vendor source {source!r}; expected 'hub' or 'viewer'")


def _vendor_viewer_dataset(  # noqa: PLR0913
    *,
    dataset: str,
    split: str,
    output: Path,
    expected_row_count: int,
    expected_language_count: int,
    fetcher: JsonFetcher,
    force: bool,
) -> None:
    split_entries = _discover_split_entries(
        dataset,
        split,
        expected_row_count=expected_row_count,
        fetcher=fetcher,
    )
    language_rows = {
        language: _fetch_language_rows(
            dataset,
            split,
            language,
            expected_row_count=expected_row_count,
            fetcher=fetcher,
        )
        for language, _ in split_entries
    }
    languages = tuple(sorted(language_rows))
    _check_language_count(languages, expected_language_count)
    source_item_ids = source_item_ids_for_rows(language_rows[languages[0]])
    for language in languages:
        actual_source_item_ids = source_item_ids_for_rows(language_rows[language])
        if actual_source_item_ids != source_item_ids:
            raise TranslationDataError(
                f"language {language!r} does not preserve the source item order or identity"
            )
    revision = _dataset_revision(dataset, fetcher)
    _install_vendor_output(
        output=output,
        dataset=dataset,
        revision=revision,
        split=split,
        languages=languages,
        language_rows=language_rows,
        source_item_ids=source_item_ids,
        force=force,
    )


def _vendor_hub_dataset(  # noqa: PLR0913
    *,
    dataset: str,
    split: str,
    output: Path,
    expected_row_count: int,
    expected_language_count: int,
    fetcher: JsonFetcher,
    content_fetcher: ContentFetcher,
    force: bool,
) -> None:
    revision = _dataset_revision(dataset, fetcher)
    paths = _hub_translation_paths(dataset, revision, split, fetcher)
    languages = tuple(sorted(paths))
    _check_language_count(languages, expected_language_count)
    language_rows = {
        language: _read_hub_csv(
            language,
            content_fetcher(_hub_raw_url(dataset, revision, paths[language])),
            expected_row_count=expected_row_count,
        )
        for language in languages
    }
    source_item_ids = source_item_ids_for_rows(language_rows[languages[0]])
    for language in languages:
        actual_source_item_ids = source_item_ids_for_rows(language_rows[language])
        if actual_source_item_ids != source_item_ids:
            raise TranslationDataError(
                f"language {language!r} does not preserve the source item order or identity"
            )
    _install_vendor_output(
        output=output,
        dataset=dataset,
        revision=revision,
        split=split,
        languages=languages,
        language_rows=language_rows,
        source_item_ids=source_item_ids,
        force=force,
    )


def _check_language_count(languages: Sequence[str], expected: int) -> None:
    if len(languages) != expected:
        raise TranslationDataError(f"found {len(languages)} language splits; expected {expected}")


def _discover_split_entries(
    dataset: str,
    split: str,
    *,
    expected_row_count: int,
    fetcher: JsonFetcher,
) -> tuple[tuple[str, int], ...]:
    payload = _mapping(fetcher(_splits_url(dataset)), "splits response")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, Sequence) or isinstance(raw_splits, (str, bytes)):
        raise TranslationDataError("splits response has no split list")
    entries: dict[str, int] = {}
    for raw_entry in raw_splits:
        entry = _mapping(raw_entry, "split entry")
        if entry.get("split") != split:
            continue
        language = entry.get("config")
        if not isinstance(language, str) or not LANGUAGE_PATTERN.fullmatch(language):
            raise TranslationDataError(f"invalid language config in splits response: {language!r}")
        if language in entries:
            raise TranslationDataError(f"duplicate language config {language!r}")
        num_rows = entry.get("num_rows")
        if num_rows is not None and (
            not isinstance(num_rows, int) or num_rows != expected_row_count
        ):
            raise TranslationDataError(
                f"language {language!r} declares {num_rows!r} rows; "
                f"expected {expected_row_count} rows"
            )
        entries[language] = expected_row_count if num_rows is None else num_rows
    if not entries:
        raise TranslationDataError(f"dataset {dataset!r} has no {split!r} language splits")
    return tuple(sorted(entries.items()))


def _fetch_language_rows(
    dataset: str,
    split: str,
    language: str,
    *,
    expected_row_count: int,
    fetcher: JsonFetcher,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    offset = 0
    while len(rows) < expected_row_count:
        url = _rows_url(dataset, language, split, offset)
        payload = _mapping(fetcher(url), f"rows response for {language!r}")
        raw_rows = payload.get("rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise TranslationDataError(f"rows response for {language!r} has no row list")
        if not raw_rows:
            break
        for raw_entry in raw_rows:
            entry = _mapping(raw_entry, f"row entry for {language!r}")
            row = _mapping(entry.get("row"), f"row payload for {language!r}")
            missing = [column for column in UPSTREAM_COLUMNS if column not in row]
            if missing:
                raise TranslationDataError(
                    f"language {language!r} is missing upstream column(s) {missing}"
                )
            rows.append(row)
        offset = len(rows)
        if len(raw_rows) < PAGE_LENGTH:
            break
    if len(rows) != expected_row_count:
        raise TranslationDataError(
            f"language {language!r} returned {len(rows)} rows; expected {expected_row_count} rows"
        )
    return tuple(rows)


def _hub_translation_paths(
    dataset: str,
    revision: str,
    split: str,
    fetcher: JsonFetcher,
) -> dict[str, str]:
    if split != "train":
        raise TranslationDataError("the Hub translation source only provides the train split")
    payload = fetcher(_tree_url(dataset, revision))
    if isinstance(payload, Mapping):
        payload = payload.get("tree")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise TranslationDataError("Hub tree response has no file list")
    paths: dict[str, str] = {}
    for raw_entry in payload:
        entry = _mapping(raw_entry, "Hub tree entry")
        if entry.get("type") != "file":
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        parts = path.split("/")
        if len(parts) != TRANSLATION_PATH_PART_COUNT or parts[:2] != ["data", "translations"]:
            continue
        language, filename = parts[2:]
        if filename != f"v3-final-{language}.csv" or not LANGUAGE_PATTERN.fullmatch(language):
            continue
        if language in paths:
            raise TranslationDataError(f"duplicate Hub translation file for {language!r}")
        paths[language] = path
    if not paths:
        raise TranslationDataError(f"Hub dataset {dataset!r} has no translation CSV files")
    return paths


def _read_hub_csv(
    language: str,
    payload: bytes,
    *,
    expected_row_count: int,
) -> tuple[Mapping[str, Any], ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TranslationDataError(f"Hub translation {language!r} is not UTF-8") from exc
    if text.startswith("\ufeff"):
        raise TranslationDataError(f"Hub translation {language!r} has a UTF-8 BOM")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != UPSTREAM_COLUMNS:
        raise TranslationDataError(
            f"Hub translation {language!r} has header {reader.fieldnames!r}; "
            f"expected {list(UPSTREAM_COLUMNS)!r}"
        )
    rows: list[Mapping[str, Any]] = []
    for line, raw_row in enumerate(reader, start=2):
        if None in raw_row or any(value is None for value in raw_row.values()):
            raise TranslationDataError(
                f"Hub translation {language!r} has a malformed row at line {line}"
            )
        rows.append(raw_row)
    if len(rows) != expected_row_count:
        raise TranslationDataError(
            f"language {language!r} returned {len(rows)} rows; expected {expected_row_count} rows"
        )
    return tuple(rows)


def _dataset_revision(dataset: str, fetcher: JsonFetcher) -> str:
    payload = _mapping(fetcher(f"{HUB_API_BASE}/{quote(dataset, safe='/')}"), "dataset metadata")
    revision = payload.get("sha")
    if not isinstance(revision, str) or not revision.strip():
        raise TranslationDataError(f"dataset metadata for {dataset!r} has no revision SHA")
    return revision


def _install_vendor_output(  # noqa: PLR0913
    *,
    output: Path,
    dataset: str,
    revision: str,
    split: str,
    languages: tuple[str, ...],
    language_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_item_ids: Sequence[str],
    force: bool,
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise TranslationDataError(
            f"output directory already exists: {output}; pass --force to replace it"
        )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        files: dict[str, TranslationFile] = {}
        for language in languages:
            relative_path = f"{language}/v3-final-{language}.csv"
            path = stage / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_language_csv(path, language_rows[language], source_item_ids, language)
            files[language] = TranslationFile(
                path=relative_path,
                rows=len(language_rows[language]),
                sha256=sha256_of_file(path),
            )
        manifest = {
            "dataset": dataset,
            "revision": revision,
            "split": split,
            "languages": list(languages),
            "row_count": len(source_item_ids),
            "source_item_ids": list(source_item_ids),
            "files": {
                language: {
                    "path": files[language].path,
                    "rows": files[language].rows,
                    "sha256": files[language].sha256,
                }
                for language in languages
            },
            "whole_set_sha256": whole_set_sha256(files),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            if not output.is_dir():
                raise TranslationDataError(f"output path is not a directory: {output}")
            shutil.rmtree(output)
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _write_language_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    source_item_ids: Sequence[str],
    language: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VENDORED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row, source_item_id in zip(rows, source_item_ids, strict=True):
            serialized = {column: _csv_value(row[column], column) for column in UPSTREAM_COLUMNS}
            serialized.update(source_item_id=source_item_id, language=language)
            writer.writerow(serialized)


def _csv_value(value: Any, column: str) -> str:
    if value is None:
        return ""
    if column == "label" and isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _splits_url(dataset: str) -> str:
    return f"{DATASET_VIEWER_BASE}/splits?{urlencode({'dataset': dataset})}"


def _rows_url(dataset: str, language: str, split: str, offset: int) -> str:
    query = urlencode(
        {
            "dataset": dataset,
            "config": language,
            "split": split,
            "offset": offset,
            "length": PAGE_LENGTH,
        }
    )
    return f"{DATASET_VIEWER_BASE}/rows?{query}"


def _tree_url(dataset: str, revision: str) -> str:
    return (
        f"{HUB_API_BASE}/{quote(dataset, safe='/')}/tree/{quote(revision, safe='')}"
        "?recursive=true&expand=false"
    )


def _hub_raw_url(dataset: str, revision: str, path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{quote(dataset, safe='/')}"
        f"/resolve/{quote(revision, safe='')}/{quote(path, safe='/')}"
    )


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TranslationDataError(f"{description} is not a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-rows", type=int, default=300)
    parser.add_argument("--expected-languages", type=int, default=EXPECTED_LANGUAGE_COUNT)
    parser.add_argument("--source", choices=("hub", "viewer"), default="hub")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        vendor_dataset(
            dataset=args.dataset,
            split=args.split,
            output=args.output,
            expected_row_count=args.expected_rows,
            expected_language_count=args.expected_languages,
            source=args.source,
            force=args.force,
        )
    except TranslationDataError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
