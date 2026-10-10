"""Publishing run results to a Hugging Face dataset repository.

The card and the viewer export are built in ``adapters/publishing``. This module keeps
the Hub upload and the public names the rest of the project imports from here.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from landuse_relevance_bench.adapters.publishing.card import dataset_card
from landuse_relevance_bench.adapters.publishing.card_agreement import DSPARK_COLUMNS
from landuse_relevance_bench.adapters.publishing.card_overview import CARD_COLUMNS
from landuse_relevance_bench.adapters.publishing.card_scoring import (
    SCORING_SUMMARY_CARD_COLUMNS,
)
from landuse_relevance_bench.adapters.publishing.card_speed import (
    EXPECTED_FULL_SWEEP_LANGUAGE_COUNT,
    SPEED_COLUMNS,
)
from landuse_relevance_bench.adapters.publishing.viewer import write_viewer_dataset
from landuse_relevance_bench.adapters.results_store import ARCHIVE_COMPONENT, read_runs
from landuse_relevance_bench.domain.records import RunResult

# Public names kept importable from here for one release after the card moved into
# adapters/publishing.
__all__ = [
    "CARD_COLUMNS",
    "DSPARK_COLUMNS",
    "EXPECTED_FULL_SWEEP_LANGUAGE_COUNT",
    "SCORING_SUMMARY_CARD_COLUMNS",
    "SPEED_COLUMNS",
    "DatasetHub",
    "dataset_card",
    "publish_results",
    "read_published_runs",
    "write_viewer_dataset",
]


class DatasetHub(Protocol):
    """The slice of :class:`huggingface_hub.HfApi` this project depends on."""

    def create_repo(self, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...


def read_published_runs(results_dir: Path) -> tuple[RunResult, ...]:
    """Read every result below ``results_dir`` in stable model-id order."""
    runs = read_runs(results_dir)
    return tuple(
        sorted(runs, key=lambda result: (result.metadata.model_id, result.metadata.language))
    )


def publish_results(
    repo_id: str,
    results_dir: Path,
    results: Sequence[RunResult],
    *,
    api: DatasetHub | None = None,
    private: bool = False,
    commit_message: str = "Publish small-LLM land-use relevance benchmark results",
    benchmark_name: str = "v3-multilingual",
    prompt_text: str,
    scorer_prompt_text: str = "",
    extra_scorer_prompt_texts: Sequence[str] = (),
    timing_results: Sequence[RunResult] = (),
    data_root: Path | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> str:
    """Write the card next to the results, then push the whole folder to the Hub."""
    if not results:
        raise ValueError("refusing to publish an empty set of results")
    _reject_archive_paths(results_dir)
    hub = api if api is not None else _default_api()
    results_dir.mkdir(parents=True, exist_ok=True)
    viewer_file = ""
    if data_root is not None:
        write_viewer_dataset(
            data_root,
            results_dir / "data" / "train.csv",
            languages=sorted({result.metadata.language for result in results}),
        )
        viewer_file = "data/train.csv"
    (results_dir / "README.md").write_text(
        dataset_card(
            results,
            benchmark_name=benchmark_name,
            prompt_text=prompt_text,
            scorer_prompt_text=scorer_prompt_text,
            extra_scorer_prompt_texts=extra_scorer_prompt_texts,
            timing_results=timing_results,
            viewer_file=viewer_file,
        ),
        encoding="utf-8",
    )
    hub.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    upload_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "folder_path": str(results_dir),
        "commit_message": commit_message,
    }
    if allow_patterns is not None:
        upload_kwargs["allow_patterns"] = list(allow_patterns)
    hub.upload_folder(**upload_kwargs)
    return f"https://huggingface.co/datasets/{repo_id}"


def _default_api() -> DatasetHub:
    from huggingface_hub import HfApi

    # HfApi satisfies DatasetHub in practice; its **kwargs signatures are wider than
    # the protocol can express.
    return cast(DatasetHub, HfApi())


def _reject_archive_paths(results_dir: Path) -> None:
    paths = results_dir.rglob("*") if results_dir.exists() else ()
    for path in paths:
        if ARCHIVE_COMPONENT in path.relative_to(results_dir).parts:
            raise ValueError(f"refusing to upload archive path: {path}")
