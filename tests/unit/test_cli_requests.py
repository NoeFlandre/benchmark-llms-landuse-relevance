"""Pin the RunRequest that run, score and run-all hand to the benchmark, field by field.

A request is what a run is asked to do. These tests record every field each command
builds and compare the whole record, so how requests are constructed can change without
changing what a run is asked to do.
"""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer

from landuse_relevance_bench import cli

SCORER = "LiquidAI/LFM2.5-Encoder-350M"
GENERATIVE = "LiquidAI/LFM2.5-350M"
SGLANG = "LiquidAI/LFM2.5-2.6B@sglang"
GENERATIVE_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
SGLANG_REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"


class _Provider:
    def __init__(self) -> None:
        self.closed = 0

    def close_cached(self) -> None:
        self.closed += 1


def _stub_plan(monkeypatch: pytest.MonkeyPatch, languages: tuple[str, ...]) -> None:
    """Fix the language selection and silence the plan so only requests are observed."""
    manifest = SimpleNamespace(
        files={language: SimpleNamespace(path=Path(f"{language}.csv")) for language in languages}
    )
    monkeypatch.setattr(cli, "selected_languages", lambda *_: (manifest, languages))
    monkeypatch.setattr(cli, "print_run_plan", lambda *args: None)


def _score_request(tmp_path: Path, language: str) -> dict[str, Any]:
    return {
        "model_id": SCORER,
        "language": language,
        "benchmark_path": tmp_path / f"{language}.csv",
        "prompt_path": Path("data/prompt_masked_lm.txt"),
        "output_dir": tmp_path / "results",
        "revision": "rev-1",
        "batch_size": 16,
        "max_new_tokens": 4096,
        "seed": 3,
        "dtype": "float32",
        "run_id": "",
        "runtime": "transformers",
        "vision": False,
        "draft_model_id": "",
        "draft_revision": None,
        "speculative": {},
        "continuous_batching": False,
        "throughput_mode": False,
        "close_generator": False,
    }


def test_score_requests_are_pinned_field_by_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_plan(monkeypatch, ("de", "en"))
    provider = _Provider()
    seen: list[tuple[dict[str, Any], object]] = []
    monkeypatch.setattr(cli, "cached_scorer_provider", lambda: provider)
    monkeypatch.setattr(
        cli, "score_one", lambda request, used: seen.append((asdict(request), used))
    )

    cli.score(
        SCORER,
        data_root=tmp_path,
        out=tmp_path / "results",
        revision="rev-1",
        seed=3,
        dtype="float32",
    )

    assert seen == [
        (_score_request(tmp_path, "de"), provider),
        (_score_request(tmp_path, "en"), provider),
    ]
    assert provider.closed == 1


def test_run_requests_are_pinned_field_by_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_plan(monkeypatch, ("en",))
    provider = _Provider()
    seen: list[tuple[dict[str, Any], object, dict[str, object]]] = []
    monkeypatch.setattr(cli, "cached_generator_provider", lambda: provider)
    monkeypatch.setattr(
        cli,
        "benchmark_one",
        lambda request, used, **kwargs: seen.append((asdict(request), used, kwargs)),
    )

    cli.run(
        GENERATIVE,
        data_root=tmp_path,
        prompt=tmp_path / "prompt.txt",
        out=tmp_path / "results",
        revision="override-rev",
        batch_size=4,
        max_new_tokens=32,
        seed=7,
        dtype="float16",
    )

    expected = {
        "model_id": GENERATIVE,
        "language": "en",
        "benchmark_path": tmp_path / "en.csv",
        "prompt_path": tmp_path / "prompt.txt",
        "output_dir": tmp_path / "results",
        "revision": "override-rev",
        "batch_size": 4,
        "max_new_tokens": 32,
        "seed": 7,
        "dtype": "float16",
        "run_id": "",
        "runtime": "transformers",
        "vision": False,
        "draft_model_id": "",
        "draft_revision": None,
        "speculative": {},
        "continuous_batching": False,
        "throughput_mode": False,
        "close_generator": False,
    }
    assert seen == [(expected, provider, {})]
    assert provider.closed == 1


def test_run_all_requests_are_pinned_for_a_generative_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_plan(monkeypatch, ("de", "en"))
    provider = _Provider()
    seen: list[tuple[dict[str, Any], dict[str, object]]] = []
    monkeypatch.setattr(cli, "selected_models", lambda *_: (GENERATIVE,))
    monkeypatch.setattr(cli, "cached_generator_provider", lambda: provider)
    monkeypatch.setattr(
        cli,
        "benchmark_one",
        lambda request, _used, **kwargs: seen.append((asdict(request), kwargs)),
    )

    cli.run_all(
        data_root=tmp_path,
        prompt=tmp_path / "prompt.txt",
        out=tmp_path / "results",
        batch_size=4,
        max_new_tokens=64,
        seed=5,
        dtype="float16",
    )

    def expected(language: str) -> dict[str, Any]:
        return {
            "model_id": GENERATIVE,
            "language": language,
            "benchmark_path": tmp_path / f"{language}.csv",
            "prompt_path": tmp_path / "prompt.txt",
            "output_dir": tmp_path / "results",
            "revision": GENERATIVE_REVISION,
            "batch_size": 4,
            "max_new_tokens": 64,
            "seed": 5,
            "dtype": "float16",
            "run_id": "",
            "runtime": "transformers",
            "vision": False,
            "draft_model_id": "",
            "draft_revision": None,
            "speculative": {},
            "continuous_batching": False,
            "throughput_mode": False,
            "close_generator": False,
        }

    assert seen == [
        (expected("de"), {"skip_existing": True}),
        (expected("en"), {"skip_existing": True}),
    ]
    assert provider.closed == 1


def test_run_all_requests_are_pinned_for_an_sglang_throughput_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_plan(monkeypatch, ("en",))
    provider = _Provider()
    seen: list[tuple[dict[str, Any], dict[str, object]]] = []
    monkeypatch.setattr(cli, "selected_models", lambda *_: (SGLANG,))
    monkeypatch.setattr(cli, "cached_generator_provider", lambda: provider)
    monkeypatch.setattr(
        cli,
        "benchmark_one",
        lambda request, _used, **kwargs: seen.append((asdict(request), kwargs)),
    )

    cli.run_all(
        data_root=tmp_path,
        prompt=tmp_path / "prompt.txt",
        out=tmp_path / "results",
        throughput=True,
    )

    expected = {
        "model_id": "LiquidAI/LFM2.5-2.6B",
        "language": "en",
        "benchmark_path": tmp_path / "en.csv",
        "prompt_path": tmp_path / "prompt.txt",
        "output_dir": tmp_path / "results",
        "revision": SGLANG_REVISION,
        "batch_size": 16,
        "max_new_tokens": 4096,
        "seed": 0,
        "dtype": "bfloat16",
        "run_id": "LiquidAI/LFM2.5-2.6B@sglang-throughput-b16",
        "runtime": "sglang",
        "vision": False,
        "draft_model_id": "",
        "draft_revision": None,
        "speculative": {"disable_radix_cache": True, "mem_fraction_static": 0.75},
        "continuous_batching": False,
        "throughput_mode": True,
        "close_generator": False,
    }
    assert seen == [(expected, {"skip_existing": True})]
    assert provider.closed == 1


def test_run_refuses_a_scoring_id_before_building_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[object] = []
    monkeypatch.setattr(cli, "benchmark_one", lambda *args, **kwargs: built.append(args))

    with pytest.raises(typer.BadParameter, match="is a scoring model"):
        cli.run(SCORER)

    assert built == []
