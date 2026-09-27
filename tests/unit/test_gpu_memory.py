import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from landuse_relevance_bench import cli
from landuse_relevance_bench.adapters.gpu_memory import MIB, gpu_memory_line
from landuse_relevance_bench.adapters.pipeline import RunRequest


def _fake_torch(*, available: bool = True, error: bool = False) -> ModuleType:
    def mem_get_info() -> tuple[int, int]:
        if error:
            raise RuntimeError("no context")
        return 3 * MIB + 5, 8 * MIB

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: available, mem_get_info=mem_get_info)
    return torch


def test_gpu_memory_line_reports_free_and_total_mebibytes(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    assert gpu_memory_line("before_load", "org/model") == (
        "gpu_mem event=before_load run=org/model free_mib=3 total_mib=8"
    )


@pytest.mark.parametrize("torch", [None, _fake_torch(available=False), _fake_torch(error=True)])
def test_gpu_memory_line_is_silent_without_torch_or_cuda(monkeypatch, torch) -> None:
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert gpu_memory_line("after_close", "org/model") is None


def test_cached_provider_logs_memory_before_load_and_after_close(capsys) -> None:
    events: list[str] = []

    class Generator:
        def close(self) -> None:
            events.append("close")

    def memory_line(event: str, run: str) -> str:
        events.append(event)
        return f"gpu_mem event={event} run={run} free_mib=1 total_mib=2"

    provider = cli._CachedProvider(lambda: lambda _request: (Generator(), "rev"), memory_line)
    request = RunRequest.for_run(
        "LiquidAI/LFM2.5-350M",
        language="en",
        benchmark_path=Path("benchmark.csv"),
        prompt_path=Path("prompt.txt"),
        output_dir=Path("results"),
    )
    provider(request)
    provider(request)
    provider.close_cached()

    assert events == ["before_load", "close", "after_close"]
    assert capsys.readouterr().out.splitlines() == [
        f"gpu_mem event=before_load run={request.name} free_mib=1 total_mib=2",
        f"gpu_mem event=after_close run={request.name} free_mib=1 total_mib=2",
    ]
