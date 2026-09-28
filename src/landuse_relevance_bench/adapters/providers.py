"""Model providers for the CLI: lazy runtime dispatch and one-model-at-a-time caching."""

import sys
from collections.abc import Callable
from typing import Generic, TypeVar

from landuse_relevance_bench.adapters.gpu_memory import gpu_memory_line
from landuse_relevance_bench.adapters.pipeline import GeneratorProvider, RunRequest, ScorerProvider
from landuse_relevance_bench.domain.engine import LabelScorer, TextGenerator
from landuse_relevance_bench.domain.roster import SGLANG

T = TypeVar("T")


def generator_provider() -> GeneratorProvider:
    """Dispatch lazily so the CLI works without either optional runtime installed."""

    def provide(request: RunRequest) -> tuple[TextGenerator, str]:
        if request.runtime == SGLANG:
            from landuse_relevance_bench.adapters.sglang_generator import provide as provide_sglang

            return provide_sglang(request)
        from landuse_relevance_bench.adapters.hf_generator import provide as provide_transformers

        return provide_transformers(request)

    return provide


def scorer_provider() -> ScorerProvider:
    """Imported lazily so the CLI stays usable without a model runtime installed."""
    from landuse_relevance_bench.adapters.hf_scorer import provide_scorer

    return provide_scorer


class CachedProvider(Generic[T]):
    """Load one model lazily and reuse it for that model's selected languages."""

    def __init__(
        self,
        factory: Callable[[], Callable[[RunRequest], T]],
        memory_line: Callable[[str, str], str | None] = gpu_memory_line,
    ) -> None:
        self._factory = factory
        self._memory_line = memory_line
        self._loaded: T | None = None
        self._loaded_name: str | None = None

    def __call__(self, request: RunRequest) -> T:
        if self._loaded_name != request.name:
            self.close_cached()
            self._log_memory("before_load", request.name)
            self._loaded = self._factory()(request)
            self._loaded_name = request.name
        if self._loaded is None:
            raise RuntimeError("cached provider failed to load a model")
        return self._loaded

    def close_cached(self) -> None:
        cached = self._loaded
        name = self._loaded_name
        self._loaded = None
        self._loaded_name = None
        if cached is None or name is None:
            return
        instance = cached[0] if isinstance(cached, tuple) else cached
        close = getattr(instance, "close", None)
        if callable(close):
            close()
        self._log_memory("after_close", name)

    def _log_memory(self, event: str, name: str) -> None:
        line = self._memory_line(event, name)
        if line is not None:
            sys.stdout.write(f"{line}\n")
            sys.stdout.flush()


def cached_scorer_provider() -> CachedProvider[tuple[LabelScorer, str]]:
    return CachedProvider(scorer_provider)


def cached_generator_provider() -> CachedProvider[tuple[TextGenerator, str]]:
    return CachedProvider(generator_provider)
