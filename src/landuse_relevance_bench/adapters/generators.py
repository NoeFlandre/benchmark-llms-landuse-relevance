"""Choosing the model runtime a run asks for, importing only that runtime."""

from landuse_relevance_bench.adapters.pipeline import RunRequest
from landuse_relevance_bench.domain.engine import TextGenerator
from landuse_relevance_bench.domain.roster import SGLANG, TRANSFORMERS


def provide(request: RunRequest) -> tuple[TextGenerator, str]:
    """The generator provider used by the CLI, dispatching on ``request.runtime``."""
    if request.runtime == TRANSFORMERS:
        from landuse_relevance_bench.adapters import hf_generator

        return hf_generator.provide(request)
    if request.runtime == SGLANG:
        from landuse_relevance_bench.adapters import sglang_generator

        return sglang_generator.provide(request)
    raise ValueError(f"unknown runtime {request.runtime!r}")
