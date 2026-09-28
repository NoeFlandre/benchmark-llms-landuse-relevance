"""Compatibility shims applied inside SGLang processes.

SGLang's DSpark worker reads ``target_model.lm_head`` and asks the target to capture
auxiliary hidden states, but ``Lfm2VlForConditionalGeneration`` keeps both on its inner
``language_model``. Forwarding them to that module reuses the same weights, so outputs
are unchanged (verified byte-identical against the no-draft run, see ADR 0010).
"""

import os
from pathlib import Path
from typing import Any

SITE_DIR = Path(__file__).with_name("sglang_site")
_FORWARDED = ("set_dflash_layers_to_capture", "get_embed_and_head", "set_embed_and_head")


def forward_to_language_model(model_class: Any) -> None:
    """Expose the inner language model's head and capture hooks on a VL wrapper class."""
    if not hasattr(model_class, "lm_head"):
        model_class.lm_head = property(lambda self: self.language_model.lm_head)
    for name in _FORWARDED:
        if not hasattr(model_class, name):
            setattr(model_class, name, _forwarder(name))


def _forwarder(name: str) -> Any:
    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        return getattr(self.language_model, name)(*args, **kwargs)

    forward.__name__ = name
    return forward


def patch_lfm2_vl() -> None:
    """Patch SGLang's LFM2-VL model class when SGLang is importable."""
    try:
        from sglang.srt.models import lfm2_vl  # ty: ignore[unresolved-import]
    except ImportError:
        return
    forward_to_language_model(lfm2_vl.Lfm2VlForConditionalGeneration)


def with_site_dir(environ: dict[str, str]) -> dict[str, str]:
    """Return ``environ`` with the shim directory first on ``PYTHONPATH``."""
    existing = environ.get("PYTHONPATH", "")
    parts = [str(SITE_DIR), *(p for p in existing.split(os.pathsep) if p and p != str(SITE_DIR))]
    return {**environ, "PYTHONPATH": os.pathsep.join(parts)}
