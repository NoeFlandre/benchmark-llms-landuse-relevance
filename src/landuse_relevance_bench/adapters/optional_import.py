"""Import an optional runtime by name, with a clear error naming the extra to install."""

import importlib
from typing import Any


def require(module: str, extra: str) -> Any:
    """Return the imported ``module``, or raise ImportError naming the missing ``extra``.

    Only an absent top-level runtime is rewritten. Any other import failure (a broken
    installed runtime, a missing transitive dependency, a missing internal submodule)
    propagates unchanged so its real cause is not reported as "not installed".
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module or "." in module:
            raise
        message = (
            f"The optional dependency '{module}' is not installed; "
            f"install it with: pip install 'landuse-relevance-bench[{extra}]'"
        )
        raise ImportError(message) from error
