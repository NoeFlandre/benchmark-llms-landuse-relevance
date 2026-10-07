"""Import an optional runtime by name, with a clear error naming the extra to install."""

import importlib
from typing import Any


def require(module: str, extra: str) -> Any:
    """Return the imported ``module``, or raise ImportError naming the missing ``extra``."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        message = (
            f"The optional dependency '{module}' is not installed; "
            f"install it with: pip install 'landuse-relevance-bench[{extra}]'"
        )
        raise ImportError(message) from error
