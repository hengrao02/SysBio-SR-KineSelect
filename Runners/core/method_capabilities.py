"""Read method capabilities from the existing lazy runner registration."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common import MethodSpec

REACTION_UNSUPPORTED = "method_reaction_extraction_unsupported"


def get_optional_method_spec(method: str) -> MethodSpec | None:
    """Read existing registration lazily, allowing unregistered legacy/custom methods.

    Missing dependencies inside a registered runner must propagate, rather than
    silently disabling that method's validation.
    """
    if not method:
        return None
    name = f"Runners.{method}"
    try:
        module = import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return None
    return getattr(module, "METHOD_SPEC", None)


def method_supports_reaction_metrics(method: str) -> bool:
    return getattr(get_optional_method_spec(method), "supports_reaction_metrics", True)
