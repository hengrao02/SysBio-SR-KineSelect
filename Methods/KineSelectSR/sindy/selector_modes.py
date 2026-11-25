"""Public names and compatibility aliases for kinetic-form selectors."""

from __future__ import annotations


KFI_SELECTOR = "kfi"
CANONICAL_SELECTOR_MODES = frozenset({KFI_SELECTOR})
SELECTOR_MODE_ALIASES = {"contribution": KFI_SELECTOR}


def normalize_selector_mode(value: str, *, field: str = "selector_mode") -> str:
    """Return the canonical selector name while accepting historical inputs."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}")
    canonical = SELECTOR_MODE_ALIASES.get(value, value)
    if canonical not in CANONICAL_SELECTOR_MODES:
        raise ValueError(
            f"{field} must be one of {sorted(CANONICAL_SELECTOR_MODES)}, "
            f"or legacy alias 'contribution'; got {value!r}"
        )
    return canonical
