"""Runtime kinetic law-space presets for KineSelectSR experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


KINETIC_PRESET_LAW_SPACE: dict[str, dict[str, list[str]]] = {
    "ma_none": {
        "substrate_laws": ["mass_action"],
        "modifier_laws": ["none"],
    },
    "ma_linear": {
        "substrate_laws": ["mass_action"],
        "modifier_laws": ["none", "linear"],
    },
    "ma_mm_linear": {
        "substrate_laws": ["mass_action", "mm"],
        "modifier_laws": ["none", "linear"],
    },
}

_PRESET_ALIASES = {
    "ma-none": "ma_none",
    "ma_none": "ma_none",
    "manone": "ma_none",
    "ma-linear": "ma_linear",
    "ma_linear": "ma_linear",
    "malinear": "ma_linear",
    "ma-mm-linear": "ma_mm_linear",
    "ma_mm_linear": "ma_mm_linear",
    "mammlinear": "ma_mm_linear",
    "onlyma": "ma_linear",
    "only_ma": "ma_linear",
    "nohill": "ma_mm_linear",
    "no_hill": "ma_mm_linear",
}


def normalize_kinetic_preset(value: Any) -> str | None:
    """Normalize a user-facing kinetic preset value.

    ``None`` and an empty string mean "no preset"; otherwise the canonical
    Phase 1 names are returned.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _PRESET_ALIASES.get(text.lower(), text)


def apply_kinetic_preset(
    values: Mapping[str, Any],
    *,
    phase_label: str = "Phase 1",
) -> dict[str, Any]:
    """Return a copy of ``values`` with the requested law-space preset applied."""
    out = dict(values)
    preset = normalize_kinetic_preset(out.get("kinetic_preset"))
    if preset is None:
        return out

    if preset not in KINETIC_PRESET_LAW_SPACE:
        supported = sorted(KINETIC_PRESET_LAW_SPACE)
        raise ValueError(
            f"Unsupported kinetic_preset={preset!r}; expected one of {supported}"
        )

    out["kinetic_preset"] = preset
    for key, value in KINETIC_PRESET_LAW_SPACE[preset].items():
        out[key] = list(value)
    out["kinetic_preset_phase"] = out.get("kinetic_preset_phase") or phase_label
    return out
