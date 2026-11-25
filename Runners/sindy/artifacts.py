"""Read-only validation of SINDy's saved numerical model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_saved_model(saved: dict[str, Any], output_json: Path) -> str | None:
    """Validate a model after the shared artifact checks have passed."""
    from Baseline.SINDy import SINDy

    artifacts = saved.get("artifacts") or {}
    try:
        selected = (
            (saved.get("equations") or {}).get("metadata", {}).get("selected_model")
        )
        if selected is None and saved.get("status") == "success":
            return "sindy_selected_model_missing"
        if selected is not None:
            for key, value in selected.get("config", {}).items():
                if (
                    key in (saved.get("fit_config") or {})
                    and value != saved["fit_config"][key]
                ):
                    return "model_configuration_mismatch"
            SINDy.from_model(selected)
            if selected["state_names"] != (saved.get("system") or {}).get(
                "state_names"
            ):
                return "sindy_state_order_mismatch"
        relative = artifacts.get("sindy_selected_model")
        if (
            relative
            and json.loads((output_json.parent / relative).read_text()) != selected
        ):
            return "sindy_selected_model_mismatch"
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return "sindy_selected_model_invalid"
    return None
