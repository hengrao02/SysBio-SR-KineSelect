"""Read-only validation of FFX's saved numerical model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_saved_model(saved: dict[str, Any], output_json: Path) -> str | None:
    """Validate a model after the shared artifact checks have passed."""
    from Baseline.FFX import FFX

    artifacts = saved.get("artifacts") or {}
    try:
        selected = (
            (saved.get("equations") or {}).get("metadata", {}).get("selected_model")
        )
        if selected is None and saved.get("status") == "success":
            return "ffx_selected_model_missing"
        if selected is not None:
            for key, value in selected.get("config", {}).items():
                if (
                    key in (saved.get("fit_config") or {})
                    and value != saved["fit_config"][key]
                ):
                    return "model_configuration_mismatch"
            predictor = FFX.from_model(selected)
            if predictor.equation() != (saved.get("generated_equation") or {}).get(
                "best"
            ):
                return "ffx_equation_mismatch"
            if predictor.equation() != (saved.get("equations") or {}).get("best"):
                return "ffx_equation_mismatch"
            if selected["state_names"] != (saved.get("system") or {}).get(
                "state_names"
            ):
                return "ffx_state_order_mismatch"
        relative = artifacts.get("ffx_selected_model")
        if (
            relative
            and json.loads((output_json.parent / relative).read_text()) != selected
        ):
            return "ffx_selected_model_mismatch"
        front_path = artifacts.get("ffx_component_fronts")
        if selected is not None and front_path:
            fronts = json.loads((output_json.parent / front_path).read_text())
            indices = selected["front_indices"]
            if (
                len(fronts) != len(indices)
                or len(indices) != len(selected["state_names"])
                or any(type(i) is not int or i < 0 for i in indices)
            ):
                return "ffx_front_indices_invalid"
            if [front[i]["component"] for front, i in zip(fronts, indices)] != selected[
                "components"
            ]:
                return "ffx_selected_front_mismatch"
    except (AttributeError, KeyError, TypeError, ValueError, OSError, IndexError):
        return "ffx_selected_model_invalid"
    return None
