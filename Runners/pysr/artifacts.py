"""Read-only validation of PySRModel's saved numerical model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_saved_model(saved: dict[str, Any], output_json: Path) -> str | None:
    """Validate a model after the shared artifact checks have passed."""
    from Baseline.PySR import PySRModel

    artifacts = saved.get("artifacts") or {}
    try:
        selected = (
            (saved.get("equations") or {}).get("metadata", {}).get("selected_model")
        )
        if selected is None and saved.get("status") == "success":
            return "pysr_selected_model_missing"
        if selected is not None:
            for key, value in selected.get("config", {}).items():
                if (
                    key in (saved.get("fit_config") or {})
                    and value != saved["fit_config"][key]
                ):
                    return "model_configuration_mismatch"
            predictor = PySRModel.from_model(selected)
            if predictor.equation() != (saved.get("generated_equation") or {}).get(
                "best"
            ):
                return "pysr_equation_mismatch"
            if selected["rhs_expressions"] != (
                saved.get("generated_equation") or {}
            ).get("best_substituted"):
                return "pysr_equation_mismatch"
            if predictor.equation() != (saved.get("equations") or {}).get("best"):
                return "pysr_equation_mismatch"
            if selected["state_names"] != (saved.get("system") or {}).get(
                "state_names"
            ):
                return "pysr_state_order_mismatch"
        relative = artifacts.get("pysr_selected_model")
        if (
            relative
            and json.loads((output_json.parent / relative).read_text()) != selected
        ):
            return "pysr_selected_model_mismatch"
        hof_path = artifacts.get("pysr_hall_of_fame")
        if selected is not None and hof_path:
            hof = json.loads((output_json.parent / hof_path).read_text())
            indices = selected["hof_indices"]
            if (
                len(hof) != len(selected["state_names"])
                or len(indices) != len(hof)
                or any(type(i) is not int or i < 0 for i in indices)
            ):
                return "pysr_hall_of_fame_invalid"
            if [rows[i]["expression"] for rows, i in zip(hof, indices)] != selected[
                "rhs_expressions"
            ]:
                return "pysr_hall_of_fame_mismatch"
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        SyntaxError,
    ):
        return "pysr_selected_model_invalid"
    return None
