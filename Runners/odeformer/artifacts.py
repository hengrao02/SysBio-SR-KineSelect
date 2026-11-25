"""Read-only validation of ODEFormer's saved numerical model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_saved_model(saved: dict[str, Any], output_json: Path) -> str | None:
    """Validate a model after the shared artifact checks have passed."""
    from Baseline.ODEFormer import ODEFormer

    artifacts = saved.get("artifacts") or {}
    try:
        selected = (
            (saved.get("equations") or {}).get("metadata", {}).get("selected_model")
        )
        if selected is None and saved.get("status") == "success":
            return "odeformer_selected_model_missing"
        if selected is not None:
            for key, value in selected.get("config", {}).items():
                if (
                    key in (saved.get("fit_config") or {})
                    and value != saved["fit_config"][key]
                ):
                    return "model_configuration_mismatch"
            predictor = ODEFormer.from_model(selected)
            if predictor.equation() != (saved.get("generated_equation") or {}).get(
                "best"
            ):
                return "odeformer_equation_mismatch"
            if predictor.equation() != (saved.get("equations") or {}).get("best"):
                return "odeformer_equation_mismatch"
            if selected["state_names"] != (saved.get("system") or {}).get(
                "state_names"
            ):
                return "odeformer_state_order_mismatch"
        relative = artifacts.get("odeformer_selected_model")
        if (
            relative
            and json.loads((output_json.parent / relative).read_text()) != selected
        ):
            return "odeformer_selected_model_mismatch"
        candidate_path = artifacts.get("odeformer_candidate_results")
        if selected is not None and candidate_path:
            candidates = json.loads((output_json.parent / candidate_path).read_text())
            index = selected["selected_index"]
            if type(index) is not int or index < 0:
                return "odeformer_candidate_index_invalid"
            candidate = candidates["candidates"][index]
            if (
                candidate.get("status") != "success"
                or candidate.get("prefixes") != selected["prefixes"]
                or candidate.get("rhs_expressions") != selected["rhs_expressions"]
                or candidates["summary"].get("selected_index") != index
            ):
                return "odeformer_selected_candidate_mismatch"
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        IndexError,
        SyntaxError,
    ):
        return "odeformer_selected_model_invalid"
    return None
