"""Read-only numerical model validation for ReactiveSINDy."""

import json


def validate_saved_model(saved, output_json):
    from Baseline.ReactiveSINDy import ReactiveSINDy

    try:
        model = (saved.get("equations") or {}).get("metadata", {}).get("selected_model")
        if model is None:
            return (
                "selected_model_missing" if saved.get("status") == "success" else None
            )
        ReactiveSINDy.from_model(model)
        if model["state_names"] != saved["system"]["state_names"]:
            return "state_order_mismatch"
        for key, value in model["config"].items():
            if key in saved.get("fit_config", {}) and value != saved["fit_config"][key]:
                return "model_configuration_mismatch"
        relative = saved.get("artifacts", {}).get("reactive_selected_model_json")
        if (
            relative
            and json.loads((output_json.parent / relative).read_text()) != model
        ):
            return "selected_model_mismatch"
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return "selected_model_invalid"
    return None
