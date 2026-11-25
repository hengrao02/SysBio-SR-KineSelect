"""Shared structural, numerical and input validation for current results."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile
import numpy as np
import pandas as pd
from .inputs import load_inputs, same_array, validate_current_inputs
from .method_capabilities import get_optional_method_spec


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _contained_file(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
        current = path
        while current != root and current != current.parent:
            if current.is_symlink():
                return None
            current = current.parent
        return path if path.is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


def _same_metrics(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same_metrics(actual[k], value) for k, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same_metrics(a, b) for a, b in zip(actual, expected))
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and bool(np.isclose(actual, expected, rtol=1e-12, atol=1e-14))
        )
    return actual == expected


def _numerical_record(saved, output_json):
    from Metrics.trajectory_metrics import METRIC_VERSION, compute_trajectory_metrics
    from .evaluation_contracts import (
        reaction_evaluation,
        to_jsonable,
        stable_input_model_payload,
    )

    try:
        context, arrays = load_inputs(output_json.parent / saved["artifacts"]["inputs"])
        if context["state_names"] != saved["system"]["state_names"]:
            return "recorded_state_order_mismatch"
        expected_model = to_jsonable(
            stable_input_model_payload(
                method_name=saved["method"],
                method_payload=saved,
                num_states=len(saved["system"]["state_names"]),
            )
        )
        if context.get("model_payload") != expected_model:
            return "metric_model_binding_mismatch"
        if saved.get("status") != "success":
            return None
        path = output_json.parent / saved["artifacts"]["generated_trajectory_csv"]
        frame = pd.read_csv(path, float_precision="round_trip")
        names = context["state_names"]
        if list(frame.columns) != ["time", *names]:
            return "prediction_columns_invalid"
        if not same_array(frame["time"].to_numpy(dtype=float), arrays["times"]):
            return "prediction_times_mismatch"
        predicted = frame[names].to_numpy(dtype=float)
        if not same_array(predicted, arrays["predicted"]):
            return "prediction_values_mismatch"
        calculated = to_jsonable(
            compute_trajectory_metrics(arrays["observed"], predicted, names)
        )
        metrics = saved.get("metrics") or {}
        if metrics.get("metric_version") != METRIC_VERSION:
            return "trajectory_metric_version_mismatch"
        expected_reaction = to_jsonable(
            reaction_evaluation(
                method_name=saved["method"],
                method_payload=saved,
                annotation=context["evaluation_view"].get("reaction_annotation"),
                fit_config=saved["fit_config"],
                num_states=len(names),
            )
        )
        actual_reaction = saved.get("reaction_evaluation") or {}
        for key in ("enabled", "included_in_aggregation", "metric_version", "scores"):
            if actual_reaction.get(key) != expected_reaction.get(key):
                return "reaction_metrics_mismatch"
        for key, expected in calculated.items():
            if not _same_metrics(metrics.get(key), expected):
                return "trajectory_metrics_mismatch"
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        EOFError,
        BadZipFile,
    ):
        return "numerical_record_invalid"
    return None


def validate_declared_artifacts(
    saved: dict[str, Any], *, output_json: Path, require_failure: bool = False
) -> str | None:
    artifacts = saved.get("artifacts")
    if not isinstance(artifacts, dict):
        return "artifacts_missing"
    if require_failure and not artifacts.get("failure"):
        return "failure_artifact_missing"
    for name, relative in artifacts.items():
        if relative is not None:
            path = _contained_file(output_json.parent, relative)
            if path is None:
                return f"artifact_missing_or_unsafe:{name}"
            try:
                if path.suffix == ".json":
                    json.loads(path.read_text())
                if name == "resolved_config":
                    import yaml

                    if yaml.safe_load(path.read_text()) != saved.get("fit_config"):
                        return "saved_configuration_mismatch"
            except (OSError, ValueError, UnicodeError):
                return f"artifact_format_invalid:{name}"
    spec = get_optional_method_spec(str(saved.get("method") or ""))
    if spec is not None:
        if not spec.supports_reaction_metrics:
            reaction = saved.get("reaction_evaluation") or {}
            if not isinstance(reaction, dict):
                return "reaction_metadata_invalid"
            if reaction.get("enabled") or reaction.get("included_in_aggregation"):
                return "reaction_capability_mismatch"
        if spec.validate_saved_model is not None:
            reason = spec.validate_saved_model(saved, output_json)
            if reason:
                return reason
    if saved.get("schema_version") == 2:
        if not artifacts.get("inputs"):
            return "recorded_inputs_missing"
        return _numerical_record(saved, output_json)
    return "unsupported_result_format"


def validate_saved_result(
    saved, *, output_json: Path, expected: dict[str, Any]
) -> str | None:
    from .schemas import result_validator
    from .evaluation_contracts import stable_input_model_payload

    if not isinstance(saved, dict) or saved.get("schema_version") != 2:
        return "unsupported_result_format"
    if list(result_validator().iter_errors(saved)):
        return "result_schema_invalid"
    if saved.get("complete") is not True:
        return "result_not_finalized"
    system, config = saved.get("system") or {}, saved.get("fit_config") or {}
    actual = {
        "method": saved.get("method"),
        "system_id": system.get("system_id"),
        "trajectory_name": system.get("trajectory_name"),
        "dataset_version": config.get("dataset_version"),
        "benchmark_partition": config.get("benchmark_partition"),
        "experiment_id": config.get("experiment_id"),
        "random_seed": config.get(
            "run_seed", config.get("random_seed", config.get("seed", 0))
        ),
        "benchmark_spec": config.get("benchmark_spec"),
        "run_config": saved.get("run_config"),
    }
    for key, value in expected.items():
        if key != "system_dir" and value is not None and actual.get(key) != value:
            return f"configuration_mismatch:{key}"
    reason = validate_declared_artifacts(
        saved, output_json=output_json, require_failure=saved.get("status") != "success"
    )
    if reason:
        return reason
    if expected.get("system_dir") is not None:
        reason = validate_current_inputs(
            output_json.parent / saved["artifacts"]["inputs"],
            system_dir=Path(expected["system_dir"]),
            trajectory_name=system["trajectory_name"],
            benchmark_spec=expected.get("benchmark_spec", config.get("benchmark_spec")),
            run_config=saved.get("run_config"),
        )
        if reason:
            return reason
    if saved.get("status") != "success":
        return None
    spec = get_optional_method_spec(saved["method"])
    if spec is not None and saved.get("record_kind") != "selector_evaluation":
        for name in spec.required_fit_artifacts:
            if not saved["artifacts"].get(name):
                return f"method_artifact_missing:{name}"
    records = saved.get("evaluations") or []
    for role in config.get("eval_trajectory_names") or []:
        matches = [
            r
            for r in records
            if isinstance(r, dict) and r.get("eval_trajectory_name") == role
        ]
        if len(matches) != 1 or matches[0].get("status") != "success":
            return f"evaluation_not_successful:{role}"
        record = matches[0]
        target = (
            output_json.parent.parent.parent
            / role
            / output_json.parent.name
            / "summary.json"
        )
        relative = record.get("output_json_relative_to_fit")
        if (
            not isinstance(relative, str)
            or (output_json.parent / relative).resolve() != target.resolve()
        ):
            return f"evaluation_path_invalid:{role}"
        # The linked result may be a sibling, but never a symlink.
        if target.is_symlink() or not target.is_file():
            return f"evaluation_summary_missing:{role}"
        evaluation = _safe_load_json(target)
        if (
            not evaluation
            or evaluation.get("schema_version") != 2
            or evaluation.get("status") != "success"
            or evaluation.get("complete") is not True
        ):
            return f"evaluation_summary_invalid:{role}"
        es = evaluation.get("system") or {}
        if (
            evaluation.get("method") != saved["method"]
            or es.get("system_id") != system["system_id"]
            or es.get("trajectory_name") != role
            or es.get("state_names") != system["state_names"]
            or evaluation.get("run_config") != saved.get("run_config")
        ):
            return f"evaluation_configuration_mismatch:{role}"
        payload = lambda d: stable_input_model_payload(
            method_name=saved["method"],
            method_payload=d,
            num_states=len(system["state_names"]),
        )
        if payload(saved) != payload(evaluation):
            return f"evaluation_model_mismatch:{role}"
        reason = validate_declared_artifacts(evaluation, output_json=target)
        if reason:
            return f"evaluation_{reason}:{role}"
        if record.get("metrics") != evaluation.get("metrics") or record.get(
            "trajectory_evaluation"
        ) != evaluation.get("trajectory_evaluation"):
            return f"evaluation_link_mismatch:{role}"
        if expected.get("system_dir") is not None:
            reason = validate_current_inputs(
                target.parent / evaluation["artifacts"]["inputs"],
                system_dir=Path(expected["system_dir"]),
                trajectory_name=role,
                benchmark_spec=expected.get(
                    "benchmark_spec", config.get("benchmark_spec")
                ),
            )
            if reason:
                return f"evaluation_{reason}:{role}"
    return None


def validate_resumable_result(
    saved, *, output_json: Path, expected: dict[str, Any]
) -> str | None:
    if not isinstance(saved, dict) or saved.get("status") != "success":
        return "summary_not_successful"
    spec = get_optional_method_spec(str(saved.get("method") or ""))
    if spec is not None and not spec.supports_resume:
        return "automatic_resume_not_supported"
    return validate_saved_result(saved, output_json=output_json, expected=expected)
