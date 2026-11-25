"""Recorded numerical inputs and direct comparisons for result reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

# These settings change orchestration, not the fitted model or evaluation.
_OPERATIONAL_KEYS = frozenset(
    {
        "config_path",
        "run_type",
        "run_mode",
        "config_stem",
        "results_root",
        "output_root",
        "output_json",
        "system_dir",
        "systems_root",
        "resume",
        "skip_existing",
        "limit",
        "continue_on_error",
        "recursive",
        "batch_num_workers",
        "verbose",
        "benchmark_definition",
        "benchmark_integrity_preflight",
    }
)


def plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return plain({k: v for k, v in config.items() if k not in _OPERATIONAL_KEYS})


def same_array(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_inputs(
    path: Path,
    *,
    times,
    observed,
    state_names,
    metadata,
    evaluation_view,
    benchmark_spec=None,
    fit_states=None,
    predicted=None,
    model_payload=None,
) -> None:
    arrays = {
        "times": np.asarray(times, dtype=np.float64),
        "observed": np.asarray(observed, dtype=np.float64),
    }
    if fit_states is not None:
        arrays["fit_states"] = np.asarray(fit_states, dtype=np.float64)
    if predicted is not None:
        arrays["predicted"] = np.asarray(predicted, dtype=np.float64)
    context = {
        "format_version": 2,
        "state_names": list(state_names),
        "metadata": metadata,
        "evaluation_view": evaluation_view,
        "benchmark_spec": benchmark_spec,
        "model_payload": model_payload,
    }
    arrays["context"] = np.asarray(
        json.dumps(plain(context), sort_keys=True, allow_nan=False)
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_inputs(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as saved:
        context = json.loads(str(saved["context"].item()))
        arrays = {k: saved[k].copy() for k in saved.files if k != "context"}
    if not isinstance(context, dict) or context.get("format_version") != 2:
        raise ValueError("unsupported recorded inputs format")
    t, x = arrays["times"], arrays["observed"]
    if (
        t.dtype != np.dtype("float64")
        or x.dtype != np.dtype("float64")
        or t.ndim != 1
        or x.shape != (len(t), len(context["state_names"]))
        or not len(t)
        or not np.isfinite(t).all()
        or not np.isfinite(x).all()
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("invalid recorded trajectory inputs")
    for key in ("fit_states", "predicted"):
        if key in arrays and (
            arrays[key].dtype != x.dtype
            or arrays[key].shape != x.shape
            or not np.isfinite(arrays[key]).all()
        ):
            raise ValueError(f"invalid recorded {key}")
    return context, arrays


def validate_current_inputs(
    path: Path,
    *,
    system_dir: Path,
    trajectory_name: str,
    benchmark_spec=None,
    run_config=None,
) -> str | None:
    from DataAdapters import SysBioSRSystem

    try:
        context, arrays = load_inputs(path)
        system = SysBioSRSystem.from_dir(system_dir, preload_trajectories=False)
        trajectory = system.get_trajectory(trajectory_name)
        if context["state_names"] != system.state_names:
            return "state_order_changed"
        if context["metadata"] != system.metadata:
            return "trajectory_metadata_changed"
        if not same_array(arrays["times"], trajectory.times) or not same_array(
            arrays["observed"], trajectory.values
        ):
            return "trajectory_inputs_changed"
        if "fit_states" in arrays and run_config is not None:
            from .base_single_system_runner import BaseSingleSystemRunner

            runner = BaseSingleSystemRunner(system_dir, trajectory_name=trajectory_name)
            actual_fit, _ = runner.build_fit_trajectory(**run_config)
            if not same_array(arrays["fit_states"], actual_fit):
                return "fit_inputs_changed"
        if context["evaluation_view"] != system.load_evaluation_view():
            return "evaluation_inputs_changed"
        if context["benchmark_spec"] != benchmark_spec:
            return "benchmark_definition_changed"
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return "recorded_inputs_invalid"
    return None
