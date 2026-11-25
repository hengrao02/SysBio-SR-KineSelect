from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from DataAdapters import SysBioSRSystem
    from Metrics.trajectory_metrics import (
        METRIC_VERSION as TRAJECTORY_METRIC_VERSION,
        compute_trajectory_metrics,
    )
    from Utils.plotting import save_trajectory_overlay_plot
    from Utils.result_status import decide_result_status, normalize_failure_reason
    from Utils.paths import repository_relative, sanitize_paths
except ImportError:  # pragma: no cover - package import fallback
    from ...DataAdapters import SysBioSRSystem
    from ...Metrics.trajectory_metrics import (
        METRIC_VERSION as TRAJECTORY_METRIC_VERSION,
        compute_trajectory_metrics,
    )
    from ...Utils.plotting import save_trajectory_overlay_plot
    from ...Utils.result_status import decide_result_status, normalize_failure_reason
    from ...Utils.paths import repository_relative, sanitize_paths

from .exceptions import is_nonfatal_timeout_exception
from .inputs import save_inputs, scientific_config, atomic_json
from .evaluation_contracts import (
    stable_input_model_payload,
    coerce_bool,
    reaction_evaluation,
    to_jsonable,
)


class BaseSingleSystemRunner:
    """Shared runner logic for single-system baseline execution."""

    persist_equations_in_evaluation = True

    def __init__(
        self, system_dir: str | Path, *, trajectory_name: str = "default"
    ) -> None:
        # Load only the fit trajectory during construction.  Evaluation
        # trajectories (notably ``perturbed``) remain unopened until after the
        # wrapper has fitted and frozen its model.
        self.system = SysBioSRSystem.from_dir(system_dir, preload_trajectories=False)
        self.trajectory_name = trajectory_name
        self.fit_trajectory_record = self.system.get_trajectory(trajectory_name)
        self.fit_times = self.fit_trajectory_record.times.copy()
        self.fit_trajectory = self.fit_trajectory_record.values.copy()
        self.fit_y0 = self.fit_trajectory_record.rollout_initial_condition()
        self.fit_trajectory_record.assert_rollout_initial_condition(self.fit_y0)

    @staticmethod
    def to_jsonable(obj: Any) -> Any:
        return to_jsonable(obj)

    @staticmethod
    def _environment_payload() -> dict[str, Any]:
        packages: dict[str, str] = {}
        for name in (
            "numpy",
            "scipy",
            "pandas",
            "sympy",
            "PyYAML",
            "matplotlib",
            "tqdm",
            "zstandard",
            "jsonschema",
        ):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not-installed"
        return {
            "schema_version": 1,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.system(),
            "platform_release": platform.release(),
            "machine": platform.machine(),
            "packages": packages,
            "determinism": {
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unrecorded"),
                "openblas_num_threads": os.environ.get(
                    "OPENBLAS_NUM_THREADS", "unrecorded"
                ),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "unrecorded"),
            },
            "accelerator": {
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "unrecorded"
                ),
                "cuda_version": os.environ.get("CUDA_VERSION", "not-detected"),
                "cudnn_version": os.environ.get("CUDNN_VERSION", "not-detected"),
            },
        }

    def load_original_equation_info(self) -> dict[str, Any]:
        if self.system.equation_path is None or not self.system.equation_path.exists():
            return {}
        return json.loads(self.system.equation_path.read_text(encoding="utf-8"))

    def _reaction_evaluation(
        self,
        *,
        method_name: str,
        method_payload: dict[str, Any],
        annotation: dict[str, Any] | None,
        fit_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate reaction recovery after fitting has completed.

        The runner calls this only from the artifact/evaluation stage. Method
        wrappers never receive the annotation object or its path.
        """
        return reaction_evaluation(
            method_name=method_name,
            method_payload=method_payload,
            annotation=annotation,
            fit_config=fit_config,
            num_states=len(self.system.state_names),
        )

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool = False) -> bool:
        return coerce_bool(value, default=default)

    def get_fit_noise_config(self, **run_kwargs: Any) -> dict[str, Any]:
        noise_type = (
            str(run_kwargs.get("fit_noise_type", "none") or "none").strip().lower()
        )
        noise_mode = (
            str(run_kwargs.get("fit_noise_mode", "relative_std") or "relative_std")
            .strip()
            .lower()
        )
        noise_scale = float(run_kwargs.get("fit_noise_scale", 0.0) or 0.0)
        base_seed = int(run_kwargs.get("fit_noise_seed", 0) or 0)
        token = f"{base_seed}:{self.system.system_id}:{self.trajectory_name}"
        derived_seed = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little"
        ) % (2**32)
        enabled = (
            noise_type not in {"", "none", "false", "off", "clean"}
            and noise_scale > 0.0
        )
        return {
            "enabled": enabled,
            "type": noise_type,
            "scale": noise_scale,
            "mode": noise_mode,
            "seed": base_seed,
            "derived_seed": derived_seed,
            "clip_nonnegative": self._coerce_bool(
                run_kwargs.get("fit_noise_clip_nonnegative"), default=False
            ),
            "preserve_initial": self._coerce_bool(
                run_kwargs.get("fit_noise_preserve_initial"), default=True
            ),
        }

    def _noise_sigma(
        self, trajectory: np.ndarray, *, scale: float, mode: str
    ) -> np.ndarray:
        trajectory = np.asarray(trajectory, dtype=float)
        if mode == "absolute":
            return np.full(trajectory.shape[1], float(scale), dtype=float)
        if mode == "relative_range":
            span = np.nanmax(trajectory, axis=0) - np.nanmin(trajectory, axis=0)
            return float(scale) * np.where(span > 0.0, span, 1.0)
        if mode == "relative_mean_abs":
            mean_abs = np.nanmean(np.abs(trajectory), axis=0)
            return float(scale) * np.where(mean_abs > 0.0, mean_abs, 1.0)
        if mode == "relative_std":
            std = np.nanstd(trajectory, axis=0)
            fallback = np.nanmean(np.abs(trajectory), axis=0)
            scale_basis = np.where(
                std > 0.0, std, np.where(fallback > 0.0, fallback, 1.0)
            )
            return float(scale) * scale_basis
        raise ValueError(
            f"Unsupported fit_noise_mode={mode!r}. "
            "Expected one of: relative_std, relative_range, relative_mean_abs, absolute."
        )

    def build_fit_trajectory(
        self, **run_kwargs: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        noise_config = self.get_fit_noise_config(**run_kwargs)
        clean = np.asarray(self.fit_trajectory, dtype=float)
        if not noise_config["enabled"]:
            return clean, noise_config

        if noise_config["type"] != "gaussian":
            raise ValueError("Only fit_noise_type='gaussian' is currently supported.")

        sigma = self._noise_sigma(
            clean, scale=noise_config["scale"], mode=noise_config["mode"]
        )
        rng = np.random.default_rng(int(noise_config["derived_seed"]))
        noisy = clean + rng.normal(loc=0.0, scale=sigma[None, :], size=clean.shape)
        if noise_config["clip_nonnegative"]:
            noisy = np.maximum(noisy, 0.0)
        if noise_config["preserve_initial"] and noisy.shape[0] > 0:
            noisy[0] = clean[0]

        noise_config = {
            **noise_config,
            "per_state_sigma": sigma.tolist(),
        }
        return noisy, noise_config

    def _save_result(
        self,
        *,
        times: np.ndarray,
        y0: np.ndarray,
        gt_trajectory: np.ndarray,
        output_json: Path,
        method_name: str,
        equation_result: Any,
        fit_config: dict[str, Any],
        extra_payload: dict[str, Any],
        plot_title: str,
        wrapper: Any = None,
        pred_trajectory: np.ndarray | None = None,
        failure_reason: str | None = None,
        error: str | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        traceback_text: str | None = None,
    ) -> Path:
        """Unified rollout + artifact save. Returns path to written summary.json."""
        output_json.parent.mkdir(parents=True, exist_ok=True)
        failure_reason = normalize_failure_reason(failure_reason)
        error = None if error is None else str(error)
        predict_kwargs = {} if predict_kwargs is None else dict(predict_kwargs)
        result_trajectory_name = str(
            extra_payload.get("eval_trajectory_name") or self.trajectory_name
        )
        result_trajectory = self.system.get_trajectory(result_trajectory_name)
        result_trajectory.assert_rollout_initial_condition(y0)
        initial_condition_audit = result_trajectory.initial_condition_audit()

        pred_arr = None
        if pred_trajectory is not None:
            pred_arr = np.asarray(pred_trajectory, dtype=float)
        elif wrapper is not None and failure_reason is None:
            try:
                pred = wrapper.predict(times, y0, **predict_kwargs)
                pred_arr = np.asarray(pred, dtype=float) if pred is not None else None
            except Exception as exc:
                failure_reason = (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else getattr(exc, "failure_reason", "rollout_failed")
                )
                error = repr(exc)
                traceback_text = traceback.format_exc()
            except BaseException as exc:
                if not is_nonfatal_timeout_exception(exc):
                    raise
                failure_reason = "timeout"
                error = repr(exc)
                traceback_text = traceback.format_exc()

        metrics = None
        pred_csv_path = None
        plot_path = None
        output_dir = output_json.parent

        if pred_arr is not None:
            if pred_arr.shape != gt_trajectory.shape:
                failure_reason = failure_reason or "invalid_prediction_shape"
            else:
                pred_csv_path = output_dir / "generated_trajectory.csv"
                try:
                    df = pd.DataFrame(pred_arr, columns=self.system.state_names)
                    df.insert(0, "time", times)
                    df.to_csv(pred_csv_path, index=False)
                except Exception as exc:
                    pred_csv_path = None
                    failure_reason = failure_reason or "artifact_save_failed"
                    error = error or repr(exc)

                try:
                    metrics = compute_trajectory_metrics(
                        gt_trajectory, pred_arr, state_names=self.system.state_names
                    )
                except Exception as exc:
                    metrics = None
                    failure_reason = failure_reason or "metrics_failed"
                    error = error or repr(exc)

                plot_path = output_dir / "overlay.png"
                try:
                    save_trajectory_overlay_plot(
                        times,
                        gt_trajectory,
                        pred_arr,
                        self.system.state_names,
                        plot_path,
                        title=plot_title,
                    )
                except Exception as exc:
                    plot_path = None
                    failure_reason = failure_reason or "artifact_save_failed"
                    error = error or repr(exc)

        best_equation = getattr(equation_result, "best", None)
        equation_metadata = getattr(equation_result, "metadata", {}) or {}
        best_equation_substituted = equation_metadata.get("best_substituted")
        if best_equation_substituted is None and isinstance(best_equation, str):
            best_equation_substituted = [
                part.strip() for part in best_equation.split("|")
            ]

        final_status = decide_result_status(
            best_equation=best_equation,
            num_states=len(self.system.state_names),
            pred_trajectory=pred_arr,
            target_trajectory=gt_trajectory,
            metrics=metrics,
            failure_reason=failure_reason,
            error=error,
        )

        # This is the first point at which ground-truth equation/annotation
        # files are loaded. The wrapper has already finished fit and rollout.
        evaluation_view = self.system.load_evaluation_view()
        original_equation = evaluation_view.get("equation") or {}
        fit_config.setdefault("method_version", 1)
        sanitized_fit_config = self.to_jsonable(sanitize_paths(fit_config))
        run_id = "/".join(
            str(value)
            for value in (
                fit_config.get("experiment_id", "adhoc"),
                method_name,
                fit_config.get("dataset_version", "unversioned"),
                fit_config.get("benchmark_partition", "unpartitioned"),
                self.system.system_id,
                result_trajectory_name,
                f"seed-{fit_config.get('run_seed', 0)}",
            )
        )
        method_payload = {
            "generated_equation": {
                "best": best_equation,
                "best_substituted": best_equation_substituted,
            },
            "equations": self.to_jsonable(equation_result),
        }
        metrics_included = final_status.status == "success" and metrics is not None
        trajectory_evaluation = {
            "metric_version": TRAJECTORY_METRIC_VERSION,
            "included_in_aggregation": metrics_included,
            "exclusion_reason": (
                None
                if metrics_included
                else (final_status.failure_reason or "no_metrics")
            ),
            "protocol": (
                "fit_reconstruction"
                if result_trajectory_name == self.trajectory_name
                else f"{self.trajectory_name}_fit_to_{result_trajectory_name}_rollout"
            ),
            **initial_condition_audit,
        }
        if metrics is not None:
            metrics = {
                **metrics,
                "metric_version": TRAJECTORY_METRIC_VERSION,
                "included_in_aggregation": metrics_included,
                "exclusion_reason": trajectory_evaluation["exclusion_reason"],
            }
        reaction_evaluation = self._reaction_evaluation(
            method_name=method_name,
            method_payload=method_payload,
            annotation=evaluation_view.get("reaction_annotation"),
            fit_config=fit_config,
        )
        inputs_path = output_dir / "inputs.npz"
        save_inputs(
            inputs_path,
            times=times,
            observed=gt_trajectory,
            state_names=self.system.state_names,
            metadata=self.system.metadata,
            evaluation_view=evaluation_view,
            model_payload=stable_input_model_payload(
                method_name=method_name,
                method_payload=method_payload,
                num_states=len(self.system.state_names),
            ),
            benchmark_spec=fit_config.get("benchmark_spec"),
            fit_states=(
                self.build_fit_trajectory(**getattr(self, "_run_settings", {}))[0]
                if result_trajectory_name == self.trajectory_name
                else None
            ),
            predicted=pred_arr if final_status.status == "success" else None,
        )
        payload = {
            "schema_version": 2,
            "complete": result_trajectory_name != self.trajectory_name,
            "run_config": scientific_config(
                {"method_version": 1, **getattr(self, "_run_settings", {})}
            ),
            "run_id": run_id,
            "status": final_status.status,
            "failure_reason": final_status.failure_reason,
            "error": final_status.error,
            "method": method_name,
            "system": {
                "system_id": self.system.system_id,
                "state_names": self.system.state_names,
                "const_names": self.system.const_names,
                "trajectory_name": result_trajectory_name,
            },
            "original_equation": {
                "eq": original_equation.get("eq"),
                "eq_raw": original_equation.get("eq_raw"),
                "substituted": original_equation.get("substituted"),
                "state_names": original_equation.get("state_names"),
                "const_names": original_equation.get("const_names"),
            },
            "generated_equation": {
                "best": best_equation,
                "best_substituted": best_equation_substituted,
            },
            "fit_config": sanitized_fit_config,
            "artifacts": {
                "inputs": "inputs.npz",
                "generated_trajectory_csv": (
                    None if pred_csv_path is None else pred_csv_path.name
                ),
                "overlay_plot": None if plot_path is None else plot_path.name,
            },
            "metrics": metrics,
            "trajectory_evaluation": trajectory_evaluation,
            "reaction_evaluation": reaction_evaluation,
            **extra_payload,
        }
        if self.persist_equations_in_evaluation:
            # Extrapolation records must retain the numerical model they score.
            payload["equations"] = method_payload["equations"]
        environment_path = output_dir / "environment.json"
        environment_path.write_text(
            json.dumps(self._environment_payload(), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        resolved_config_path = output_dir / "resolved_config.yaml"
        resolved_config_path.write_text(
            yaml.safe_dump(sanitized_fit_config, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        payload["artifacts"].update(
            {
                "environment": environment_path.name,
                "resolved_config": resolved_config_path.name,
            }
        )
        if final_status.status != "success":
            failure_path = output_dir / "failure.json"
            failure_payload = {
                "schema_version": 1,
                "stage": "fit_or_rollout",
                "status": final_status.status,
                "failure_reason": final_status.failure_reason,
                "exception_type": (
                    None
                    if not final_status.error
                    else str(final_status.error).split("(", 1)[0]
                ),
                "message": final_status.error,
                "traceback": traceback_text,
                "elapsed_seconds": (
                    time.monotonic() - self._run_started_monotonic
                    if hasattr(self, "_run_started_monotonic")
                    else None
                ),
                "retry_count": 0,
            }
            failure_path.write_text(
                json.dumps(
                    sanitize_paths(failure_payload),
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            payload["artifacts"]["failure"] = failure_path.name
        atomic_json(output_json, self.to_jsonable(sanitize_paths(payload)))
        return output_json

    def eval_single_trajectory(
        self,
        wrapper: Any,
        eval_trajectory_name: str,
        *,
        method_name: str,
        results_root: Path,
        run_mode: str,
        config_stem: str | None = None,
        fit_config: dict[str, Any],
        equation_result: Any,
        predict_kwargs: dict[str, Any] | None = None,
        fit_output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Rollout the fitted wrapper on a different trajectory and save results."""
        from .common import build_result_path

        try:
            eval_traj = self.system.get_trajectory(eval_trajectory_name)
        except KeyError as exc:
            return {
                "eval_trajectory_name": eval_trajectory_name,
                "status": "failed",
                "failure_reason": "trajectory_not_found",
                "error": repr(exc),
            }

        eval_times = eval_traj.times.copy()
        eval_y0 = eval_traj.rollout_initial_condition()
        eval_traj.assert_rollout_initial_condition(eval_y0)

        output_json = self._save_result(
            times=eval_times,
            y0=eval_y0,
            gt_trajectory=eval_traj.values.copy(),
            output_json=build_result_path(
                results_root,
                trajectory_name=eval_trajectory_name,
                method_name=method_name,
                run_mode=run_mode,
                system_id=self.system.system_id,
                config_stem=config_stem,
                experiment_id=str(fit_config.get("experiment_id") or "adhoc"),
                dataset_version=str(fit_config.get("dataset_version") or "unversioned"),
                benchmark_partition=str(
                    fit_config.get("benchmark_partition") or "unpartitioned"
                ),
                seed=int(
                    fit_config.get(
                        "run_seed",
                        fit_config.get("random_seed", fit_config.get("seed", 0)),
                    )
                ),
            ),
            method_name=method_name,
            equation_result=equation_result,
            fit_config=fit_config,
            extra_payload={
                "eval_trajectory_name": eval_trajectory_name,
                "fit_trajectory_name": self.trajectory_name,
            },
            plot_title=f"{method_name}: ground-truth vs rollout ({eval_trajectory_name})",
            wrapper=wrapper,
            predict_kwargs=predict_kwargs,
        )
        saved = json.loads(output_json.read_text(encoding="utf-8"))
        return {
            "eval_trajectory_name": eval_trajectory_name,
            "status": saved.get("status"),
            "output_json": repository_relative(output_json),
            "output_json_relative_to_fit": (
                None
                if fit_output_dir is None
                else Path(os.path.relpath(output_json, start=fit_output_dir)).as_posix()
            ),
            "metrics": saved.get("metrics"),
            "trajectory_evaluation": saved.get("trajectory_evaluation"),
            "reaction_evaluation": saved.get("reaction_evaluation"),
        }

    def save_summary(
        self,
        *,
        method_name: str,
        fit_config: dict[str, Any],
        equation_result: Any,
        pred_trajectory: np.ndarray | None,
        output_json: str | Path,
        failure_reason: str | None = None,
        error: str | None = None,
        traceback_text: str | None = None,
    ) -> Path:
        return self._save_result(
            times=self.fit_times,
            y0=self.fit_y0,
            gt_trajectory=self.fit_trajectory,
            output_json=Path(output_json),
            method_name=method_name,
            equation_result=equation_result,
            fit_config=fit_config,
            extra_payload={"equations": equation_result},
            plot_title=f"{method_name}: fit trajectory vs generated-equation rollout",
            pred_trajectory=pred_trajectory,
            failure_reason=failure_reason,
            error=error,
            traceback_text=traceback_text,
        )
