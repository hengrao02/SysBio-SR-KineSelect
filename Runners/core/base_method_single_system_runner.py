from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import json
import time
import traceback
from typing import Any

from Utils import detect_invalid_equation_dimension
from Utils.paths import sanitize_paths
from Wrappers.base_ode_sr_wrapper import EquationResult

from .base_single_system_runner import BaseSingleSystemRunner
from .exceptions import is_nonfatal_timeout_exception


class BaseMethodSingleSystemRunner(BaseSingleSystemRunner, ABC):
    """Template runner that centralizes the shared single-system fit/predict flow."""

    initial_equation_metadata_exclude_keys: tuple[str, ...] = ()

    @abstractmethod
    def build_wrapper(self, *, output_json: Path, **run_kwargs: Any) -> Any:
        """Construct the method-specific wrapper."""

    @abstractmethod
    def build_method_fit_config(
        self,
        *,
        wrapper: Any,
        output_json: Path,
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        """Return method-specific config fields to be saved in summary.json."""

    def build_initial_equation_metadata(
        self,
        *,
        wrapper: Any,
        output_json: Path,
        fit_config: dict[str, Any],
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        """Return fallback metadata used before fit() succeeds.

        By default we reuse the method fit config and drop orchestration-only fields.
        """
        exclude_keys = {
            "config_path",
            "run_type",
            "trajectory_name",
            *self.initial_equation_metadata_exclude_keys,
        }
        return {
            key: value for key, value in fit_config.items() if key not in exclude_keys
        }

    def build_fit_config(
        self,
        *,
        wrapper: Any,
        output_json: Path,
        config_path: str | Path | None = None,
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        from DataAdapters import reject_runtime_integrity_request

        reject_runtime_integrity_request(run_kwargs.get("verify_data_integrity", False))
        _, fit_noise_config = self.build_fit_trajectory(**run_kwargs)
        return {
            "config_path": None if config_path is None else Path(config_path),
            "run_type": "single",
            "trajectory_name": self.trajectory_name,
            "eval_trajectory_names": [
                str(name) for name in (run_kwargs.get("eval_trajectory_names") or [])
            ],
            "dataset_version": run_kwargs.get("dataset_version"),
            "verify_data_integrity": bool(
                run_kwargs.get("verify_data_integrity", False)
            ),
            "benchmark_partition": run_kwargs.get("benchmark_partition"),
            "benchmark_spec": run_kwargs.get("benchmark_spec"),
            "benchmark_integrity_preflight": run_kwargs.get(
                "benchmark_integrity_preflight"
            ),
            "experiment_id": run_kwargs.get("experiment_id", "adhoc"),
            "run_seed": run_kwargs.get(
                "run_seed", run_kwargs.get("random_seed", run_kwargs.get("seed", 0))
            ),
            "reaction_metrics": run_kwargs.get("reaction_metrics"),
            "fit_input_noise": fit_noise_config,
            **self.build_method_fit_config(
                wrapper=wrapper,
                output_json=output_json,
                **run_kwargs,
            ),
        }

    def build_initial_equation_result(
        self,
        *,
        wrapper: Any,
        output_json: Path,
        fit_config: dict[str, Any],
        **run_kwargs: Any,
    ) -> EquationResult:
        return EquationResult(
            status="failed",
            metadata=self.build_initial_equation_metadata(
                wrapper=wrapper,
                output_json=output_json,
                fit_config=fit_config,
                **run_kwargs,
            ),
        )

    def fit_wrapper(self, *, wrapper: Any, **run_kwargs: Any) -> EquationResult:
        fit_trajectory, _ = self.build_fit_trajectory(**run_kwargs)
        return wrapper.fit(self.fit_times, fit_trajectory)

    def get_fit_failure_reason(self, equation_result: EquationResult) -> str | None:
        return detect_invalid_equation_dimension(
            getattr(equation_result, "best", None),
            len(self.system.state_names),
        )

    def has_valid_equation(self, equation_result: EquationResult) -> bool:
        best_equation = getattr(equation_result, "best", None)
        return isinstance(best_equation, str) and bool(best_equation.strip())

    @staticmethod
    def record_artifact_finalization_failure(
        summary_path: Path,
        exc: BaseException,
    ) -> None:
        """Atomically invalidate a base success if method artifacts fail later."""
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["status"] = "failed"
        payload["failure_reason"] = "artifact_finalization_failed"
        payload["error"] = sanitize_paths(repr(exc))
        trajectory = payload.get("trajectory_evaluation") or {}
        trajectory.update(
            {
                "included_in_aggregation": False,
                "exclusion_reason": "artifact_finalization_failed",
            }
        )
        payload["trajectory_evaluation"] = trajectory
        if isinstance(payload.get("metrics"), dict):
            payload["metrics"].update(
                {
                    "included_in_aggregation": False,
                    "exclusion_reason": "artifact_finalization_failed",
                }
            )
        if isinstance(payload.get("reaction_evaluation"), dict):
            payload["reaction_evaluation"].update(
                {
                    "included_in_aggregation": False,
                    "exclusion_reason": "artifact_finalization_failed",
                }
            )
        failure = {
            "schema_version": 1,
            "stage": "artifact_finalization",
            "status": "failed",
            "failure_reason": "artifact_finalization_failed",
            "exception_type": type(exc).__name__,
            "message": sanitize_paths(repr(exc)),
            "traceback": sanitize_paths(traceback.format_exc()),
            "elapsed_seconds": None,
            "retry_count": 0,
        }
        failure_path = summary_path.parent / "failure.json"
        failure_path.write_text(
            json.dumps(failure, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        payload.setdefault("artifacts", {})["failure"] = failure_path.name
        summary_path.write_text(
            json.dumps(
                sanitize_paths(payload), indent=2, ensure_ascii=False, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )

    def get_predict_kwargs(
        self,
        *,
        wrapper: Any,
        equation_result: EquationResult,
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        return {}

    def run(
        self,
        *,
        output_json: str | Path,
        config_path: str | Path | None = None,
        eval_trajectory_names: list[str] | None = None,
        results_root: str | Path | None = None,
        run_mode: str = "single",
        config_stem: str | None = None,
        **run_kwargs: Any,
    ) -> Path:
        self._run_settings = {
            **run_kwargs,
            "eval_trajectory_names": list(eval_trajectory_names or []),
        }
        self._run_started_monotonic = time.monotonic()
        output_json = Path(output_json)
        results_root_path = None if results_root is None else Path(results_root)
        eval_names = (
            []
            if eval_trajectory_names is None
            else [str(name) for name in eval_trajectory_names]
        )

        wrapper = self.build_wrapper(output_json=output_json, **run_kwargs)
        fit_config = self.build_fit_config(
            wrapper=wrapper,
            output_json=output_json,
            config_path=config_path,
            eval_trajectory_names=eval_names,
            **run_kwargs,
        )
        equation_result = self.build_initial_equation_result(
            wrapper=wrapper,
            output_json=output_json,
            fit_config=fit_config,
            **run_kwargs,
        )

        pred_trajectory = None
        failure_reason = None
        error = None
        traceback_text = None
        predict_kwargs: dict[str, Any] = {}
        evaluation_records: list[dict[str, Any]] = []

        try:
            equation_result = self.fit_wrapper(wrapper=wrapper, **run_kwargs)
        except Exception as exc:
            failure_reason = "timeout" if isinstance(exc, TimeoutError) else "exception"
            error = repr(exc)
            traceback_text = traceback.format_exc()
        except BaseException as exc:
            if not is_nonfatal_timeout_exception(exc):
                raise
            failure_reason = "timeout"
            error = repr(exc)
            traceback_text = traceback.format_exc()
        else:
            failure_reason = self.get_fit_failure_reason(equation_result)
            if failure_reason is None and self.has_valid_equation(equation_result):
                predict_kwargs = self.get_predict_kwargs(
                    wrapper=wrapper,
                    equation_result=equation_result,
                    **run_kwargs,
                )
                try:
                    self.fit_trajectory_record.assert_rollout_initial_condition(
                        self.fit_y0
                    )
                    pred_trajectory = wrapper.predict(
                        self.fit_times, self.fit_y0, **predict_kwargs
                    )
                    if pred_trajectory is None:
                        failure_reason = "rollout_failed"
                        error = "wrapper.predict returned None"
                except Exception as exc:
                    failure_reason = (
                        "timeout" if isinstance(exc, TimeoutError) else "rollout_failed"
                    )
                    error = repr(exc)
                    traceback_text = traceback.format_exc()
                except BaseException as exc:
                    if not is_nonfatal_timeout_exception(exc):
                        raise
                    failure_reason = "timeout"
                    error = repr(exc)
                    traceback_text = traceback.format_exc()

                if eval_names and results_root_path is not None:
                    for eval_traj_name in eval_names:
                        try:
                            evaluation_records.append(
                                self.eval_single_trajectory(
                                    wrapper,
                                    eval_traj_name,
                                    method_name=wrapper.method_name,
                                    results_root=results_root_path,
                                    run_mode=run_mode,
                                    config_stem=config_stem,
                                    fit_config=fit_config,
                                    equation_result=equation_result,
                                    predict_kwargs=predict_kwargs,
                                    fit_output_dir=output_json.parent,
                                )
                            )
                        except BaseException as exc:
                            if not is_nonfatal_timeout_exception(exc):
                                raise

        summary_path = self.save_summary(
            method_name=wrapper.method_name,
            fit_config=fit_config,
            equation_result=equation_result,
            pred_trajectory=pred_trajectory,
            output_json=output_json,
            failure_reason=failure_reason,
            error=error,
            traceback_text=traceback_text,
        )
        if evaluation_records:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["evaluations"] = self.to_jsonable(evaluation_records)
            summary_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return summary_path
