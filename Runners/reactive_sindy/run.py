from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from Baseline.ReactiveSINDy import ReactiveSINDyConfig
from Runners.core import (
    BaseMethodSingleSystemRunner,
    MethodSpec,
    finalize_run_config,
    run_batch_from_spec,
    run_from_spec,
    run_single_from_spec,
)
from Utils import (
    ConfigError,
    load_yaml_config_with_sources,
    resolve_config_key_path,
    validate_config_keys,
)
from Utils.paths import sanitize_paths
from Wrappers.reactive_sindy_wrapper import ReactiveSINDyWrapper

from .artifacts import validate_saved_model

METHOD_NAME = "reactive_sindy"
DISPLAY_NAME = "Reactive SINDy"


def _normalize_float_list(
    value: Any, *, key: str, default: tuple[float, ...]
) -> tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    if isinstance(value, str):
        text = value.strip()
        return tuple(float(part.strip()) for part in text.split(",") if part.strip())
    raise ConfigError(f"Invalid '{key}' config value: {value!r}")


def _algorithm_config(config: dict[str, Any]) -> ReactiveSINDyConfig:
    try:
        return ReactiveSINDyConfig(
            library_mode=str(config.get("library_mode", "enumerated")),
            max_reaction_order=int(config.get("max_reaction_order", 2)),
            include_fission=bool(config.get("include_fission", True)),
            include_double_conversion=bool(
                config.get("include_double_conversion", True)
            ),
            deduplicate_reactions=bool(config.get("deduplicate_reactions", True)),
            derivative_method=str(config.get("derivative_method", "numpy_gradient")),
            derivative_edge_order=int(config.get("derivative_edge_order", 2)),
            short_trajectory_edge_order=int(
                config.get("short_trajectory_edge_order", 1)
            ),
            drop_boundary_points=bool(config.get("drop_boundary_points", True)),
            normalize_library=bool(config.get("normalize_library", True)),
            library_normalization=str(config.get("library_normalization", "column_l2")),
            library_zero_norm_threshold=float(
                config.get("library_zero_norm_threshold", 1.0e-14)
            ),
            library_scale_fit=str(
                config.get("library_scale_fit", "full_effective_trajectory")
            ),
            library_scale_refit=str(
                config.get("library_scale_refit", "reuse_full_effective_scale")
            ),
            derivative_block_policy=str(
                config.get(
                    "derivative_block_policy",
                    "differentiate_full_then_split_effective",
                )
            ),
            optimize_hyperparams=bool(config.get("optimize_hyperparams", True)),
            hyper_opt_eval_fraction=(
                None
                if config.get("hyper_opt_eval_fraction") is None
                else float(config["hyper_opt_eval_fraction"])
            ),
            hyper_opt_split=str(config.get("hyper_opt_split", "chronological")),
            hyper_opt_train_steps_rounding=str(
                config.get(
                    "hyper_opt_train_steps_rounding", "python_round_half_to_even"
                )
            ),
            alpha_scale_grid=_normalize_float_list(
                config.get("alpha_scale_grid"),
                key="alpha_scale_grid",
                default=ReactiveSINDyConfig.alpha_scale_grid,
            ),
            l1_ratio_grid=_normalize_float_list(
                config.get("l1_ratio_grid"),
                key="l1_ratio_grid",
                default=ReactiveSINDyConfig.l1_ratio_grid,
            ),
            alpha=None if config.get("alpha") is None else float(config["alpha"]),
            l1_ratio=float(config.get("l1_ratio", 1.0)),
            alpha_max_normalizer=str(
                config.get("alpha_max_normalizer", "num_scalar_rows")
            ),
            objective_normalizer=str(
                config.get("objective_normalizer", "num_time_steps")
            ),
            final_refit_derivative_blocks=str(
                config.get(
                    "final_refit_derivative_blocks",
                    "full_effective_trajectory",
                )
            ),
            rate_threshold=float(config.get("rate_threshold", 1.0e-8)),
            relative_rate_threshold=float(
                config.get("relative_rate_threshold", 1.0e-4)
            ),
            sparsity_selection_tolerance=float(
                config.get("sparsity_selection_tolerance", 0.05)
            ),
            optimizer_method=str(config.get("optimizer_method", "L-BFGS-B")),
            optimizer_max_iter=int(config.get("optimizer_max_iter", 5000)),
            optimizer_tol=float(config.get("optimizer_tol", 1.0e-10)),
            ode_integrator=str(config.get("ode_integrator", "solve_ivp")),
            ode_method=str(config.get("ode_method", "RK45")),
            ode_rtol=float(config.get("ode_rtol", 1.0e-3)),
            ode_atol=float(config.get("ode_atol", 1.0e-6)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid Reactive SINDy algorithm config: {exc}") from exc


class ReactiveSINDySingleSystemRunner(BaseMethodSingleSystemRunner):
    initial_equation_metadata_exclude_keys = (
        "ode_integrator",
        "ode_method",
        "ode_rtol",
        "ode_atol",
    )

    def build_wrapper(
        self, *, output_json: Path, **run_kwargs: Any
    ) -> ReactiveSINDyWrapper:
        del output_json
        algorithm = ReactiveSINDyConfig(
            **{
                key: value
                for key, value in run_kwargs.items()
                if key in ReactiveSINDyConfig.__dataclass_fields__
            }
        )
        self._reactive_wrapper = ReactiveSINDyWrapper(
            state_names=self.system.state_names,
            **algorithm.to_dict(),
        )
        return self._reactive_wrapper

    def build_method_fit_config(
        self,
        *,
        wrapper: ReactiveSINDyWrapper,
        output_json: Path,
        **_: Any,
    ) -> dict[str, Any]:
        del output_json
        return {
            **wrapper.config.to_dict(),
        }

    def get_predict_kwargs(
        self,
        *,
        wrapper: ReactiveSINDyWrapper,
        equation_result: Any,
        ode_integrator: str = "solve_ivp",
        **_: Any,
    ) -> dict[str, Any]:
        del wrapper, equation_result
        return {"ode_integrator": ode_integrator}

    def eval_single_trajectory(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        record = super().eval_single_trajectory(*args, **kwargs)
        if not hasattr(self, "_reactive_eval_records"):
            self._reactive_eval_records: list[dict[str, Any]] = []
        self._reactive_eval_records.append(record)
        return record

    @staticmethod
    def _json_safe(payload: Any) -> Any:
        if isinstance(payload, float):
            return payload if math.isfinite(payload) else None
        if isinstance(payload, dict):
            return {
                str(key): ReactiveSINDySingleSystemRunner._json_safe(value)
                for key, value in payload.items()
            }
        if isinstance(payload, (list, tuple)):
            return [
                ReactiveSINDySingleSystemRunner._json_safe(value) for value in payload
            ]
        return payload

    @classmethod
    def _write_json(cls, path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(
                cls._json_safe(sanitize_paths(payload)),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_release_artifacts(
        self,
        summary_path: Path,
        *,
        eval_trajectory_names: list[str],
        dataset_version: str | None,
        benchmark_partition: str | None,
    ) -> None:
        output_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        wrapper = getattr(self, "_reactive_wrapper", None)
        result = None if wrapper is None else wrapper.fit_result
        contract = None if wrapper is None else wrapper.config.to_dict()

        if contract is not None:
            (output_dir / "resolved_algorithm_config.yaml").write_text(
                yaml.safe_dump(contract, sort_keys=True, allow_unicode=True),
                encoding="utf-8",
            )
        run_summary: dict[str, Any] = {
            "schema_version": 1,
            "run_id": summary.get("run_id"),
            "status": summary.get("status"),
            "failure_reason": summary.get("failure_reason"),
            "method": METHOD_NAME,
            "system_id": self.system.system_id,
            "dataset_version": dataset_version,
            "benchmark_partition": benchmark_partition,
            "fit_trajectory": self.trajectory_name,
            "evaluation_trajectories": list(eval_trajectory_names),
            "evaluation_protocol": (
                "default_to_perturbed"
                if "perturbed" in eval_trajectory_names
                else "fit_reconstruction"
            ),
            "benchmark_spec": (summary.get("fit_config") or {}).get("benchmark_spec"),
            "trajectory_evaluation": summary.get("trajectory_evaluation"),
            "reaction_evaluation": summary.get("reaction_evaluation"),
            "fit_reconstruction_evaluation": summary.get("trajectory_evaluation"),
        }

        if result is not None:
            candidate_rows = result.candidate_library_payload()
            try:
                import zstandard
            except ImportError as exc:  # pragma: no cover - install contract failure
                raise RuntimeError(
                    "zstandard is required for candidate_library.jsonl.zst"
                ) from exc
            compressor = zstandard.ZstdCompressor(level=9, write_checksum=True)
            with (output_dir / "candidate_library.jsonl.zst").open("wb") as raw_file:
                with compressor.stream_writer(raw_file) as compressed:
                    for row in candidate_rows:
                        encoded = json.dumps(
                            row,
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        compressed.write(encoded + b"\n")

            selected_reactions = result.selected_reactions_payload()
            self._write_json(
                output_dir / "selected_model.json", wrapper.estimator.to_model()
            )
            self._write_json(output_dir / "selected_reactions.json", selected_reactions)
            self._write_json(
                output_dir / "hyperparameter_trace.json",
                {
                    "records": list(result.hyperparameter_records),
                    "selected_alpha": result.selected_alpha,
                    "selected_l1_ratio": result.selected_l1_ratio,
                    "selected_candidate_index": result.metadata[
                        "selected_candidate_index"
                    ],
                },
            )
            run_summary.update(
                {
                    "num_candidate_reactions": len(result.reactions),
                    "num_active_reactions": len(selected_reactions),
                    "derivative_policy": {
                        "method": wrapper.config.derivative_method,
                        "edge_order": wrapper.config.derivative_edge_order,
                        "short_trajectory_edge_order": wrapper.config.short_trajectory_edge_order,
                        "drop_boundary_points": wrapper.config.drop_boundary_points,
                        "block_policy": wrapper.config.derivative_block_policy,
                        "final_refit_blocks": wrapper.config.final_refit_derivative_blocks,
                    },
                    "alpha_max_normalizer": wrapper.config.alpha_max_normalizer,
                    "objective_normalizer": wrapper.config.objective_normalizer,
                    "final_refit_effective_steps": result.metadata[
                        "final_refit_effective_steps"
                    ],
                    "train_time_range": result.metadata["train_time_range"],
                    "validation_time_range": result.metadata["validation_time_range"],
                    "selected_alpha": result.selected_alpha,
                    "selected_l1_ratio": result.selected_l1_ratio,
                    "solver_status": {
                        "success": result.metadata["final_solver_success"],
                        "message": result.metadata["final_solver_message"],
                        "iterations": result.metadata["final_solver_nit"],
                    },
                    "effective_rate_threshold": result.metadata[
                        "effective_rate_threshold"
                    ],
                }
            )
        metrics = summary.get("metrics")
        if metrics is not None:
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            self._write_json(
                metrics_dir / "trajectory_metrics.json",
                {
                    "evaluation": summary.get("trajectory_evaluation"),
                    "scores": metrics,
                },
            )
        reaction_evaluation = summary.get("reaction_evaluation")
        if reaction_evaluation is not None:
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            self._write_json(metrics_dir / "reaction_metrics.json", reaction_evaluation)
        evaluation_records = getattr(self, "_reactive_eval_records", [])
        if evaluation_records:
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            for record in evaluation_records:
                trajectory_name = str(record.get("eval_trajectory_name", "unknown"))
                if record.get("metrics") is not None:
                    self._write_json(
                        metrics_dir / f"{trajectory_name}_trajectory_metrics.json",
                        {
                            "evaluation": record.get("trajectory_evaluation"),
                            "scores": record["metrics"],
                        },
                    )
            run_summary["evaluation_statuses"] = [
                {
                    "trajectory_name": record.get("eval_trajectory_name"),
                    "status": record.get("status"),
                }
                for record in evaluation_records
            ]
            official = next(
                (
                    record
                    for record in evaluation_records
                    if record.get("eval_trajectory_name") == "perturbed"
                ),
                evaluation_records[0],
            )
            official_name = str(official.get("eval_trajectory_name", "unknown"))
            official_metrics_path = (
                output_dir / "metrics" / f"{official_name}_trajectory_metrics.json"
            )
            official_evaluation = official.get("trajectory_evaluation") or {}
            run_summary["trajectory_evaluation"] = official_evaluation
            run_summary["reaction_evaluation"] = official.get("reaction_evaluation")
            run_summary["official_evaluation"] = {
                "trajectory_name": official_name,
                "status": official.get("status"),
                "output_json": official.get("output_json"),
                "metrics_artifact": (
                    official_metrics_path.relative_to(output_dir).as_posix()
                    if official_metrics_path.is_file()
                    else None
                ),
                "metric_version": official_evaluation.get("metric_version"),
                "included_in_aggregation": official_evaluation.get(
                    "included_in_aggregation"
                ),
                "exclusion_reason": official_evaluation.get("exclusion_reason"),
                "scores": official.get("metrics"),
            }

        self._write_json(output_dir / "run_summary.json", run_summary)

        method_artifacts: dict[str, str] = {}
        for relative in (
            "run_summary.json",
            "selected_model.json",
            "selected_reactions.json",
            "candidate_library.jsonl.zst",
            "hyperparameter_trace.json",
            "resolved_algorithm_config.yaml",
            "metrics/trajectory_metrics.json",
            "metrics/reaction_metrics.json",
            *(
                f"metrics/{name}_trajectory_metrics.json"
                for name in eval_trajectory_names
            ),
        ):
            path = output_dir / relative
            if path.is_file():
                key = "reactive_" + relative.replace("/", "_").replace(".", "_")
                method_artifacts[key] = relative
        summary.setdefault("artifacts", {}).update(method_artifacts)
        summary["complete"] = True
        self._write_json(summary_path, summary)

    def run(self, **run_kwargs: Any) -> Path:
        eval_names = [
            str(name) for name in (run_kwargs.get("eval_trajectory_names") or [])
        ]
        dataset_version = (
            None
            if run_kwargs.get("dataset_version") is None
            else str(run_kwargs["dataset_version"])
        )
        benchmark_partition = (
            None
            if run_kwargs.get("benchmark_partition") is None
            else str(run_kwargs["benchmark_partition"])
        )
        summary_path = Path(super().run(**run_kwargs))
        try:
            self._write_release_artifacts(
                summary_path,
                eval_trajectory_names=eval_names,
                dataset_version=dataset_version,
                benchmark_partition=benchmark_partition,
            )
        except BaseException as exc:
            self.record_artifact_finalization_failure(summary_path, exc)
            raise
        return summary_path


def get_method_spec() -> MethodSpec:
    return MethodSpec(
        method_name=METHOD_NAME,
        display_name=DISPLAY_NAME,
        single_runner_cls=ReactiveSINDySingleSystemRunner,
        config_loader=load_run_config,
        validate_saved_model=validate_saved_model,
        required_fit_artifacts=frozenset(
            {
                "reactive_candidate_library_jsonl_zst",
                "reactive_hyperparameter_trace_json",
                "reactive_resolved_algorithm_config_yaml",
                "reactive_run_summary_json",
                "reactive_selected_model_json",
                "reactive_selected_reactions_json",
            }
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the released Reactive SINDy baseline from one YAML config.",
    )
    parser.add_argument("config", type=Path, help="Path to a YAML config file.")
    return parser.parse_args()


ALGORITHM_CONFIG_KEYS = set(ReactiveSINDyConfig.__dataclass_fields__)
VALID_CONFIG_KEYS = {
    "method",
    "run_type",
    "results_root",
    "experiment_id",
    "systems_root",
    "system_dir",
    "trajectory_name",
    "eval_trajectory_names",
    "output_json",
    "output_root",
    "limit",
    "skip_existing",
    "resume",
    "continue_on_error",
    "recursive",
    "batch_num_workers",
    "verbose",
    "benchmark_definition",
    "benchmark_partition",
    "verify_data_integrity",
    "dataset_version",
    *ALGORITHM_CONFIG_KEYS,
}


def load_run_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path, config, sources = load_yaml_config_with_sources(config_path)
    validate_config_keys(config, valid_keys=VALID_CONFIG_KEYS, config_path=config_path)
    run_type = config.get("run_type")
    if run_type not in {"single", "batch"}:
        raise ConfigError(
            f"'run_type' must be either 'single' or 'batch': {config_path}"
        )
    raw_eval = config.get("eval_trajectory_names") or []
    if not isinstance(raw_eval, list):
        raise ConfigError("eval_trajectory_names must be a list")

    algorithm = _algorithm_config(config)
    resolved: dict[str, Any] = {
        "method": METHOD_NAME,
        "run_type": run_type,
        "trajectory_name": str(config.get("trajectory_name", "default")),
        "eval_trajectory_names": [str(name) for name in raw_eval],
        "results_root": resolve_config_key_path(
            config, sources, "results_root", fallback_config_path=config_path
        ),
        "benchmark_definition": resolve_config_key_path(
            config,
            sources,
            "benchmark_definition",
            fallback_config_path=config_path,
        ),
        "benchmark_partition": (
            None
            if config.get("benchmark_partition") is None
            else str(config["benchmark_partition"])
        ),
        "verify_data_integrity": bool(config.get("verify_data_integrity", False)),
        "dataset_version": (
            None
            if config.get("dataset_version") is None
            else str(config["dataset_version"])
        ),
        **algorithm.to_dict(),
    }
    return finalize_run_config(
        method_name=METHOD_NAME,
        config_path=config_path,
        config=config,
        sources=sources,
        resolved=resolved,
    )


def run_single_from_config(config_path: Path, config: dict[str, Any]) -> Path:
    return run_single_from_spec(get_method_spec(), config_path, config)


def run_batch_from_config(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return run_batch_from_spec(get_method_spec(), config_path, config)


def main() -> None:
    args = parse_args()
    config_path, config = load_run_config(args.config)
    run_from_spec(get_method_spec(), config_path, config)


if __name__ == "__main__":
    main()
