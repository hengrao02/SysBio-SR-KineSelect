"""Run PySR through the shared SysBioSR training and evaluation protocol."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
from Baseline.PySR import PySRConfig
from .artifacts import validate_saved_model
from Runners.core import BaseMethodSingleSystemRunner, MethodSpec, finalize_run_config
from Utils import (
    ConfigError,
    load_yaml_config_with_sources,
    resolve_config_key_path,
    validate_config_keys,
)
from Runners.core.evaluation_contracts import to_jsonable

if TYPE_CHECKING:
    from Wrappers.pysr_wrapper import PySRWrapper

METHOD_NAME = "pysr"


class PySRSingleSystemRunner(BaseMethodSingleSystemRunner):
    persist_equations_in_evaluation = True

    def build_wrapper(self, *, output_json: Path, **kwargs: Any) -> PySRWrapper:
        from Wrappers.pysr_wrapper import PySRWrapper

        self.wrapper = PySRWrapper(
            state_names=self.system.state_names,
            output_dir=output_json.parent,
            julia_executable=kwargs.get("julia_executable"),
            julia_project=kwargs.get("julia_project"),
            **{k: v for k, v in kwargs.items() if k in PySRConfig.__dataclass_fields__},
        )
        return self.wrapper

    def build_method_fit_config(self, *, wrapper, output_json, **kwargs):
        return {
            **wrapper.config.to_dict(),
        }

    def get_fit_failure_reason(self, result):
        return result.metadata.get("failure_reason") or super().get_fit_failure_reason(
            result
        )

    def run(self, **kwargs: Any) -> Path:
        path = super().run(**kwargs)
        try:
            summary = json.loads(path.read_text())
            estimator = self.wrapper.estimator
            for key, name, value in (
                ("pysr_selected_model", "selected_model.json", estimator.model),
                ("pysr_hall_of_fame", "hall_of_fame.json", estimator.hall_of_fame),
                (
                    "pysr_search_summary",
                    "search_summary.json",
                    estimator.search_summary,
                ),
            ):
                if value is None:
                    continue
                artifact = path.parent / name
                artifact.write_text(
                    json.dumps(to_jsonable(value), indent=2, allow_nan=False) + "\n"
                )
                summary["artifacts"][key] = name
            if estimator.worker_log is not None and estimator.worker_log.is_file():
                summary["artifacts"]["pysr_worker_log"] = estimator.worker_log.name
            summary["complete"] = True
            path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
        except Exception as exc:
            self.record_artifact_finalization_failure(path, exc)
            raise
        return path


ALGORITHM_KEYS = set(PySRConfig.__dataclass_fields__)
VALID_CONFIG_KEYS = ALGORITHM_KEYS | {
    "julia_executable",
    "julia_project",
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
    "benchmark_definition",
    "benchmark_partition",
    "verify_data_integrity",
    "dataset_version",
}


def load_run_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path, config, sources = load_yaml_config_with_sources(config_path)
    if "max_candidate_rollouts" in config:
        raise ConfigError(
            "Remove max_candidate_rollouts: final PySR selection now evaluates all "
            "candidate combinations. Use a fresh results directory for this protocol."
        )
    validate_config_keys(config, valid_keys=VALID_CONFIG_KEYS, config_path=path)
    if config.get("method") != METHOD_NAME:
        raise ConfigError("PySR configs must use method: pysr")
    if config.get("run_type") not in ("single", "batch"):
        raise ConfigError("run_type must be single or batch")
    evaluations = config.get("eval_trajectory_names") or []
    if not isinstance(evaluations, list):
        raise ConfigError("eval_trajectory_names must be a list")
    try:
        algorithm = PySRConfig(
            **{k: v for k, v in config.items() if k in ALGORITHM_KEYS}
        )
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"Invalid PySR configuration: {exc}") from exc
    resolved = {
        "method": METHOD_NAME,
        "run_type": config["run_type"],
        "trajectory_name": str(config.get("trajectory_name", "default")),
        "eval_trajectory_names": [str(name) for name in evaluations],
        "benchmark_partition": config.get("benchmark_partition"),
        "dataset_version": config.get("dataset_version"),
        "verify_data_integrity": bool(config.get("verify_data_integrity", False)),
        **algorithm.to_dict(),
    }
    for key in (
        "results_root",
        "benchmark_definition",
        "julia_executable",
        "julia_project",
    ):
        resolved[key] = resolve_config_key_path(
            config, sources, key, fallback_config_path=path
        )
    return finalize_run_config(
        method_name=METHOD_NAME,
        config_path=path,
        config=config,
        sources=sources,
        resolved=resolved,
    )


def get_method_spec() -> MethodSpec:
    return MethodSpec(
        method_name=METHOD_NAME,
        display_name="PySR (evolutionary symbolic regression)",
        supports_reaction_metrics=False,
        single_runner_cls=PySRSingleSystemRunner,
        config_loader=load_run_config,
        required_fit_artifacts=frozenset(
            {
                "pysr_hall_of_fame",
                "pysr_search_summary",
                "pysr_selected_model",
            }
        ),
        validate_saved_model=validate_saved_model,
    )
