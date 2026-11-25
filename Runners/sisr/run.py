"""Run SISR through the shared SysBioSR training and evaluation protocol."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
from Baseline.SISR import SISRConfig
from Runners.core import BaseMethodSingleSystemRunner, MethodSpec, finalize_run_config
from Utils import (
    ConfigError,
    load_yaml_config_with_sources,
    resolve_config_key_path,
    validate_config_keys,
)
from Runners.core.evaluation_contracts import to_jsonable
from Utils.paths import sanitize_paths

if TYPE_CHECKING:
    from Wrappers.sisr_wrapper import SISRWrapper

from .artifacts import validate_saved_model

METHOD_NAME = "sisr"


class SISRSingleSystemRunner(BaseMethodSingleSystemRunner):
    def build_wrapper(self, *, output_json: Path, **kwargs: Any) -> SISRWrapper:
        from Wrappers.sisr_wrapper import SISRWrapper

        self.wrapper = SISRWrapper(
            state_names=self.system.state_names,
            **{k: v for k, v in kwargs.items() if k in SISRConfig.__dataclass_fields__},
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
            result = self.wrapper.fit_result
            if result is not None:
                for key, name, value in (
                    ("sisr_pareto_front", "pareto_front.json", result.front),
                    ("sisr_selected_model", "selected_model.json", result.model),
                ):
                    if value is None:
                        continue
                    artifact = path.parent / name
                    artifact.write_text(
                        json.dumps(
                            to_jsonable(sanitize_paths(value)),
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    summary["artifacts"][key] = name
            summary["complete"] = True
            path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
        except Exception as exc:
            self.record_artifact_finalization_failure(path, exc)
            raise
        return path


ALGORITHM_KEYS = set(SISRConfig.__dataclass_fields__)
VALID_CONFIG_KEYS = ALGORITHM_KEYS | {
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
    validate_config_keys(config, valid_keys=VALID_CONFIG_KEYS, config_path=path)
    if config.get("method") != METHOD_NAME:
        raise ConfigError("SISR configs must use method: sisr")
    if config.get("run_type") not in ("single", "batch"):
        raise ConfigError("run_type must be single or batch")
    evaluations = config.get("eval_trajectory_names") or []
    if not isinstance(evaluations, list):
        raise ConfigError("eval_trajectory_names must be a list")
    try:
        algorithm = SISRConfig(
            **{k: v for k, v in config.items() if k in ALGORITHM_KEYS}
        )
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"Invalid SISR configuration: {exc}") from exc
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
    for key in ("results_root", "benchmark_definition"):
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
        display_name="SISR-NNLS-Pareto (R2-R12)",
        single_runner_cls=SISRSingleSystemRunner,
        config_loader=load_run_config,
        validate_saved_model=validate_saved_model,
        required_fit_artifacts=frozenset(
            {
                "sisr_pareto_front",
                "sisr_selected_model",
            }
        ),
    )
