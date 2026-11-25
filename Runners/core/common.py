from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from DataAdapters import validate_system_preflight
from Utils import ConfigError, require_config_value, resolve_config_key_path

from .base_batch_runner import BaseBatchRunner
from .artifact_validation import validate_resumable_result
from .inputs import scientific_config

_RESERVED_RUN_CONFIG_KEYS = {
    "method",
    "run_type",
    "trajectory_name",
    "system_dir",
    "output_json",
    "systems_root",
    "output_root",
    "results_root",
    "config_stem",
    "limit",
    "skip_existing",
    "resume",
    "continue_on_error",
    "recursive",
    "batch_num_workers",
    "benchmark_definition",
}

FIT_NOISE_CONFIG_KEYS = {
    "fit_noise_type",
    "fit_noise_scale",
    "fit_noise_mode",
    "fit_noise_seed",
    "fit_noise_clip_nonnegative",
    "fit_noise_preserve_initial",
}


def _path_token(value: Any, *, key: str) -> str:
    token = str(value)
    if not token or token in {".", ".."} or "/" in token or "\\" in token:
        raise ConfigError(
            f"'{key}' must be one non-empty path-safe token, got {value!r}"
        )
    return token


@dataclass(frozen=True)
class MethodSpec:
    method_name: str
    single_runner_cls: type | None
    config_loader: Callable[[str | Path], tuple[Path, dict[str, Any]]]
    display_name: str | None = None
    experiment_runner: (
        Callable[[Path, dict[str, Any]], Path | dict[str, Any]] | None
    ) = None
    supports_reaction_metrics: bool = True
    supports_resume: bool = True
    method_version: int = 1
    validate_saved_model: Callable[[dict[str, Any], Path], str | None] | None = None
    required_fit_artifacts: frozenset[str] = frozenset()

    @property
    def label(self) -> str:
        return self.display_name or self.method_name


def build_result_path(
    results_root: str | Path,
    *,
    trajectory_name: str,
    method_name: str,
    run_mode: str,
    system_id: str,
    config_stem: str | None = None,
    experiment_id: str = "adhoc",
    dataset_version: str = "unversioned",
    benchmark_partition: str = "unpartitioned",
    seed: int = 0,
) -> Path:
    """Single source of truth for all SysBioSR result paths.

    <results_root>/<experiment>/<method>/<dataset>/<partition>/
      <system>/<trajectory>/seed-<seed>/summary.json
    """
    if run_mode not in {"single", "batch"}:
        raise ValueError(f"unsupported run_mode={run_mode!r}")
    if run_mode == "batch" and config_stem is None:
        raise ValueError("config_stem is required for batch run_mode")
    return (
        Path(results_root)
        / _path_token(experiment_id, key="experiment_id")
        / _path_token(method_name, key="method_name")
        / _path_token(dataset_version, key="dataset_version")
        / _path_token(benchmark_partition, key="benchmark_partition")
        / _path_token(system_id, key="system_id")
        / _path_token(trajectory_name, key="trajectory_name")
        / f"seed-{int(seed)}"
        / "summary.json"
    )


def build_batch_output_root(
    results_root: str | Path,
    *,
    trajectory_name: str,
    method_name: str,
    config_stem: str,
    experiment_id: str = "adhoc",
    dataset_version: str = "unversioned",
    benchmark_partition: str = "unpartitioned",
) -> Path:
    del trajectory_name, config_stem
    return (
        Path(results_root)
        / _path_token(experiment_id, key="experiment_id")
        / _path_token(method_name, key="method_name")
        / _path_token(dataset_version, key="dataset_version")
        / _path_token(benchmark_partition, key="benchmark_partition")
    )


def build_runner_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in _RESERVED_RUN_CONFIG_KEYS
    }


def finalize_run_config(
    *,
    method_name: str,
    config_path: Path,
    config: dict[str, Any],
    sources: Any,
    resolved: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if (
        resolved.get("dataset_version")
        and resolved.get("benchmark_partition")
        and not resolved.get("benchmark_definition")
    ):
        raise ConfigError(
            "named benchmark runs require an explicit benchmark_definition"
        )
    if resolved.get("verify_data_integrity"):
        from DataAdapters import PUBLISHED_INTEGRITY_MIGRATION

        raise ConfigError(PUBLISHED_INTEGRITY_MIGRATION)
    from .method_capabilities import get_optional_method_spec

    spec = get_optional_method_spec(method_name)
    resolved["method_version"] = spec.method_version if spec is not None else 1
    run_type = resolved["run_type"]
    trajectory_name = resolved["trajectory_name"]
    results_root = resolved.get("results_root")
    if results_root is None:
        results_root = resolve_config_key_path(
            config, sources, "results_root", fallback_config_path=config_path
        )

    for key in FIT_NOISE_CONFIG_KEYS:
        if key in config:
            resolved[key] = config[key]

    experiment_id = str(
        config.get("experiment_id", resolved.get("experiment_id", "adhoc"))
    )
    dataset_version = str(
        resolved.get("dataset_version")
        or config.get("dataset_version")
        or "unversioned"
    )
    benchmark_partition = str(
        resolved.get("benchmark_partition")
        or config.get("benchmark_partition")
        or "unpartitioned"
    )
    run_seed = int(
        resolved.get("random_seed", config.get("random_seed", config.get("seed", 0)))
    )
    resolved.update(
        {
            "experiment_id": experiment_id,
            "run_seed": run_seed,
        }
    )

    if run_type == "single":
        require_config_value(
            config.get("system_dir"), key="system_dir", config_path=config_path
        )
        system_dir = resolve_config_key_path(
            config, sources, "system_dir", fallback_config_path=config_path
        )
        output_json = resolve_config_key_path(
            config, sources, "output_json", fallback_config_path=config_path
        )
        if output_json is None:
            require_config_value(
                results_root, key="results_root", config_path=config_path
            )
            output_json = build_result_path(
                results_root,
                trajectory_name=trajectory_name,
                method_name=method_name,
                run_mode="single",
                system_id=Path(system_dir).name,
                experiment_id=experiment_id,
                dataset_version=dataset_version,
                benchmark_partition=benchmark_partition,
                seed=run_seed,
            )
        resolved.update(
            {
                "system_dir": system_dir,
                "output_json": output_json,
                "results_root": results_root,
                "resume": bool(config.get("resume", False)),
            }
        )
        return config_path, resolved

    require_config_value(
        config.get("systems_root"), key="systems_root", config_path=config_path
    )
    systems_root = resolve_config_key_path(
        config, sources, "systems_root", fallback_config_path=config_path
    )
    output_root = resolve_config_key_path(
        config, sources, "output_root", fallback_config_path=config_path
    )

    if output_root is None:
        require_config_value(results_root, key="results_root", config_path=config_path)
        output_root = build_batch_output_root(
            results_root,
            trajectory_name=trajectory_name,
            method_name=method_name,
            config_stem=config_path.stem,
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            benchmark_partition=benchmark_partition,
        )
    elif resolved.get("eval_trajectory_names"):
        raise ConfigError(
            f"'output_root' is not supported together with non-empty 'eval_trajectory_names' in batch mode: {config_path}\n"
            "Use 'results_root' instead so extrapolation outputs can be placed in per-trajectory directories."
        )

    resolved.update(
        {
            "systems_root": systems_root,
            "results_root": results_root,
            "output_root": output_root,
            "config_stem": config_path.stem,
            "limit": None if config.get("limit") is None else int(config["limit"]),
            "skip_existing": bool(config.get("skip_existing", True)),
            "continue_on_error": bool(config.get("continue_on_error", True)),
            "recursive": bool(config.get("recursive", True)),
            "batch_num_workers": max(1, int(config.get("batch_num_workers", 1))),
            "benchmark_definition": resolve_config_key_path(
                config,
                sources,
                "benchmark_definition",
                fallback_config_path=config_path,
            ),
            "benchmark_partition": config.get("benchmark_partition"),
            "verify_data_integrity": bool(config.get("verify_data_integrity", False)),
            "dataset_version": config.get("dataset_version"),
        }
    )
    if bool(config.get("resume", False)):
        resolved["skip_existing"] = True
    return config_path, resolved


def _print_single_run_summary(
    spec: MethodSpec,
    runner: Any,
    *,
    config_path: Path,
    config: dict[str, Any],
    saved_json: Path,
) -> None:
    print("=" * 80)
    print(f"{spec.label} single-system run finished")
    print(f"config         : {config_path}")
    print(f"system_id      : {runner.system.system_id}")
    print(f"trajectory     : {config['trajectory_name']}")
    print(f"saved json     : {saved_json}")
    try:
        saved_payload = json.loads(saved_json.read_text(encoding="utf-8"))
    except Exception:
        saved_payload = {}
    print(f"status         : {saved_payload.get('status')}")
    print(f"failure_reason : {saved_payload.get('failure_reason')}")
    print(f"overlay plot   : {saved_payload.get('artifacts', {}).get('overlay_plot')}")
    print(
        f"generated csv  : {saved_payload.get('artifacts', {}).get('generated_trajectory_csv')}"
    )
    print("=" * 80)


def run_single_from_spec(
    spec: MethodSpec, config_path: Path, config: dict[str, Any]
) -> Path:
    if spec.single_runner_cls is None:
        raise ConfigError(
            f"Method '{spec.method_name}' does not support single-system runs."
        )
    run_config = dict(config)
    manifest_path = run_config.get("benchmark_definition")
    if manifest_path is not None:
        partition = run_config.get("benchmark_partition")
        if not partition:
            raise ConfigError(
                "benchmark_partition is required with benchmark_definition"
            )
        with ProcessPoolExecutor(max_workers=1) as executor:
            row, context = executor.submit(
                validate_system_preflight,
                manifest_path,
                run_config["system_dir"],
                partition=str(partition),
            ).result()
        configured_version = run_config.get("dataset_version")
        if configured_version not in {None, "", row.dataset_version}:
            raise ConfigError(
                f"dataset_version mismatch for {row.system_id}: "
                f"config={configured_version!r}, manifest={row.dataset_version!r}"
            )
        run_config.update(
            {
                "dataset_version": row.dataset_version,
                "reaction_metrics": row.reaction_metrics
                and spec.supports_reaction_metrics,
                "benchmark_spec": context["benchmark_spec"],
                "benchmark_integrity_preflight": "isolated_process",
            }
        )
    if run_config.get("resume") and not spec.supports_resume:
        raise ConfigError(
            f"{spec.label} does not support automatic resume; use a fresh results root"
        )
    output_json = Path(run_config["output_json"])
    if output_json.exists():
        if not run_config.get("resume", False):
            raise FileExistsError(
                f"Refusing to overwrite existing run identity: {output_json}. "
                "Choose a new --results-root or set resume: true for an identical input."
            )
        saved = BaseBatchRunner._safe_load_json(output_json)
        rejection = validate_resumable_result(
            saved,
            output_json=output_json,
            expected={
                "method": spec.method_name,
                "run_config": scientific_config(
                    {
                        **build_runner_kwargs(run_config),
                        "eval_trajectory_names": run_config.get(
                            "eval_trajectory_names", []
                        ),
                    }
                ),
                "system_dir": Path(run_config["system_dir"]),
                "benchmark_spec": run_config.get("benchmark_spec"),
                "system_id": Path(run_config["system_dir"]).name,
                "trajectory_name": run_config["trajectory_name"],
                "dataset_version": run_config.get("dataset_version"),
                "benchmark_partition": run_config.get("benchmark_partition"),
                "experiment_id": run_config.get("experiment_id"),
                "random_seed": run_config.get(
                    "random_seed", run_config.get("run_seed", run_config.get("seed", 0))
                ),
            },
        )
        if rejection is not None:
            raise RuntimeError(
                f"Existing result cannot be resumed ({rejection}): {output_json}. Choose a new --results-root."
            )
        return output_json
    if output_json.parent.exists() and any(output_json.parent.iterdir()):
        raise FileExistsError(
            f"Incomplete existing result directory: {output_json.parent}; use a fresh results root"
        )
    runner = spec.single_runner_cls(
        run_config["system_dir"],
        trajectory_name=run_config["trajectory_name"],
    )
    results_root = run_config.get("results_root")
    saved_json = Path(
        runner.run(
            output_json=output_json,
            config_path=config_path,
            results_root=results_root,
            run_mode="single",
            **build_runner_kwargs(run_config),
        )
    )
    _print_single_run_summary(
        spec, runner, config_path=config_path, config=run_config, saved_json=saved_json
    )
    return saved_json


def run_batch_from_spec(
    spec: MethodSpec, config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    if spec.single_runner_cls is None:
        raise ConfigError(f"Method '{spec.method_name}' does not support batch runs.")
    if (
        config.get("skip_existing") or config.get("resume")
    ) and not spec.supports_resume:
        raise ConfigError(
            f"{spec.label} does not support automatic resume; use a fresh results root"
        )
    batch_runner = BaseBatchRunner(
        systems_root=config["systems_root"],
        output_root=config["output_root"],
        results_root=config.get("results_root"),
        config_stem=config["config_stem"],
        single_runner_cls=spec.single_runner_cls,
        method_name=spec.method_name,
        trajectory_name=config["trajectory_name"],
        recursive=config["recursive"],
        benchmark_definition=config.get("benchmark_definition"),
        benchmark_partition=config.get("benchmark_partition"),
        verify_data_integrity=config.get("verify_data_integrity", False),
        run_seed=config.get("run_seed", config.get("random_seed", 0)),
    )
    return batch_runner.run(
        limit=config["limit"],
        skip_existing=config["skip_existing"],
        continue_on_error=config["continue_on_error"],
        num_workers=config["batch_num_workers"],
        config_path=config_path,
        **build_runner_kwargs(config),
    )


def run_from_spec(
    spec: MethodSpec, config_path: Path, config: dict[str, Any]
) -> Path | dict[str, Any]:
    if config["run_type"] == "experiment":
        if spec.experiment_runner is None:
            raise ConfigError(
                f"Method '{spec.method_name}' does not support experiment runs."
            )
        return spec.experiment_runner(config_path, config)
    if config["run_type"] == "single":
        return run_single_from_spec(spec, config_path, config)
    return run_batch_from_spec(spec, config_path, config)
