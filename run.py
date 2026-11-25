"""Unified SysBioSR entry point.

Usage:
    python run.py Configs/reactive_sindy/single_system.yaml
    python run.py Configs/smoke/kineselectsr_tiny.yaml
    python run.py Configs/benchmark/kineselectsr/main.yaml

The YAML config must contain a `method` key (set in each Configs/<method>/base.yaml).
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from Runners.core import (
    MethodSpec,
    build_batch_output_root,
    build_result_path,
    run_from_spec,
)
from Utils.config import ConfigError, load_yaml_config


def _load_method_spec(method: str) -> MethodSpec:
    try:
        module = importlib.import_module(f"Runners.{method}")
    except ModuleNotFoundError as exc:
        if exc.name in {f"Runners.{method}", method}:
            raise ConfigError(
                f"Unknown method '{method}'. Expected a runner package at Runners/{method}/ with a METHOD_SPEC export."
            ) from exc
        raise

    spec = getattr(module, "METHOD_SPEC", None)
    if not isinstance(spec, MethodSpec):
        raise ConfigError(
            f"Runner package 'Runners.{method}' must export METHOD_SPEC: MethodSpec(...)."
        )
    return spec


def _apply_cli_overrides(
    config: dict,
    *,
    dataset_root: Path | None,
    results_root: Path | None,
    resume: bool,
) -> dict:
    resolved = dict(config)
    partition = str(resolved.get("benchmark_partition") or "unpartitioned")
    if dataset_root is not None:
        dataset_root = dataset_root.expanduser().resolve()
        if (dataset_root / "benchmark").is_dir():
            dataset_root = dataset_root / "benchmark"
        if resolved["run_type"] == "batch":
            if partition == "unpartitioned":
                raise ConfigError(
                    "--dataset-root requires benchmark_partition for batch runs"
                )
            resolved["systems_root"] = dataset_root / partition
        else:
            system_id = Path(resolved["system_dir"]).name
            if partition == "unpartitioned":
                raise ConfigError(
                    "--dataset-root requires benchmark_partition for single runs"
                )
            resolved["system_dir"] = dataset_root / partition / system_id

    if results_root is not None:
        resolved["results_root"] = results_root.expanduser().resolve()

    if results_root is not None or dataset_root is not None:
        result_root = resolved.get("results_root")
        if result_root is None:
            raise ConfigError(
                "a results_root is required when applying CLI path overrides"
            )
        identity = {
            "experiment_id": str(resolved.get("experiment_id") or "adhoc"),
            "dataset_version": str(resolved.get("dataset_version") or "unversioned"),
            "benchmark_partition": partition,
        }
        if resolved["run_type"] == "single":
            resolved["output_json"] = build_result_path(
                result_root,
                trajectory_name=str(resolved["trajectory_name"]),
                method_name=str(resolved["method"]),
                run_mode="single",
                system_id=Path(resolved["system_dir"]).name,
                seed=int(resolved.get("run_seed", resolved.get("random_seed", 0))),
                **identity,
            )
        else:
            resolved["output_root"] = build_batch_output_root(
                result_root,
                trajectory_name=str(resolved["trajectory_name"]),
                method_name=str(resolved["method"]),
                config_stem=str(resolved.get("config_stem") or "batch"),
                **identity,
            )

    if resume:
        if resolved["run_type"] == "single":
            resolved["resume"] = True
        else:
            resolved["skip_existing"] = True
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified SysBioSR runner. Pass a method YAML config (KineSelectSR, Reactive SINDy, SISR, SINDy, PySR, SINDy-PI, ODEFormer, or FFX).",
    )
    parser.add_argument("config", type=Path, help="Path to a YAML config file.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override the dataset root or the benchmark directory containing main/challenge/stress.",
    )
    parser.add_argument("--results-root", type=Path, help="Override the result root.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete successful results only when recorded inputs, settings, models and evaluations match.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    _, raw_config = load_yaml_config(config_path)

    method = raw_config.get("method")
    if not method:
        raise ConfigError(
            f"'method' key not found in config (after inheritance): {config_path}\n"
            "Add 'method: <name>' to the method's base.yaml."
        )

    spec = _load_method_spec(str(method).strip().lower())
    config_path, config = spec.config_loader(args.config)
    config = _apply_cli_overrides(
        config,
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        resume=args.resume,
    )
    run_from_spec(spec, config_path, config)


if __name__ == "__main__":
    main()
