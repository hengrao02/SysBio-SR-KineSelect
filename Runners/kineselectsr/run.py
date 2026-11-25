"""KineSelectSR runner — genetic algorithm for ODE discovery.

Usage:
    # Reduced single-system smoke
    python run.py Configs/smoke/kineselectsr_tiny.yaml

    # Full paper-budget main partition
    python run.py Configs/benchmark/kineselectsr/main.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
from dataclasses import fields
from pathlib import Path
from typing import Any

from Runners.core import (
    BaseMethodSingleSystemRunner,
    MethodSpec,
    finalize_run_config,
    run_batch_from_spec,
    run_from_spec,
    run_single_from_spec,
)
from Wrappers.base_ode_sr_wrapper import EquationResult
from Utils import (
    ConfigError,
    load_yaml_config_with_sources,
    resolve_config_key_path,
    validate_config_keys,
)
from Wrappers.kineselectsr_wrapper import KineSelectSRWrapper
from Methods.KineSelectSR.formulation import FormulationConfig
from Methods.KineSelectSR.search import KineSelectSRSearchConfig
from Methods.KineSelectSR.presets import apply_kinetic_preset
from Methods.KineSelectSR.sindy.selector_modes import (
    KFI_SELECTOR,
    normalize_selector_mode,
)
from .artifacts import write_search_artifacts, validate_saved_model

METHOD_NAME = "kineselectsr"
DISPLAY_NAME = "KineSelectSR"
FORMAL_CONFIG_MARKER = "kineselectsr-formal-v2"

_FORMAL_CONFIG_VALUES: dict[str, Any] = {
    "sindy_selector_mode": "kfi",
    "selection_objective": "rollout",
    "final_selection_mode": "fpms",
    "fpms_criteria": "DR_KX",
    "sindy_collinearity_pruning": True,
    "min_reactions": 2,
    "max_reactions": 12,
    "population_size": 2000,
    "num_generations": 20,
}


def _validate_formal_config_marker(
    config: dict[str, Any],
    *,
    config_path: Path,
) -> str | None:
    """Validate an explicit formal-configuration marker before applying defaults.

    Configs without a marker remain ordinary adhoc configs. Once the key is
    present, however, it is fail-closed: neither defaults, legacy aliases nor
    later numeric coercion may be used to satisfy the formal paper contract.
    """
    if "formal_config_marker" not in config:
        return None

    marker = config["formal_config_marker"]
    if marker is None:
        raise ConfigError(
            f"'formal_config_marker' must not be null in formal KineSelectSR config: {config_path}"
        )
    if marker != FORMAL_CONFIG_MARKER:
        raise ConfigError(
            f"Unknown KineSelectSR formal_config_marker={marker!r} in {config_path}; "
            f"expected {FORMAL_CONFIG_MARKER!r}"
        )

    for key, expected in _FORMAL_CONFIG_VALUES.items():
        if key not in config or config[key] is None:
            raise ConfigError(
                f"KineSelectSR formal config {marker!r} requires an explicit, "
                f"non-null canonical '{key}' value in {config_path}"
            )
        actual = config[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ConfigError(
                f"KineSelectSR formal config {marker!r} requires "
                f"{key}={expected!r}, got {actual!r} in {config_path}"
            )

    if "eps" not in config or config["eps"] is None:
        raise ConfigError(
            f"KineSelectSR formal config {marker!r} requires an explicit, "
            f"non-null 'eps' value in {config_path}"
        )
    eps = config["eps"]
    if (
        isinstance(eps, bool)
        or not isinstance(eps, numbers.Real)
        or not math.isfinite(float(eps))
        or eps <= 0
    ):
        raise ConfigError(
            f"KineSelectSR formal config {marker!r} requires eps > 0, "
            f"got {eps!r} in {config_path}"
        )
    return marker


def _float_list_config(value: Any, *, key: str, default: list[float]) -> list[float]:
    """Parse a YAML list of floats with a clear config error on misuse."""
    if value is None:
        return list(default)
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"'{key}' must be a list of numbers.")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}' must be a list of numbers.") from exc


def _reaction_count_config(
    config: dict[str, Any],
    *,
    canonical: str,
    alias: str,
    default: int,
) -> int:
    """Resolve canonical reaction-count config with a deprecated alias.

    ``min_reactions`` / ``max_reactions`` are the GA-facing keys. The
    ``*_per_model`` names are accepted only for legacy configs and are not
    emitted into the resolved run config.
    """
    has_canonical = canonical in config
    has_alias = alias in config

    if has_canonical and has_alias:
        canonical_value = int(config[canonical])
        alias_value = int(config[alias])
        if canonical_value != alias_value:
            raise ConfigError(
                f"Conflicting reaction-count config: {canonical}={canonical_value} "
                f"but deprecated alias {alias}={alias_value}. Set only "
                f"'{canonical}' (preferred) or keep both equal."
            )
        return canonical_value

    if has_canonical:
        return int(config[canonical])
    if has_alias:
        return int(config[alias])
    return int(default)


# =============================================================================
# Single-system runner
# =============================================================================


class KineSelectSRSingleSystemRunner(BaseMethodSingleSystemRunner):
    """Runner for KineSelectSR on a single system."""

    initial_equation_metadata_exclude_keys = ()

    def build_wrapper(self, *, output_json: Path, **kwargs: Any) -> KineSelectSRWrapper:
        # The release marker is provenance for resolved configs/results, not an
        # algorithm argument. ``kwargs`` is a per-call copy, so fit_config still
        # receives the marker below.
        kwargs.pop("formal_config_marker", None)
        self._kineselectsr_wrapper = KineSelectSRWrapper(**kwargs)
        return self._kineselectsr_wrapper

    def build_method_fit_config(
        self,
        *,
        wrapper: KineSelectSRWrapper,
        output_json: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        fit_config = {
            "method_version": 2,
            "random_seed": kwargs.get("random_seed", 42),
            "verbose": bool(kwargs.get("verbose", False)),
            "max_reactant_order": kwargs.get("max_reactant_order", 2),
            "max_product_order": kwargs.get("max_product_order", 2),
            "max_stoich_per_species": kwargs.get("max_stoich_per_species", 2),
            "kinetic_preset": kwargs.get("kinetic_preset", None),
            "kinetic_preset_phase": kwargs.get("kinetic_preset_phase", None),
            "substrate_laws": kwargs.get("substrate_laws", None),
            "modifier_laws": kwargs.get("modifier_laws", None),
            "K_grid": kwargs.get("K_grid", None),
            "hill_n_values": kwargs.get("hill_n_values", None),
            "species_K_grid": kwargs.get("species_K_grid", None),
            "negative_state_policy": kwargs.get(
                "negative_state_policy", "clip_or_fail"
            ),
            "negative_state_tol": kwargs.get("negative_state_tol", 1e-10),
            "rate_threshold_ratio": kwargs.get("rate_threshold_ratio", 5.0),
            "rollout_timeout": kwargs.get("rollout_timeout", 300.0),
            "l2_lambda": kwargs.get("l2_lambda", 1e-6),
            "population_size": kwargs.get("population_size", 100),
            "num_generations": kwargs.get("num_generations", 50),
            "elitism": kwargs.get("elitism", 0.1),
            "min_reactions": kwargs.get("min_reactions", 1),
            "max_reactions": kwargs.get("max_reactions", 8),
            "mutation_rate": kwargs.get("mutation_rate", 0.3),
            "crossover_rate": kwargs.get("crossover_rate", 0.7),
            "tournament_size": kwargs.get("tournament_size", 3),
            "alpha": kwargs.get("alpha", 1.05),
            "eps": kwargs.get("eps", 1e-12),
            "selection_objective": kwargs.get("selection_objective", "rollout"),
            "rollout_per_island": kwargs.get("rollout_per_island", 32),
            "final_selection_mode": kwargs.get("final_selection_mode", "fpms"),
            "fpms_active_threshold": kwargs.get("fpms_active_threshold", 1e-14),
            "fpms_criteria": kwargs.get("fpms_criteria", "DR_KX"),
            "island_early_stop_patience": kwargs.get("island_early_stop_patience", 0),
            "island_early_stop_min_rel_improvement": kwargs.get(
                "island_early_stop_min_rel_improvement", 0.0
            ),
            "sindy_K_grid_mode": kwargs.get("sindy_K_grid_mode", "species_quantile"),
            "sindy_species_K_quantiles": kwargs.get(
                "sindy_species_K_quantiles", [0.05, 0.50]
            ),
            "sindy_species_K_positive_eps": kwargs.get(
                "sindy_species_K_positive_eps", 1e-12
            ),
            "sindy_species_K_positive_rel_eps": kwargs.get(
                "sindy_species_K_positive_rel_eps", 1e-6
            ),
            "sindy_species_K_floor": kwargs.get("sindy_species_K_floor", 1e-12),
            "sindy_species_K_dedup_rtol": kwargs.get(
                "sindy_species_K_dedup_rtol", 0.10
            ),
            "sindy_selector_mode": normalize_selector_mode(
                kwargs.get("sindy_selector_mode", KFI_SELECTOR),
                field="sindy_selector_mode",
            ),
            "sindy_active_threshold": kwargs.get("sindy_active_threshold", 1e-4),
            "sindy_numerical_zero_threshold": kwargs.get(
                "sindy_numerical_zero_threshold", 1e-14
            ),
            "sindy_top_k_per_reaction": kwargs.get("sindy_top_k_per_reaction", 1),
            "sindy_delta_mod": kwargs.get("sindy_delta_mod", 0.05),
            "sindy_delta_sat": kwargs.get("sindy_delta_sat", 0.05),
            "sindy_collinearity_pruning": kwargs.get(
                "sindy_collinearity_pruning", False
            ),
            "sindy_collinearity_cos_threshold": kwargs.get(
                "sindy_collinearity_cos_threshold", 0.999
            ),
            "sindy_hill_collinearity_cos_threshold": kwargs.get(
                "sindy_hill_collinearity_cos_threshold", 0.995
            ),
        }
        if "formal_config_marker" in kwargs:
            fit_config["formal_config_marker"] = kwargs["formal_config_marker"]
        return fit_config

    def get_predict_kwargs(
        self,
        *,
        wrapper: KineSelectSRWrapper,
        equation_result: EquationResult,
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        return {}

    def run(self, **run_kwargs: Any) -> Path:
        summary_path = Path(super().run(**run_kwargs))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_value = run_kwargs.get("config_path")
        config_path = None if config_value is None else Path(config_value).resolve()
        try:
            artifact_contracts = write_search_artifacts(
                output_dir=summary_path.parent,
                summary=summary,
                wrapper=self._kineselectsr_wrapper,
                config_path=config_path,
            )
        except BaseException as exc:
            self.record_artifact_finalization_failure(summary_path, exc)
            raise
        if artifact_contracts is not None:
            artifact_files = artifact_contracts.pop("artifact_files", {})
            summary["kineselectsr_release_artifacts"] = artifact_contracts
            summary.setdefault("artifacts", {}).update(
                {
                    name: relative
                    for name, relative in artifact_files.items()
                    if (summary_path.parent / relative).is_file()
                }
            )
            summary["complete"] = True
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        if not summary.get("complete"):
            summary["complete"] = True
            from Runners.core.inputs import atomic_json

            atomic_json(summary_path, summary)
        return summary_path


# =============================================================================
# Module-level helpers
# =============================================================================


def get_method_spec() -> MethodSpec:
    return MethodSpec(
        method_name=METHOD_NAME,
        display_name=DISPLAY_NAME,
        method_version=2,
        single_runner_cls=KineSelectSRSingleSystemRunner,
        config_loader=load_run_config,
        validate_saved_model=validate_saved_model,
        required_fit_artifacts=frozenset(
            {
                "kineselectsr_alpha_selected_model",
                "kineselectsr_alpha_selector_config",
                "kineselectsr_alpha_selector_summary",
                "kineselectsr_fpms_selected_model",
                "kineselectsr_fpms_selector_config",
                "kineselectsr_fpms_selector_summary",
                "kineselectsr_pareto_front",
                "kineselectsr_resolved_search_config",
                "kineselectsr_search_summary",
            }
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KineSelectSR using one YAML config file.",
    )
    parser.add_argument("config", type=Path, help="Path to a YAML config file.")
    return parser.parse_args()


VALID_CONFIG_KEYS: set[str] = {
    "method",
    "formal_config_marker",
    "run_type",
    "trajectory_name",
    "eval_trajectory_names",
    "system_dir",
    "systems_root",
    "output_json",
    "output_root",
    "results_root",
    "experiment_id",
    "verbose",
    "limit",
    "skip_existing",
    "resume",
    "continue_on_error",
    "recursive",
    "batch_num_workers",
    "benchmark_definition",
    "benchmark_partition",
    "dataset_version",
    "verify_data_integrity",
    # --- Formulation space ---
    "max_reactant_order",
    "max_product_order",
    "max_stoich_per_species",
    "kinetic_preset",
    "kinetic_preset_phase",
    "substrate_laws",
    "modifier_laws",
    "K_grid",
    "hill_n_values",
    "species_K_grid",
    "max_reactions_per_model",
    "min_reactions_per_model",
    "negative_state_policy",
    "negative_state_tol",
    "rate_threshold_ratio",
    "rollout_timeout",
    "l2_lambda",
    # --- GA search ---
    "random_seed",
    "population_size",
    "num_generations",
    "elitism",
    "min_reactions",
    "max_reactions",
    "mutation_rate",
    "crossover_rate",
    "tournament_size",
    "alpha",
    "eps",
    "selection_objective",
    "rollout_per_island",
    "island_early_stop_patience",
    "island_early_stop_min_rel_improvement",
    # --- Final selection ---
    "final_selection_mode",
    "fpms_active_threshold",
    "fpms_criteria",
    "sindy_K_grid_mode",
    "sindy_species_K_quantiles",
    "sindy_species_K_positive_eps",
    "sindy_species_K_positive_rel_eps",
    "sindy_species_K_floor",
    "sindy_species_K_dedup_rtol",
    "sindy_selector_mode",
    "sindy_active_threshold",
    "sindy_numerical_zero_threshold",
    "sindy_top_k_per_reaction",
    "sindy_delta_mod",
    "sindy_delta_sat",
    # --- Collinearity pruning ---
    "sindy_collinearity_pruning",
    "sindy_collinearity_cos_threshold",
    "sindy_hill_collinearity_cos_threshold",
}


def load_run_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path, config, sources = load_yaml_config_with_sources(config_path)
    validate_config_keys(config, valid_keys=VALID_CONFIG_KEYS, config_path=config_path)
    if config.get("method", METHOD_NAME) != METHOD_NAME:
        raise ConfigError(
            f"Expected method={METHOD_NAME!r}, got {config.get('method')!r}"
        )
    formal_config_marker = _validate_formal_config_marker(
        config, config_path=config_path
    )
    try:
        config = apply_kinetic_preset(config)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid KineSelectSR kinetic preset in {config_path}: {exc}"
        ) from exc

    run_type = config.get("run_type")
    if run_type not in {"single", "batch"}:
        raise ConfigError(
            f"'run_type' must be either 'single' or 'batch': {config_path}"
        )

    raw_eval = config.get("eval_trajectory_names") or []
    eval_trajectory_names = (
        [str(name) for name in raw_eval] if isinstance(raw_eval, list) else []
    )

    formulation_defaults = FormulationConfig()
    resolved: dict[str, Any] = {
        "method": METHOD_NAME,
        "run_type": run_type,
        "trajectory_name": str(config.get("trajectory_name", "default")),
        "eval_trajectory_names": eval_trajectory_names,
        "verbose": bool(config.get("verbose", False)),
        "results_root": resolve_config_key_path(
            config, sources, "results_root", fallback_config_path=config_path
        ),
        "benchmark_definition": resolve_config_key_path(
            config, sources, "benchmark_definition", fallback_config_path=config_path
        ),
        "benchmark_partition": (
            None
            if config.get("benchmark_partition") is None
            else str(config["benchmark_partition"])
        ),
        "dataset_version": (
            None
            if config.get("dataset_version") is None
            else str(config["dataset_version"])
        ),
        "verify_data_integrity": bool(config.get("verify_data_integrity", False)),
        # --- Formulation ---
        "max_reactant_order": int(config.get("max_reactant_order", 2)),
        "max_product_order": int(config.get("max_product_order", 2)),
        "max_stoich_per_species": int(config.get("max_stoich_per_species", 2)),
        "kinetic_preset": config.get("kinetic_preset", None),
        "kinetic_preset_phase": config.get("kinetic_preset_phase", None),
        "substrate_laws": (
            list(formulation_defaults.substrate_laws)
            if config.get("substrate_laws") is None
            else config["substrate_laws"]
        ),
        "modifier_laws": (
            list(formulation_defaults.modifier_laws)
            if config.get("modifier_laws") is None
            else config["modifier_laws"]
        ),
        "K_grid": (
            list(formulation_defaults.K_grid)
            if config.get("K_grid") is None
            else config["K_grid"]
        ),
        "hill_n_values": (
            list(formulation_defaults.hill_n_values)
            if config.get("hill_n_values") is None
            else config["hill_n_values"]
        ),
        "species_K_grid": config.get("species_K_grid", None),
        "negative_state_policy": str(
            config.get("negative_state_policy", "clip_or_fail")
        ),
        "negative_state_tol": float(config.get("negative_state_tol", 1e-10)),
        "rate_threshold_ratio": float(config.get("rate_threshold_ratio", 5.0)),
        "rollout_timeout": config.get("rollout_timeout", 300.0),
        "l2_lambda": float(config.get("l2_lambda", 1e-6)),
        # --- GA ---
        "random_seed": int(config.get("random_seed", 42)),
        "population_size": int(config.get("population_size", 100)),
        "num_generations": int(config.get("num_generations", 50)),
        "elitism": float(config.get("elitism", 0.1)),
        "min_reactions": _reaction_count_config(
            config,
            canonical="min_reactions",
            alias="min_reactions_per_model",
            default=1,
        ),
        "max_reactions": _reaction_count_config(
            config,
            canonical="max_reactions",
            alias="max_reactions_per_model",
            default=8,
        ),
        "mutation_rate": float(config.get("mutation_rate", 0.3)),
        "crossover_rate": float(config.get("crossover_rate", 0.7)),
        "tournament_size": int(config.get("tournament_size", 3)),
        "alpha": float(config.get("alpha", 1.05)),
        "eps": float(config.get("eps", 1e-12)),
        "selection_objective": str(config.get("selection_objective", "rollout")),
        "rollout_per_island": int(config.get("rollout_per_island", 32)),
        "island_early_stop_patience": int(config.get("island_early_stop_patience", 0)),
        "island_early_stop_min_rel_improvement": float(
            config.get("island_early_stop_min_rel_improvement", 0.0)
        ),
        "sindy_K_grid_mode": str(config.get("sindy_K_grid_mode", "species_quantile")),
        "sindy_species_K_quantiles": _float_list_config(
            config.get("sindy_species_K_quantiles", [0.05, 0.50]),
            key="sindy_species_K_quantiles",
            default=[0.05, 0.50],
        ),
        "sindy_species_K_positive_eps": float(
            config.get("sindy_species_K_positive_eps", 1e-12)
        ),
        "sindy_species_K_positive_rel_eps": float(
            config.get("sindy_species_K_positive_rel_eps", 1e-6)
        ),
        "sindy_species_K_floor": float(config.get("sindy_species_K_floor", 1e-12)),
        "sindy_species_K_dedup_rtol": float(
            config.get("sindy_species_K_dedup_rtol", 0.10)
        ),
        "sindy_selector_mode": normalize_selector_mode(
            str(config.get("sindy_selector_mode", KFI_SELECTOR)),
            field="sindy_selector_mode",
        ),
        "sindy_active_threshold": float(config.get("sindy_active_threshold", 1e-4)),
        "sindy_numerical_zero_threshold": float(
            config.get("sindy_numerical_zero_threshold", 1e-14)
        ),
        "sindy_top_k_per_reaction": int(config.get("sindy_top_k_per_reaction", 1)),
        "sindy_delta_mod": float(config.get("sindy_delta_mod", 0.05)),
        "sindy_delta_sat": float(config.get("sindy_delta_sat", 0.05)),
        # --- Collinearity pruning ---
        "sindy_collinearity_pruning": bool(
            config.get("sindy_collinearity_pruning", False)
        ),
        "sindy_collinearity_cos_threshold": float(
            config.get("sindy_collinearity_cos_threshold", 0.999)
        ),
        "sindy_hill_collinearity_cos_threshold": float(
            config.get("sindy_hill_collinearity_cos_threshold", 0.995)
        ),
        # --- Final selection ---
        "final_selection_mode": str(config.get("final_selection_mode", "fpms")),
        "fpms_active_threshold": float(config.get("fpms_active_threshold", 1e-14)),
        "fpms_criteria": config.get("fpms_criteria", "DR_KX"),
    }
    if formal_config_marker is not None:
        resolved["formal_config_marker"] = formal_config_marker

    if resolved["rollout_timeout"] is not None:
        resolved["rollout_timeout"] = float(resolved["rollout_timeout"])

    search_fields = {f.name for f in fields(KineSelectSRSearchConfig)}
    try:
        KineSelectSRSearchConfig(
            **{k: v for k, v in resolved.items() if k in search_fields}
        )
        formulation_fields = {f.name for f in fields(FormulationConfig)}
        FormulationConfig(
            **{k: v for k, v in resolved.items() if k in formulation_fields}
        )
    except ValueError as exc:
        raise ConfigError(
            f"Invalid KineSelectSR GA config in {config_path}: {exc}"
        ) from exc

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
