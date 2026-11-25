from __future__ import annotations
from dataclasses import dataclass, field
import math
import numbers

from .fpms import FPMS_CRITERIA
from ..sindy.selector_modes import KFI_SELECTOR, normalize_selector_mode


def _require_integral(name: str, value: object, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value_int = int(value)
    if min_value is not None and value_int < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value_int}")
    return value_int


def _require_finite_real(
    name: str,
    value: object,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None and value_float < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value_float}")
    if max_value is not None and value_float > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value_float}")
    return value_float

@dataclass
class KineSelectSRSearchConfig:
    """Parameters for the KineSelectSR genetic algorithm search."""
    population_size: int = 100
    num_generations: int = 50
    elitism: float = 0.1
    min_reactions: int = 1
    max_reactions: int = 8
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 3
    random_seed: int = 42

    # Public RC final selectors. ``alpha``/``eps`` also affect tournament
    # pressure, so they are part of the search contract as well as the Alpha
    # selector contract until those two roles are explicitly decoupled.
    final_selection_mode: str = "fpms"   # alpha | fpms
    fpms_active_threshold: float = 1e-14
    fpms_criteria: str = FPMS_CRITERIA

    # Pareto selection hyperparameters
    alpha: float = 1.05
    eps: float = 1e-12
    selection_objective: str = "rollout"
    rollout_per_island: int = 32
    island_early_stop_patience: int = 0
    island_early_stop_min_rel_improvement: float = 0.0

    # SINDy kinetic K-grid controls.
    #   "global"           - use FormulationConfig.K_grid as a fixed grid
    #   "species_quantile" - compute per-species K values from trajectory data
    sindy_K_grid_mode: str = "global"
    sindy_species_K_quantiles: list[float] = field(default_factory=lambda: [0.05, 0.50])
    sindy_species_K_positive_eps: float = 1e-12
    sindy_species_K_positive_rel_eps: float = 1e-6
    sindy_species_K_floor: float = 1e-12
    sindy_species_K_dedup_rtol: float = 0.10
    sindy_active_threshold: float = 1e-4
    sindy_numerical_zero_threshold: float = 1e-14
    sindy_top_k_per_reaction: int = 1
    sindy_delta_mod: float = 0.05
    sindy_delta_sat: float = 0.05
    sindy_selector_mode: str = KFI_SELECTOR
    sindy_collinearity_pruning: bool = False
    sindy_collinearity_cos_threshold: float = 0.999
    sindy_hill_collinearity_cos_threshold: float = 0.995

    def __post_init__(self) -> None:
        self.population_size = _require_integral(
            "population_size", self.population_size, min_value=1
        )
        self.num_generations = _require_integral(
            "num_generations", self.num_generations, min_value=0
        )
        self.min_reactions = _require_integral(
            "min_reactions", self.min_reactions, min_value=0
        )
        self.max_reactions = _require_integral(
            "max_reactions", self.max_reactions, min_value=0
        )
        if self.min_reactions > self.max_reactions:
            raise ValueError(
                "min_reactions must be <= max_reactions, got "
                f"{self.min_reactions} > {self.max_reactions}"
            )
        self.elitism = _require_finite_real(
            "elitism", self.elitism, min_value=0.0, max_value=1.0
        )
        self.mutation_rate = _require_finite_real(
            "mutation_rate", self.mutation_rate, min_value=0.0, max_value=1.0
        )
        self.crossover_rate = _require_finite_real(
            "crossover_rate", self.crossover_rate, min_value=0.0, max_value=1.0
        )
        self.tournament_size = _require_integral(
            "tournament_size", self.tournament_size, min_value=1
        )
        self.alpha = _require_finite_real("alpha", self.alpha, min_value=1.0)
        self.eps = _require_finite_real("eps", self.eps, min_value=0.0)
        self.rollout_per_island = _require_integral(
            "rollout_per_island",
            self.rollout_per_island,
            min_value=0,
        )
        self.island_early_stop_patience = _require_integral(
            "island_early_stop_patience",
            self.island_early_stop_patience,
            min_value=0,
        )
        self.island_early_stop_min_rel_improvement = _require_finite_real(
            "island_early_stop_min_rel_improvement",
            self.island_early_stop_min_rel_improvement,
            min_value=0.0,
        )
        valid_selection_objectives = {"deriv", "rollout"}
        if self.selection_objective not in valid_selection_objectives:
            raise ValueError(
                "selection_objective must be one of "
                f"{valid_selection_objectives}, got {self.selection_objective!r}"
            )
        if self.selection_objective == "rollout" and self.rollout_per_island <= 0:
            raise ValueError(
                "rollout_per_island must be > 0 when "
                "selection_objective='rollout'"
            )

        if self.fpms_criteria != FPMS_CRITERIA:
            raise ValueError(f"fpms_criteria must be {FPMS_CRITERIA!r}, got {self.fpms_criteria!r}")
        valid_final_selection_modes = {"alpha", "fpms"}
        if self.final_selection_mode not in valid_final_selection_modes:
            raise ValueError(
                f"final_selection_mode must be one of "
                f"{valid_final_selection_modes}, got {self.final_selection_mode!r}"
            )
        if self.final_selection_mode == "fpms":
            if self.selection_objective != "rollout":
                raise ValueError(
                    "final_selection_mode='fpms' requires "
                    "selection_objective='rollout'"
                )
            if self.eps <= 0.0:
                raise ValueError("final_selection_mode='fpms' requires eps > 0")
        self.fpms_active_threshold = _require_finite_real(
            "fpms_active_threshold", self.fpms_active_threshold, min_value=0.0
        )

        valid_k_modes = {"global", "species_quantile"}
        if self.sindy_K_grid_mode not in valid_k_modes:
            raise ValueError(
                f"sindy_K_grid_mode must be one of {valid_k_modes}, "
                f"got {self.sindy_K_grid_mode!r}"
            )

        self.sindy_species_K_quantiles = [
            float(q) for q in self.sindy_species_K_quantiles
        ]
        if not self.sindy_species_K_quantiles:
            raise ValueError("sindy_species_K_quantiles must not be empty")
        for i, q in enumerate(self.sindy_species_K_quantiles):
            if not 0.0 <= q <= 1.0:
                raise ValueError(
                    f"sindy_species_K_quantiles[{i}] must be in [0, 1], got {q}"
                )

        if self.sindy_species_K_positive_eps < 0:
            raise ValueError(
                "sindy_species_K_positive_eps must be >= 0, "
                f"got {self.sindy_species_K_positive_eps}"
            )
        if self.sindy_species_K_positive_rel_eps < 0:
            raise ValueError(
                "sindy_species_K_positive_rel_eps must be >= 0, "
                f"got {self.sindy_species_K_positive_rel_eps}"
            )
        if self.sindy_species_K_floor <= 0:
            raise ValueError(
                "sindy_species_K_floor must be > 0, "
                f"got {self.sindy_species_K_floor}"
            )
        if self.sindy_species_K_dedup_rtol < 0:
            raise ValueError(
                "sindy_species_K_dedup_rtol must be >= 0, "
                f"got {self.sindy_species_K_dedup_rtol}"
            )

        self.sindy_active_threshold = _require_finite_real(
            "sindy_active_threshold", self.sindy_active_threshold, min_value=0.0
        )
        self.sindy_numerical_zero_threshold = _require_finite_real(
            "sindy_numerical_zero_threshold",
            self.sindy_numerical_zero_threshold,
            min_value=0.0,
        )
        self.sindy_top_k_per_reaction = _require_integral(
            "sindy_top_k_per_reaction",
            self.sindy_top_k_per_reaction,
            min_value=1,
        )
        self.sindy_delta_mod = _require_finite_real(
            "sindy_delta_mod", self.sindy_delta_mod, min_value=0.0
        )
        self.sindy_delta_sat = _require_finite_real(
            "sindy_delta_sat", self.sindy_delta_sat, min_value=0.0
        )

        self.sindy_selector_mode = normalize_selector_mode(
            self.sindy_selector_mode, field="sindy_selector_mode"
        )
        self.sindy_collinearity_cos_threshold = _require_finite_real(
            "sindy_collinearity_cos_threshold",
            self.sindy_collinearity_cos_threshold,
            min_value=0.0,
            max_value=1.0,
        )
        self.sindy_hill_collinearity_cos_threshold = _require_finite_real(
            "sindy_hill_collinearity_cos_threshold",
            self.sindy_hill_collinearity_cos_threshold,
            min_value=0.0,
            max_value=1.0,
        )
