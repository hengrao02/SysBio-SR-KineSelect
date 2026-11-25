"""Configuration options for KineSelectSR kinetic-form identification."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
import numbers
import numpy as np
from Methods.KineSelectSR.formulation import FormulationConfig
from .selector_modes import KFI_SELECTOR, normalize_selector_mode


_SUPPORTED_SUBSTRATE_LAWS = {"mass_action", "mm"}
_SUPPORTED_MODIFIER_LAWS = {
    "none",
    "linear",
    "hill_activation",
    "hill_inhibition",
}


def _validate_laws(name: str, laws: list[str], supported: set[str]) -> list[str]:
    laws = list(laws)
    if not laws:
        raise ValueError(f"{name} must not be empty")
    normalized = []
    for i, law in enumerate(laws):
        if not isinstance(law, str):
            raise TypeError(f"{name}[{i}] must be a string, got {type(law).__name__}")
        if law not in supported:
            raise ValueError(
                f"Unsupported {name}[{i}]={law!r}; expected one of {sorted(supported)}"
            )
        normalized.append(law)
    return normalized


def _validate_positive_finite_grid(name: str, values: list[float]) -> list[float]:
    values = list(values)
    if not values:
        raise ValueError(f"{name} must not be empty")
    normalized = []
    for i, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name}[{i}] must be finite and > 0, got {value!r}")
        normalized.append(float(value))
    return normalized


def _validate_hill_n_values(values: list[int]) -> list[int]:
    values = list(values)
    if not values:
        raise ValueError("hill_n_values must not be empty")
    normalized = []
    for i, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or int(value) < 1
        ):
            raise ValueError(f"hill_n_values[{i}] must be an int >= 1, got {value!r}")
        normalized.append(int(value))
    return normalized


def _validate_nonnegative_finite(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return float(value)


def _validate_positive_finite(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


def _validate_unit_interval(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        or float(value) > 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return float(value)


def _validate_positive_int(name: str, value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


@dataclass
class SINDyKineticOptions:
    """Configuration for KineSelectSR kinetic-form identification."""

    # Coefficient fitting
    l2_lambda: float = 1e-6

    # --- Three distinct thresholds; do NOT conflate them (see M2 fix) ---
    # active_threshold: DIMENSIONLESS contribution threshold used during
    #   per-gene variant selection. It does NOT prune reactions; it only makes
    #   select_kfi_variants fall back to the simplest MA_none variant for
    #   a gene whose entire variant group contributes below this fraction of the
    #   target (a conservative regularizer against over-fitting near-dead genes).
    #   In the GA full-law-space path a real reaction has contribution ~O(1), so
    #   the default 1e-4 fallback effectively never triggers — set it to 0.0 for
    #   pure contribution-argmax.
    active_threshold: float = 1e-4
    # numerical_zero_threshold: ABSOLUTE |k| noise floor applied AFTER the
    #   relative contribution pruning (rate_threshold_ratio) has already done the
    #   scale-invariant reaction removal. It only cleans up NNLS solver dust, so
    #   it is safe to keep absolute. n_active / complexity counting uses the same
    #   floor so "non-zero in the dynamics <=> counted" holds exactly.
    numerical_zero_threshold: float = 1e-14
    top_k_per_reaction: int = 1
    # KFI is the only public kinetic-form selector.
    selector_mode: str = KFI_SELECTOR
    delta_mod: float = 0.05
    delta_sat: float = 0.05

    # Collinearity pruning
    collinearity_pruning: bool = False
    collinearity_cos_threshold: float = 0.999
    hill_collinearity_cos_threshold: float = 0.995

    # Kinetic law search space
    substrate_laws: list[str] = field(
        default_factory=lambda: ["mass_action", "mm"]
    )
    modifier_laws: list[str] = field(
        default_factory=lambda: ["none", "linear"]
    )
    K_grid: list[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    hill_n_values: list[int] = field(default_factory=lambda: [1])

    # Species-adaptive K grid
    K_grid_mode: str = "species_quantile"
    species_K_quantiles: list[float] = field(default_factory=lambda: [0.05, 0.50])
    species_K_positive_eps: float = 1e-12
    species_K_positive_rel_eps: float = 1e-6
    species_K_floor: float = 1e-12
    species_K_dedup_rtol: float = 0.10
    species_K_grid: list[list[float]] | None = None

    # Numerical
    rate_threshold_ratio: float = 5.0
    negative_state_policy: str = "clip_or_fail"
    negative_state_tol: float = 1e-10
    rollout_timeout: float | None = 300.0
    max_reactant_order: int = 2
    max_product_order: int = 2
    max_stoich_per_species: int = 2
    max_reactions_per_model: int = 8
    min_reactions_per_model: int = 1

    def __post_init__(self):
        self.substrate_laws = _validate_laws(
            "substrate_laws", self.substrate_laws, _SUPPORTED_SUBSTRATE_LAWS
        )
        self.modifier_laws = _validate_laws(
            "modifier_laws", self.modifier_laws, _SUPPORTED_MODIFIER_LAWS
        )
        if "mass_action" not in self.substrate_laws:
            raise ValueError(
                "substrate_laws must include 'mass_action' because the "
                "KFI inactive-group fallback uses the MA_none family."
            )
        if "none" not in self.modifier_laws:
            raise ValueError(
                "modifier_laws must include 'none' because the KFI "
                "inactive-group fallback uses the MA_none family."
            )
        self.K_grid = _validate_positive_finite_grid("K_grid", self.K_grid)
        self.hill_n_values = _validate_hill_n_values(self.hill_n_values)

        if self.species_K_grid is not None:
            if not self.species_K_grid:
                raise ValueError("species_K_grid must not be empty")
            self.species_K_grid = [
                _validate_positive_finite_grid(f"species_K_grid[{i}]", row)
                for i, row in enumerate(self.species_K_grid)
            ]

        _valid_k_modes = {"global", "species_quantile"}
        if self.K_grid_mode not in _valid_k_modes:
            raise ValueError(
                f"K_grid_mode must be one of {_valid_k_modes}, "
                f"got {self.K_grid_mode!r}"
            )

        self.selector_mode = normalize_selector_mode(self.selector_mode)

        if self.collinearity_pruning:
            self.collinearity_cos_threshold = _validate_unit_interval(
                "collinearity_cos_threshold", self.collinearity_cos_threshold
            )
            self.hill_collinearity_cos_threshold = _validate_unit_interval(
                "hill_collinearity_cos_threshold",
                self.hill_collinearity_cos_threshold,
            )

        self.species_K_quantiles = [float(q) for q in self.species_K_quantiles]
        if not self.species_K_quantiles:
            raise ValueError("species_K_quantiles must not be empty")

        for i, q in enumerate(self.species_K_quantiles):
            if not (0.0 <= q <= 1.0):
                raise ValueError(
                    f"species_K_quantiles[{i}] must be in [0, 1], got {q}"
                )

    def to_formulation_config(self) -> FormulationConfig:
        """Convert to FormulationConfig."""
        return FormulationConfig(
            max_reactant_order=self.max_reactant_order,
            max_product_order=self.max_product_order,
            max_stoich_per_species=self.max_stoich_per_species,
            substrate_laws=list(self.substrate_laws),
            modifier_laws=list(self.modifier_laws),
            K_grid=list(self.K_grid),
            hill_n_values=list(self.hill_n_values),
            species_K_grid=(
                [list(row) for row in self.species_K_grid]
                if self.species_K_grid is not None else None
            ),
            max_reactions_per_model=self.max_reactions_per_model,
            min_reactions_per_model=self.min_reactions_per_model,
            negative_state_policy=self.negative_state_policy,
            negative_state_tol=self.negative_state_tol,
            rate_threshold_ratio=self.rate_threshold_ratio,
            rollout_timeout=self.rollout_timeout,
            l2_lambda=self.l2_lambda,
        )


def resolve_sindy_options_for_X(
    options: SINDyKineticOptions,
    X: np.ndarray,
) -> SINDyKineticOptions:
    """Return a copy of options with data-dependent SINDy fields resolved."""
    from Methods.KineSelectSR.formulation.constraints import compute_species_quantile_K_grid

    resolved = copy.deepcopy(options)
    if resolved.K_grid_mode == "species_quantile" and resolved.species_K_grid is None:
        X_arr = np.asarray(X, dtype=float)
        resolved.species_K_grid = compute_species_quantile_K_grid(
            X_arr,
            quantiles=resolved.species_K_quantiles,
            positive_eps=resolved.species_K_positive_eps,
            k_floor=resolved.species_K_floor,
            positive_rel_eps=resolved.species_K_positive_rel_eps,
            dedup_rtol=resolved.species_K_dedup_rtol,
        )
    return resolved
