"""Data structures for KineSelectSR formulation."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class KineSelectSRReactionGene:
    """A reaction gene representing stoichiometric reactants and products,
    and optionally a modifier index. Used as the blueprint for kinetics.
    """
    reactants: tuple[int, ...]
    products: tuple[int, ...]
    modifier_idx: int | None = None

    def __post_init__(self):
        # Canonicalize numpy integers to Python int
        if isinstance(self.reactants, tuple):
            needs_fix = any(type(x) is not int and isinstance(x, numbers.Integral)
                           for x in self.reactants)
            if needs_fix:
                object.__setattr__(self, "reactants",
                    tuple(int(x) for x in self.reactants))
        if isinstance(self.products, tuple):
            needs_fix = any(type(x) is not int and isinstance(x, numbers.Integral)
                           for x in self.products)
            if needs_fix:
                object.__setattr__(self, "products",
                    tuple(int(x) for x in self.products))
        if self.modifier_idx is not None:
            object.__setattr__(self, "modifier_idx", int(self.modifier_idx))

        if not isinstance(self.reactants, tuple) or not all(
            isinstance(x, numbers.Integral) for x in self.reactants
        ):
            raise TypeError("reactants must be a tuple of integers")
        if not isinstance(self.products, tuple) or not all(
            isinstance(x, numbers.Integral) for x in self.products
        ):
            raise TypeError("products must be a tuple of integers")
        if len(self.reactants) != len(self.products):
            raise ValueError("reactants and products must have the same length")
        if any(x < 0 for x in self.reactants):
            raise ValueError("reactant coefficients must be non-negative")
        if any(x < 0 for x in self.products):
            raise ValueError("product coefficients must be non-negative")


@dataclass(frozen=True)
class KineSelectSRKineticVariant:
    """A single reaction with explicit kinetic law and optional modifier.
    This is the KineSelectSR-local kinetic variant selected by the SINDy profiler.
    """
    reactants: tuple[int, ...]      # [D] stoichiometric coefficients (≥ 0)
    products: tuple[int, ...]       # [D] stoichiometric coefficients (≥ 0)

    # Substrate kinetic law (applied to one reactant species)
    substrate_law: str              # "mass_action" | "mm"
    substrate_idx: int | None       # which reactant is saturated (None for MA)
    substrate_K: float | None       # MM half-saturation constant (None for MA)

    # Modifier / regulator (external species influencing rate)
    modifier_law: str               # "none" | "linear" | "hill_activation" | "hill_inhibition"
    modifier_idx: int | None        # which species is the regulator (None if no modifier)
    modifier_K: float | None        # Hill K (None if no modifier or linear)
    modifier_n: int | None          # Hill coefficient (None if no modifier or linear)

    def __post_init__(self):
        # Canonicalize numpy integers to Python int
        if isinstance(self.reactants, tuple):
            needs_fix = any(type(x) is not int and isinstance(x, numbers.Integral)
                           for x in self.reactants)
            if needs_fix:
                object.__setattr__(self, "reactants",
                    tuple(int(x) for x in self.reactants))
        if isinstance(self.products, tuple):
            needs_fix = any(type(x) is not int and isinstance(x, numbers.Integral)
                           for x in self.products)
            if needs_fix:
                object.__setattr__(self, "products",
                    tuple(int(x) for x in self.products))

        # Type and basic consistency checks
        if not isinstance(self.reactants, tuple) or not all(
            isinstance(x, numbers.Integral) for x in self.reactants
        ):
            raise TypeError("reactants must be a tuple of integers")
        if not isinstance(self.products, tuple) or not all(
            isinstance(x, numbers.Integral) for x in self.products
        ):
            raise TypeError("products must be a tuple of integers")
        if len(self.reactants) != len(self.products):
            raise ValueError("reactants and products must have the same length")
        if any(x < 0 for x in self.reactants):
            raise ValueError("reactant coefficients must be non-negative")
        if any(x < 0 for x in self.products):
            raise ValueError("product coefficients must be non-negative")

        # Substrate consistency
        if self.substrate_law == "mass_action":
            if self.substrate_idx is not None or self.substrate_K is not None:
                raise ValueError("MA law: substrate_idx and substrate_K must be None")
        elif self.substrate_law == "mm":
            if self.substrate_idx is None or self.substrate_K is None:
                raise ValueError("MM law: substrate_idx and substrate_K are required")
            if self.substrate_idx < 0 or self.substrate_idx >= len(self.reactants):
                raise ValueError("substrate_idx out of bounds")
            if self.reactants[self.substrate_idx] == 0:
                raise ValueError("MM substrate must be a reactant (stoich > 0)")
            if not (isinstance(self.substrate_K, (int, float)) and np.isfinite(self.substrate_K) and self.substrate_K > 0):
                raise ValueError(f"MM law: substrate_K must be finite and > 0, got {self.substrate_K}")
        else:
            raise ValueError(f"Unknown substrate law: {self.substrate_law}")

        # Modifier consistency
        if self.modifier_law == "none":
            if self.modifier_idx is not None or self.modifier_K is not None or self.modifier_n is not None:
                raise ValueError("No modifier: modifier_idx, modifier_K, and modifier_n must be None")
        elif self.modifier_law == "linear":
            if self.modifier_idx is None:
                raise ValueError("linear modifier requires modifier_idx")
            if self.modifier_idx < 0 or self.modifier_idx >= len(self.reactants):
                raise ValueError("modifier_idx out of bounds")
            if self.modifier_K is not None or self.modifier_n is not None:
                raise ValueError("linear modifier: modifier_K and modifier_n must be None")
        elif self.modifier_law in ("hill_activation", "hill_inhibition"):
            if self.modifier_idx is None or self.modifier_K is None or self.modifier_n is None:
                raise ValueError(f"{self.modifier_law} requires modifier_idx, modifier_K, and modifier_n")
            if self.modifier_idx < 0 or self.modifier_idx >= len(self.reactants):
                raise ValueError("modifier_idx out of bounds")
            if not (isinstance(self.modifier_K, (int, float)) and np.isfinite(self.modifier_K) and self.modifier_K > 0):
                raise ValueError(f"{self.modifier_law}: modifier_K must be finite and > 0, got {self.modifier_K}")
            if not (isinstance(self.modifier_n, numbers.Integral) and self.modifier_n >= 1):
                raise ValueError(f"{self.modifier_law}: modifier_n must be int >= 1, got {self.modifier_n}")
        else:
            raise ValueError(f"Unknown modifier law: {self.modifier_law}")

        # Overlap rules (v1)
        if self.modifier_idx is not None and self.reactants[self.modifier_idx] > 0:
            raise ValueError("v1: modifier must not be a reactant")
        if self.substrate_idx is not None and self.substrate_idx == self.modifier_idx:
            raise ValueError("substrate_idx and modifier_idx must differ")


class KineSelectSRModel:
    """A complete ODE model: a list of kinetic variants + fitted rate constants.
    Contains bidirectional synchronization between `reactions` and `selected_variants`.
    """
    def __init__(
        self,
        reactions: tuple[KineSelectSRKineticVariant, ...] | None = None,
        rate_constants: np.ndarray | None = None,
        deriv_loss: float | None = None,
        selected_variants: tuple[KineSelectSRKineticVariant, ...] | None = None,
    ):
        val = reactions if reactions is not None else selected_variants
        if val is None:
            val = ()
        self._variants = tuple(val)
        self.rate_constants = rate_constants
        self.deriv_loss = deriv_loss

    @property
    def reactions(self) -> tuple[KineSelectSRKineticVariant, ...]:
        return self._variants

    @reactions.setter
    def reactions(self, val: tuple[KineSelectSRKineticVariant, ...]):
        self._variants = tuple(val)

    @property
    def selected_variants(self) -> tuple[KineSelectSRKineticVariant, ...]:
        return self._variants

    @selected_variants.setter
    def selected_variants(self, val: tuple[KineSelectSRKineticVariant, ...]):
        self._variants = tuple(val)

    def __repr__(self) -> str:
        return f"KineSelectSRModel(reactions={self.reactions}, rate_constants={self.rate_constants}, deriv_loss={self.deriv_loss})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, KineSelectSRModel):
            return False
        return (
            self.reactions == other.reactions
            and np.array_equal(self.rate_constants, other.rate_constants)
            and self.deriv_loss == other.deriv_loss
        )


def validate_model(model: KineSelectSRModel) -> int:
    """Validate model consistency and return the species dimension D."""
    if not model.reactions:
        return 0

    D = len(model.reactions[0].reactants)
    for j, rxn in enumerate(model.reactions):
        if len(rxn.reactants) != D:
            raise ValueError(
                f"All reactions must have the same dimension. "
                f"Reaction 0 has {D} species, reaction {j} has "
                f"{len(rxn.reactants)} species."
            )

    if model.rate_constants is not None:
        if len(model.rate_constants) != len(model.reactions):
            raise ValueError(
                f"rate_constants length ({len(model.rate_constants)}) "
                f"must match number of reactions ({len(model.reactions)})"
            )

    return D


@dataclass
class KineSelectSREvalMetrics:
    """Evaluation result for a KineSelectSRModel on a dataset."""
    deriv_scaled_nmse: float        # NMSE in per-species-scaled space, with k_active
    deriv_scaled_nmse_raw: float    # Same formula, with k_raw (diagnostic only)
    rollout_nmse: float             # per-species NMSE averaged over D; inf if integration failed
    complexity: int                 # AST node count (active reactions only, k_active > threshold)
    rate_constants: np.ndarray      # [m] post-threshold k_active values
    n_active_reactions: int         # reactions with k > threshold


@dataclass
class KineSelectSRFormulationConfig:
    """Defines the discrete search space boundaries."""
    max_reactant_order: int = 2          # max sum of reactant stoichiometry
    max_product_order: int = 2           # max sum of product stoichiometry
    max_stoich_per_species: int = 2      # max stoich coefficient for one species
    substrate_laws: list[str] = field(default_factory=lambda: ["mass_action", "mm"])
    modifier_laws: list[str] = field(default_factory=lambda: ["none", "linear", "hill_activation", "hill_inhibition"])
    K_grid: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    hill_n_values: list[int] = field(default_factory=lambda: [1, 2, 4])
    species_K_grid: list[list[float]] | None = None
    max_reactions_per_model: int = 8
    min_reactions_per_model: int = 1

    # Numerical policies
    negative_state_policy: str = "clip_or_fail"
    negative_state_tol: float = 1e-10
    rate_threshold_ratio: float = 5.0     # max k / min k ratio for thresholding
    rollout_timeout: float | None = 300.0  # wall-clock timeout (s) for solve_ivp
    l2_lambda: float = 1e-6   # L2 ridge penalty for NNLS (0 = no regularization)


FormulationConfig = KineSelectSRFormulationConfig
