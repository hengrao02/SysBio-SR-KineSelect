"""Per-reaction rate computation — scalar and vectorized."""

from __future__ import annotations

import numpy as np

from .reaction import KineSelectSRKineticVariant


# ---------------------------------------------------------------------------
# Scalar rate (for ODE integration)
# ---------------------------------------------------------------------------

def compute_rate(reaction: KineSelectSRKineticVariant, x: np.ndarray) -> float:
    """Compute the kinetic rate factor (without k) at a single time point."""
    D = len(x)
    if D != len(reaction.reactants):
        raise ValueError(
            f"Dimension mismatch: x has length {D}, "
            f"but reaction expects {len(reaction.reactants)} species"
        )
    reactants = np.array(reaction.reactants, dtype=int)
    x_clipped = np.maximum(x, 0.0)  # negative concentration protection

    # --- 1. Substrate (mass-action) part ---
    core = _compute_substrate_factor(
        x_clipped, reactants, reaction.substrate_law,
        reaction.substrate_idx, reaction.substrate_K,
    )

    # --- 2. Modifier / regulator part ---
    factor = _compute_modifier_factor(
        x_clipped, reaction.modifier_law,
        reaction.modifier_idx, reaction.modifier_K, reaction.modifier_n,
    )

    return core * factor


# ---------------------------------------------------------------------------
# Vectorized rate (for NNLS fitting)
# ---------------------------------------------------------------------------

def compute_rate_batch(
    reaction: KineSelectSRKineticVariant,
    X: np.ndarray,
) -> np.ndarray:
    """Compute the kinetic rate factor (without k) over all time points."""
    T, D = X.shape
    if D != len(reaction.reactants):
        raise ValueError(
            f"Dimension mismatch: X has {D} species columns, "
            f"but reaction expects {len(reaction.reactants)} species"
        )
    reactants = np.array(reaction.reactants, dtype=int)
    X_clipped = np.maximum(X, 0.0)  # [T, D]

    # --- 1. Substrate (mass-action) part ---
    core = _compute_substrate_factor_batch(
        X_clipped, reactants, reaction.substrate_law,
        reaction.substrate_idx, reaction.substrate_K,
    )

    # --- 2. Modifier / regulator part ---
    factor = _compute_modifier_factor_batch(
        X_clipped, reaction.modifier_law,
        reaction.modifier_idx, reaction.modifier_K, reaction.modifier_n,
    )

    return core * factor


# ---------------------------------------------------------------------------
# Internal: substrate factor (scalar)
# ---------------------------------------------------------------------------

def _compute_substrate_factor(
    x: np.ndarray,
    reactants: np.ndarray,
    substrate_law: str,
    substrate_idx: int | None,
    substrate_K: float | None,
) -> float:
    """Compute the substrate kinetics part of the rate (scalar)."""
    D = len(x)
    core = 1.0

    for i in range(D):
        order = int(reactants[i])
        if order <= 0:
            continue

        if substrate_law == "mm" and i == substrate_idx:
            # MM: one power replaced by saturation term
            if order > 1:
                core *= x[i] ** (order - 1)
        else:
            core *= x[i] ** order

    # Apply the MM saturation factor
    if substrate_law == "mm" and substrate_idx is not None:
        val = x[substrate_idx]
        denom = substrate_K + val
        core *= (val / denom) if denom > 0.0 else 0.0

    return core


# ---------------------------------------------------------------------------
# Internal: substrate factor (vectorized)
# ---------------------------------------------------------------------------

def _compute_substrate_factor_batch(
    X: np.ndarray,
    reactants: np.ndarray,
    substrate_law: str,
    substrate_idx: int | None,
    substrate_K: float | None,
) -> np.ndarray:
    """Compute the substrate kinetics part of the rate (vectorized over T)."""
    T, D = X.shape
    core = np.ones(T)

    for i in range(D):
        order = int(reactants[i])
        if order <= 0:
            continue

        if substrate_law == "mm" and i == substrate_idx:
            if order > 1:
                core *= X[:, i] ** (order - 1)
        else:
            core *= X[:, i] ** order

    # Apply the MM saturation factor
    if substrate_law == "mm" and substrate_idx is not None:
        val = X[:, substrate_idx]
        core *= val / (substrate_K + val)

    return core


# ---------------------------------------------------------------------------
# Internal: modifier factor (scalar)
# ---------------------------------------------------------------------------

def _compute_modifier_factor(
    x: np.ndarray,
    modifier_law: str,
    modifier_idx: int | None,
    modifier_K: float | None,
    modifier_n: int | None,
) -> float:
    """Compute the modifier/regulation factor (scalar)."""
    if modifier_law == "none":
        return 1.0

    if modifier_idx is None:
        return 1.0

    val = x[modifier_idx]

    if modifier_law == "linear":
        return val

    # Hill activation or inhibition — ratio-based to prevent overflow
    if modifier_law in ("hill_activation", "hill_inhibition"):
        n = modifier_n or 1
        K = modifier_K or 1.0

        if val >= K:
            ratio = K / val
        else:
            ratio = val / K
        ratio_n = ratio ** n

        if modifier_law == "hill_activation":
            if val >= K:
                factor = 1.0 / (ratio_n + 1.0)
            else:
                factor = ratio_n / (ratio_n + 1.0)
        else:  # hill_inhibition
            if val >= K:
                factor = ratio_n / (ratio_n + 1.0)
            else:
                factor = 1.0 / (ratio_n + 1.0)

        return factor

    return 1.0


# ---------------------------------------------------------------------------
# Internal: modifier factor (vectorized)
# ---------------------------------------------------------------------------

def _compute_modifier_factor_batch(
    X: np.ndarray,
    modifier_law: str,
    modifier_idx: int | None,
    modifier_K: float | None,
    modifier_n: int | None,
) -> np.ndarray:
    """Compute the modifier/regulation factor (vectorized over T)."""
    T = X.shape[0]

    if modifier_law == "none":
        return np.ones(T)

    if modifier_idx is None:
        return np.ones(T)

    val = X[:, modifier_idx]

    if modifier_law == "linear":
        return val.copy()

    # Hill activation or inhibition — ratio-based to prevent overflow
    if modifier_law in ("hill_activation", "hill_inhibition"):
        n = modifier_n or 1
        K = modifier_K or 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                val >= K,
                K / np.maximum(val, 1e-300),
                val / K,
            )
        ratio_n = ratio ** n

        if modifier_law == "hill_activation":
            factor = np.where(
                val >= K,
                1.0 / (ratio_n + 1.0),
                ratio_n / (ratio_n + 1.0),
            )
        else:  # hill_inhibition
            factor = np.where(
                val >= K,
                ratio_n / (ratio_n + 1.0),
                1.0 / (ratio_n + 1.0),
            )

        return factor

    return np.ones(T)
