"""NNLS coefficient fitting for kinetic ODE models."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .rate import compute_rate_batch
from .reaction import KineSelectSRModel, KineSelectSRKineticVariant


# ---------------------------------------------------------------------------
# X_dot_max safety
# ---------------------------------------------------------------------------

def safe_x_dot_max(
    X_dot_max: np.ndarray,
    D: int,
    floor: float = 1e-14,
) -> np.ndarray:
    """Validate and clip X_dot_max for numerical safety."""
    arr = np.asarray(X_dot_max, dtype=float)
    if arr.shape != (1, D):
        raise ValueError(
            f"X_dot_max must have shape (1, {D}), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("X_dot_max must be finite for all species")
    arr = arr.copy()
    arr[arr < floor] = 1.0
    return arr


# ---------------------------------------------------------------------------
# Design matrix construction
# ---------------------------------------------------------------------------

def build_design_matrix(
    model_or_variants: KineSelectSRModel | list[KineSelectSRKineticVariant] | tuple[KineSelectSRKineticVariant, ...],
    X: np.ndarray,
    X_dot_max: np.ndarray,
) -> np.ndarray:
    """Build the normalized design matrix for NNLS fitting."""
    T, D = X.shape
    if hasattr(model_or_variants, "reactions"):
        reactions = model_or_variants.reactions
    else:
        reactions = model_or_variants

    m = len(reactions)
    X_dot_max_safe = safe_x_dot_max(X_dot_max, D)

    if m == 0:
        return np.zeros((T * D, 0))

    A = np.zeros((T * D, m))

    for j, rxn in enumerate(reactions):
        props = compute_rate_batch(rxn, X)  # [T]
        stoich = np.array(rxn.products, dtype=int) - np.array(rxn.reactants, dtype=int)

        col = np.zeros(T * D)
        for i in range(D):
            if stoich[i] != 0:
                col[i::D] = props * stoich[i] / X_dot_max_safe[0, i]
        A[:, j] = col

    return A


def build_target(
    X_dot: np.ndarray,
    X_dot_max: np.ndarray,
) -> np.ndarray:
    """Build the normalized target vector."""
    T, D = X_dot.shape
    X_dot_max_safe = safe_x_dot_max(X_dot_max, D)
    return (X_dot / X_dot_max_safe).reshape(-1)


# ---------------------------------------------------------------------------
# NNLS solver
# ---------------------------------------------------------------------------

def fit_rate_constants(
    A_norm: np.ndarray,
    target_norm: np.ndarray,
    l2_lambda: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Fit rate constants k >= 0 by NNLS given a pre-built design matrix."""
    if l2_lambda < 0.0:
        raise ValueError(f"l2_lambda must be >= 0, got {l2_lambda}")

    if A_norm.ndim != 2:
        raise ValueError(f"A_norm must be 2D, got shape {A_norm.shape}")
    if target_norm.ndim != 1:
        raise ValueError(f"target_norm must be 1D, got shape {target_norm.shape}")
    if A_norm.shape[0] != target_norm.shape[0]:
        raise ValueError(
            f"A_norm has {A_norm.shape[0]} rows but target_norm has "
            f"{target_norm.shape[0]} elements — must match"
        )

    m = A_norm.shape[1]
    if m == 0:
        raise ValueError("Cannot fit: design matrix has zero columns (no reactions)")

    A_solve = A_norm
    b_solve = target_norm
    if l2_lambda > 0.0:
        eye_m = np.eye(m, dtype=A_norm.dtype)
        A_solve = np.vstack([A_norm, np.sqrt(l2_lambda) * eye_m])
        b_solve = np.hstack([target_norm, np.zeros(m, dtype=target_norm.dtype)])

    try:
        k_raw, rnorm_raw = nnls(A_solve, b_solve, maxiter=max(1000, 10 * m))
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"NNLS failed: {exc}") from exc

    return k_raw, float(rnorm_raw)


# ---------------------------------------------------------------------------
# Rate thresholding
# ---------------------------------------------------------------------------

def threshold_rates(
    ks: np.ndarray,
    max_ratio: float = 5.0,
) -> np.ndarray:
    """Zero out rate constants far below the largest (log10-ratio threshold)."""
    ks = np.asarray(ks, dtype=float)
    nonzero = np.abs(ks) > 1e-300
    if not np.any(nonzero):
        return np.zeros_like(ks)

    log_ks = np.full_like(ks, -np.inf, dtype=float)
    log_ks[nonzero] = np.log10(np.abs(ks[nonzero]))
    kmax = float(np.max(log_ks[nonzero]))
    valid = log_ks >= kmax - max_ratio

    return np.where(valid, ks, 0.0)


# ---------------------------------------------------------------------------
# Contribution-based thresholding
# ---------------------------------------------------------------------------

def compute_contributions(
    A_norm: np.ndarray,
    ks: np.ndarray,
    target_norm: np.ndarray,
) -> np.ndarray:
    """Compute per-reaction fractional contribution to the target."""
    target_norm_val = float(np.linalg.norm(target_norm))
    if target_norm_val < 1e-15:
        return np.zeros(len(ks))

    m = A_norm.shape[1]
    contributions = np.empty(m, dtype=float)
    for j in range(m):
        contributions[j] = float(np.linalg.norm(A_norm[:, j] * ks[j])) / target_norm_val
    return contributions


def threshold_by_contribution(
    ks: np.ndarray,
    contributions: np.ndarray,
    max_ratio: float = 5.0,
) -> np.ndarray:
    """Zero out rate constants with negligible contribution."""
    ks = np.asarray(ks, dtype=float)
    contributions = np.asarray(contributions, dtype=float)

    mask = contributions > 1e-300
    if not mask.any():
        return np.zeros_like(ks)

    c_max = float(np.max(contributions[mask]))
    threshold = c_max / (10.0 ** max_ratio)
    return np.where(contributions >= threshold, ks, 0.0)
