"""Full evaluation pipeline for kinetic ODE models."""

from __future__ import annotations

import numpy as np

from .complexity import compute_complexity
from .fitter import (
    build_design_matrix,
    build_target,
    compute_contributions,
    fit_rate_constants,
    safe_x_dot_max,
    threshold_by_contribution,
)
from .ode import rollout
from .reaction import KineSelectSREvalMetrics, FormulationConfig, KineSelectSRModel, validate_model


def evaluate(
    model: KineSelectSRModel,
    t: np.ndarray,
    X: np.ndarray,
    config: FormulationConfig | None = None,
) -> KineSelectSREvalMetrics:
    """Evaluate a kinetic model on a trajectory — the full pipeline."""
    if config is None:
        config = FormulationConfig()

    t = np.asarray(t, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)

    # --- Input validation ---
    if X.ndim != 2:
        raise ValueError(f"X must be 2D [T, D], got shape {X.shape}")
    T, D = X.shape

    if t.shape[0] != T:
        raise ValueError(f"t length ({len(t)}) must match X rows ({T})")
    if T < 2:
        raise ValueError(f"Need at least 2 time points, got {T}")
    if not np.all(np.isfinite(t)):
        raise ValueError("t contains non-finite values")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    _ = validate_model(model)

    # --- 1. Numerical derivatives ---
    X_dot = np.gradient(X, t, axis=0)

    # --- 2. Build normalized design matrix and target ---
    X_dot_max = np.max(np.abs(X_dot), axis=0, keepdims=True)
    X_dot_max_safe = safe_x_dot_max(X_dot_max, D)
    A_norm = build_design_matrix(model, X, X_dot_max_safe)
    target_norm = build_target(X_dot, X_dot_max_safe)

    # --- 3. NNLS fit ---
    denom = max(np.dot(target_norm, target_norm), 1e-300)
    m = len(model.reactions)

    if m == 0:
        k_raw = np.array([])
        rnorm_raw = float(np.linalg.norm(target_norm))
    else:
        try:
            k_raw, rnorm_raw = fit_rate_constants(A_norm, target_norm, l2_lambda=config.l2_lambda)
        except RuntimeError:
            return KineSelectSREvalMetrics(
                deriv_scaled_nmse=float("inf"),
                deriv_scaled_nmse_raw=float("inf"),
                rollout_nmse=float("inf"),
                complexity=0,
                rate_constants=np.zeros(m),
                n_active_reactions=0,
            )

    deriv_scaled_nmse_raw = rnorm_raw**2 / denom

    # --- 4-5. Threshold + recompute NMSE ---
    contributions = compute_contributions(A_norm, k_raw, target_norm)
    k_active = threshold_by_contribution(k_raw, contributions,
                                         max_ratio=config.rate_threshold_ratio)
    k_active = np.where(np.abs(k_active) > 1e-14, k_active, 0.0)

    # Recompute residual with k_active
    residual = target_norm - A_norm @ k_active
    deriv_scaled_nmse = float(np.dot(residual, residual)) / denom

    # --- 7. Set model rate constants ---
    model.rate_constants = k_active

    # --- 8-9. Rollout ---
    y0 = X[0, :].copy()

    if m == 0 or np.all(np.abs(k_active) <= 1e-14):
        X_hat = np.repeat(y0.reshape(1, -1), T, axis=0)
    else:
        X_hat = rollout(model, t, y0, config, timeout=config.rollout_timeout)

    if not np.all(np.isfinite(X_hat)):
        rollout_nmse = float("inf")
    else:
        rollout_nmse = _per_species_nmse(X, X_hat)

    # --- 10. Complexity ---
    complexity = compute_complexity(model)

    # --- 11. Assemble result ---
    n_active = int(np.sum(np.abs(k_active) > 1e-14))

    return KineSelectSREvalMetrics(
        deriv_scaled_nmse=deriv_scaled_nmse,
        deriv_scaled_nmse_raw=deriv_scaled_nmse_raw,
        rollout_nmse=rollout_nmse,
        complexity=complexity,
        rate_constants=k_active,
        n_active_reactions=n_active,
    )


def _per_species_nmse(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """Compute per-species NMSE averaged over D species."""
    T, D = X_true.shape

    if D == 0:
        return 0.0

    nmse_values = np.zeros(D)
    for d in range(D):
        x_d = X_true[:, d]
        x_hat_d = X_pred[:, d]
        numerator = np.sum((x_d - x_hat_d) ** 2)
        denominator = np.sum((x_d - np.mean(x_d)) ** 2)

        if denominator < 1e-15:
            nmse_values[d] = 0.0 if numerator < 1e-15 else float("inf")
        else:
            nmse_values[d] = numerator / denominator

    return float(np.mean(nmse_values))
