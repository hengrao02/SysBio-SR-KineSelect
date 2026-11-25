"""System-level precomputation for KineSelectSR KFI."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from Methods.KineSelectSR.formulation import build_target, safe_x_dot_max


@dataclass
class TargetContext:
    """Precomputed derivative target and normalization context for KFI."""

    X_dot: np.ndarray
    X_dot_max_safe: np.ndarray
    target_norm: np.ndarray
    denom: float
    D: int
    T: int


def prepare_target_context(t: np.ndarray, X: np.ndarray) -> TargetContext:
    """Compute target quantities from raw trajectory data."""
    t = np.asarray(t, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D [T, D], got shape {X.shape}")
    T, D = X.shape

    if t.shape[0] != T:
        raise ValueError(f"t length ({t.shape[0]}) must match X rows ({T})")
    if T < 2:
        raise ValueError(f"Need at least 2 time points, got {T}")
    if not np.all(np.isfinite(t)):
        raise ValueError("t contains non-finite values")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    # 1. Numerical derivatives
    X_dot = np.gradient(X, t, axis=0)

    # 2. Per-species max absolute derivative, with safety clipping
    X_dot_max = np.max(np.abs(X_dot), axis=0, keepdims=True)
    X_dot_max_safe = safe_x_dot_max(X_dot_max, D)

    # 3. Build normalised target vector
    target_norm = build_target(X_dot, X_dot_max_safe)

    # 4. Denominator for NMSE computation
    denom = max(float(np.dot(target_norm, target_norm)), 1e-300)

    return TargetContext(
        X_dot=X_dot,
        X_dot_max_safe=X_dot_max_safe,
        target_norm=target_norm,
        denom=denom,
        D=D,
        T=T,
    )
