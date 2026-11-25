from __future__ import annotations

from typing import Any

import numpy as np

METRIC_VERSION = "sysbiosr-trajectory-1.0.0-rc.2"


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else float("nan")
    return 1.0 - ss_res / ss_tot


def _safe_nmse(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= eps:
        return 0.0 if ss_res <= eps else float("nan")
    return ss_res / (ss_tot + eps)


def compute_trajectory_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true{y_true.shape} vs y_pred{y_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D trajectories [T, D], got shape {y_true.shape}")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred must contain only finite values")

    n_states = y_true.shape[1]
    if state_names is None:
        state_names = [f"x_{i}" for i in range(n_states)]
    if len(state_names) != n_states:
        raise ValueError(f"len(state_names)={len(state_names)} does not match D={n_states}")

    per_state: list[dict[str, Any]] = []
    rmse_list, nrmse_list, nmse_list, r2_list, max_abs_list = [], [], [], [], []

    for idx, name in enumerate(state_names):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        diff = yp - yt
        rmse = float(np.sqrt(np.mean(diff**2)))
        denom = float(np.max(yt) - np.min(yt))
        nrmse = float(rmse / denom) if denom > 1e-12 else float("nan")
        nmse = _safe_nmse(yt, yp)
        r2 = _safe_r2(yt, yp)
        max_abs_err = float(np.max(np.abs(diff)))
        final_abs_err = float(np.abs(yp[-1] - yt[-1]))

        per_state.append(
            {
                "name": name,
                "rmse": rmse,
                "nrmse": nrmse,
                "nmse": nmse,
                "r2": r2,
                "max_abs_err": max_abs_err,
                "final_abs_err": final_abs_err,
            }
        )
        rmse_list.append(rmse)
        nrmse_list.append(nrmse)
        nmse_list.append(nmse)
        r2_list.append(r2)
        max_abs_list.append(max_abs_err)

    overall = {
        "rmse_mean": float(np.nanmean(rmse_list)),
        "nrmse_mean": float(np.nanmean(nrmse_list)),
        "nmse_mean": float(np.nanmean(nmse_list)),
        "r2_mean": float(np.nanmean(r2_list)),
        "max_abs_err_max": float(np.nanmax(max_abs_list)),
    }

    return {
        "overall": overall,
        "per_state": per_state,
    }
