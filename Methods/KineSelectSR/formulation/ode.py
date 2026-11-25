"""ODE right-hand side builder and trajectory rollout."""

from __future__ import annotations

import logging
import signal
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from .rate import compute_rate
from .reaction import FormulationConfig, KineSelectSRModel, validate_model


logger = logging.getLogger(__name__)
_LAST_ROLLOUT_FAILURE_REASON: str | None = None


def build_ode_rhs(
    model: KineSelectSRModel,
    config: FormulationConfig,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build a solve_ivp-compatible ODE right-hand side function."""
    if model.rate_constants is None:
        raise ValueError("model.rate_constants must be set before building ODE RHS")

    D = validate_model(model)

    tol = config.negative_state_tol
    policy = config.negative_state_policy
    _VALID_POLICIES = {"clip_or_fail", "clip", "fail"}
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"Unknown negative_state_policy: '{policy}'. "
            f"Must be one of {_VALID_POLICIES}."
        )

    reactions = model.reactions
    rate_constants = model.rate_constants

    stoich_vectors = [
        np.array(rxn.products, dtype=int) - np.array(rxn.reactants, dtype=int)
        for rxn in reactions
    ]

    active = np.abs(rate_constants) > 1e-14
    active_reactions = [reactions[i] for i in range(len(reactions)) if active[i]]
    active_stoich = [stoich_vectors[i] for i in range(len(reactions)) if active[i]]
    active_ks = rate_constants[active]

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        if policy == "clip_or_fail":
            if np.any(y < -tol):
                return np.full_like(y, np.nan)
            y_safe = np.maximum(y, 0.0)
        elif policy == "clip":
            y_safe = np.maximum(y, 0.0)
        else:  # policy == "fail"
            if np.any(y < 0.0):
                return np.full_like(y, np.nan)
            y_safe = y

        dy = np.zeros(D) if D > 0 else np.array([])

        for rxn, stoich, k in zip(active_reactions, active_stoich, active_ks):
            rate = k * compute_rate(rxn, y_safe)
            dy += stoich * rate

        return dy

    return rhs


class _RolloutTimeout(Exception):
    """Raised when solve_ivp exceeds the configured wall-clock limit."""


def get_last_rollout_failure_reason() -> str | None:
    """Return the most recent rollout failure reason in this process."""
    return _LAST_ROLLOUT_FAILURE_REASON


def _set_last_rollout_failure_reason(reason: str | None) -> None:
    global _LAST_ROLLOUT_FAILURE_REASON
    _LAST_ROLLOUT_FAILURE_REASON = reason


def _failure_array(
    reason: str,
    t: np.ndarray,
    y0: np.ndarray,
    *,
    exc: Exception | None = None,
) -> np.ndarray:
    detail = reason if exc is None else f"{reason}: {exc!r}"
    _set_last_rollout_failure_reason(detail)
    logger.warning(
        "KineSelectSR rollout failed: %s",
        detail,
        exc_info=exc is not None and logger.isEnabledFor(logging.DEBUG),
    )
    return np.full((len(t), len(y0)), np.nan)


def _install_timeout(timeout: float) -> tuple[Any, tuple[float, float]]:
    """Arm a SIGALRM that raises _RolloutTimeout after timeout seconds."""
    def _handler(signum: int, frame: object) -> None:
        raise _RolloutTimeout(f"solve_ivp exceeded {timeout}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    return old_handler, old_timer


def _clear_timeout(timeout_state: tuple[Any, tuple[float, float]]) -> None:
    """Disarm the SIGALRM timer and restore the previous handler/timer."""
    signal.setitimer(signal.ITIMER_REAL, 0)
    old_handler, old_timer = timeout_state
    signal.signal(signal.SIGALRM, old_handler)
    delay, interval = old_timer
    if delay > 0.0 or interval > 0.0:
        signal.setitimer(signal.ITIMER_REAL, delay, interval)


def rollout(
    model: KineSelectSRModel,
    t: np.ndarray,
    y0: np.ndarray,
    config: FormulationConfig,
    *,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    min_step: float = 1e-12,
    timeout: float | None = None,
) -> np.ndarray:
    """Integrate the kinetic ODE over a time grid."""
    t = np.asarray(t, dtype=float).reshape(-1)
    y0 = np.asarray(y0, dtype=float).reshape(-1)

    if t.size < 2:
        raise ValueError("t must have at least 2 time points")

    _set_last_rollout_failure_reason(None)

    try:
        rhs = build_ode_rhs(model, config)
        kwargs = dict(
            t_eval=t, method=method, rtol=rtol, atol=atol,
        )
        if method == "LSODA":
            kwargs["min_step"] = min_step

        if timeout is not None:
            timeout_state = _install_timeout(timeout)
            try:
                sol = solve_ivp(
                    rhs,
                    [float(t[0]), float(t[-1])],
                    y0,
                    **kwargs,
                )
            except _RolloutTimeout as exc:
                return _failure_array("timeout", t, y0, exc=exc)
            finally:
                _clear_timeout(timeout_state)
        else:
            sol = solve_ivp(
                rhs,
                [float(t[0]), float(t[-1])],
                y0,
                **kwargs,
            )

        if sol.status == 0:
            traj = sol.y.T
            if traj.shape == (len(t), len(y0)) and np.all(np.isfinite(traj)):
                return traj
            return _failure_array("nonfinite_or_invalid_shape", t, y0)
        return _failure_array(f"solve_ivp_status_{sol.status}: {sol.message}", t, y0)
    except _RolloutTimeout as exc:
        return _failure_array("timeout", t, y0, exc=exc)
    except Exception as exc:
        return _failure_array("exception", t, y0, exc=exc)

    return _failure_array("unknown", t, y0)
