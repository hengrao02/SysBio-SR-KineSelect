from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def load_reference_model(system_dir: str | Path) -> Any:
    """Load a trusted released model, returning fresh parameters on each call.

    Compile from source to avoid writing bytecode into a frozen system directory.
    No shared module cache or singleton is used between calls or dataset copies.
    """
    path = Path(system_dir).resolve() / "model.py"
    module = ModuleType("_sysbiosr_reference_model")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module.Model()


def simulate_reference(
    model: Any,
    t_eval: Any,
    *,
    y0: Any,
    consts: Any = None,
    method: str = "LSODA",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Integrate a reference model; return (time, state) values or raise.

    The caller explicitly chooses the initial condition. Historical replay uses
    trajectory metadata; scored rollouts retain the benchmark's first-row rule.
    """
    from scipy.integrate import solve_ivp

    times = np.asarray(t_eval, dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("t_eval must be a finite 1D array with at least two points")
    if np.any(np.diff(times) <= 0):
        raise ValueError("t_eval must be strictly increasing")
    initial = model.validate_state(y0)
    parameters = model.validate_consts(model.params.consts if consts is None else consts)
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(parameters)):
        raise ValueError("initial state and parameters must be finite")
    result = solve_ivp(
        lambda t, x: model.rhs(t, x, parameters),
        (float(times[0]), float(times[-1])), initial, t_eval=times,
        method=method, rtol=rtol, atol=atol,
    )
    values = np.asarray(result.y).T
    if not result.success or values.shape != (len(times), model.dim):
        raise RuntimeError(f"reference integration failed: {result.message}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("reference integration returned non-finite values")
    return values
