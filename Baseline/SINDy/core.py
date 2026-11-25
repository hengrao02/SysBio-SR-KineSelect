"""Polynomial/STLSQ SINDy, independent of the ODEFormer runtime.

Upstream projects: https://github.com/dynamicslab/pysindy and
https://github.com/sdascoli/odeformer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from importlib.metadata import version
from itertools import product
import math
import time
from typing import Any
import warnings

import numpy as np


@dataclass(frozen=True)
class SINDyConfig:
    polynomial_degree: int = 5
    functions: tuple[str, ...] = ()
    optimizer_threshold: float = 0.1
    optimizer_alpha: float = 0.05
    optimizer_max_iter: int = 100
    finite_difference_order: int = 2
    smoother_window_length: int | None = None
    state_normalization: str = "none"
    grid_search_polynomial_degree: bool = True
    grid_search_functions: bool = False
    optimize_hyperparams: bool = True
    hyper_opt_eval_fraction: float = 0.3
    sorting_metric: str = "r2"
    grid_search_n_jobs: int = 1
    ode_integrator: str = "solve_ivp"
    rollout_rtol: float = 1e-3
    rollout_atol: float = 1e-6
    rollout_timeout: float | None = 1.0

    def __post_init__(self):
        if not isinstance(self.functions, (list, tuple)) or self.functions:
            raise ValueError("SINDy currently supports functions: [] only")
        object.__setattr__(self, "functions", tuple(self.functions))
        for key in (
            "polynomial_degree",
            "optimizer_max_iter",
            "finite_difference_order",
            "grid_search_n_jobs",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in (
            "grid_search_polynomial_degree",
            "grid_search_functions",
            "optimize_hyperparams",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.grid_search_functions or self.state_normalization != "none":
            raise ValueError(
                "SINDy supports polynomial search with state_normalization: none"
            )
        if self.sorting_metric != "r2" or self.ode_integrator != "solve_ivp":
            raise ValueError(
                "SINDy requires sorting_metric: r2 and ode_integrator: solve_ivp"
            )
        window = self.smoother_window_length
        if window is not None and (
            type(window) is not int or window < 5 or window % 2 != 1
        ):
            raise ValueError(
                "smoother_window_length must be null or an odd integer >= 5"
            )
        for key in (
            "optimizer_threshold",
            "optimizer_alpha",
            "hyper_opt_eval_fraction",
            "rollout_rtol",
            "rollout_atol",
            "rollout_timeout",
        ):
            value = getattr(self, key)
            if key == "rollout_timeout" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
        if min(self.optimizer_threshold, self.optimizer_alpha) < 0:
            raise ValueError("STLSQ threshold and alpha must be nonnegative")
        if not 0 < self.hyper_opt_eval_fraction < 1:
            raise ValueError("hyper_opt_eval_fraction must be between zero and one")
        if min(self.rollout_rtol, self.rollout_atol) <= 0:
            raise ValueError("rollout tolerances must be positive")
        if self.rollout_timeout is not None and self.rollout_timeout <= 0:
            raise ValueError("rollout_timeout must be null or positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["functions"] = list(self.functions)
        return result

    def initial_parameters(self) -> dict[str, Any]:
        """Numerical component settings established before the declared search."""
        return {
            key: getattr(self, key)
            for key in (
                "polynomial_degree",
                "optimizer_threshold",
                "optimizer_alpha",
                "optimizer_max_iter",
                "finite_difference_order",
                "smoother_window_length",
            )
        }

    def parameter_grid(self) -> list[dict[str, Any]]:
        if not self.optimize_hyperparams:
            return [self.initial_parameters()]
        grid = {
            "optimizer_threshold": sorted({0.05, 0.1, 0.15, self.optimizer_threshold}),
            "optimizer_alpha": sorted({0.025, 0.05, 0.075, self.optimizer_alpha}),
            "optimizer_max_iter": sorted({20, 100, self.optimizer_max_iter}),
            "finite_difference_order": sorted({2, 3, 4, self.finite_difference_order}),
            "smoother_window_length": [
                None,
                *sorted(
                    {15}
                    | (
                        {self.smoother_window_length}
                        if self.smoother_window_length is not None
                        else set()
                    )
                ),
            ],
            "polynomial_degree": (
                list(range(1, self.polynomial_degree + 1))
                if self.grid_search_polynomial_degree
                else [self.polynomial_degree]
            ),
        }
        keys = sorted(grid)
        return [
            dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))
        ]


def _require_pysindy():
    try:
        import pysindy
    except ImportError as exc:
        raise ImportError(
            "SINDy requires its optional dependencies: python -m pip install '.[sindy]'"
        ) from exc
    if pysindy.__version__ != "2.1.0":
        raise ImportError(
            f"SINDy requires pysindy==2.1.0, found {pysindy.__version__}; install '.[sindy]'"
        )
    return pysindy


def _estimator(params):
    ps = _require_pysindy()
    diff = ps.FiniteDifference(order=params["finite_difference_order"])
    if params["smoother_window_length"] is not None:
        diff = ps.SmoothedFiniteDifference(
            order=params["finite_difference_order"],
            save_smooth=False,
            smoother_kws={
                "window_length": params["smoother_window_length"],
                "polyorder": 3,
                "axis": 0,
            },
        )
    return ps.SINDy(
        feature_library=ps.PolynomialLibrary(
            degree=params["polynomial_degree"],
            include_bias=True,
            include_interaction=True,
        ),
        optimizer=ps.STLSQ(
            threshold=params["optimizer_threshold"],
            alpha=params["optimizer_alpha"],
            max_iter=params["optimizer_max_iter"],
            normalize_columns=False,
            unbias=True,
        ),
        differentiation_method=diff,
    )


def _trial(candidate_id, params, times, states, split, effective_parameters):
    from sklearn.metrics import r2_score

    record = {
        "candidate_id": candidate_id,
        "parameters": params,
        "effective_parameters": effective_parameters,
        "status": "failed",
        "validation_derivative_r2": None,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            estimator = _estimator(effective_parameters)
            estimator.fit(states[:split], t=times[:split])
            score = float(
                estimator.score(
                    states[split:],
                    t=times[split:],
                    metric=r2_score,
                    multioutput="variance_weighted",
                )
            )
            if not np.isfinite(score):
                raise ValueError("validation derivative R2 is non-finite")
            record.update(status="success", validation_derivative_r2=score)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["warnings"] = sorted({str(w.message) for w in caught})
    return record


def _equation_coefficients(coefficients):
    """Return numeric coefficients of PySINDy's equations(precision=3).

    Match both its np.round zero-term filter and formatting of the original
    float, so saved coefficients and equation-based prediction agree.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    return np.array(
        [
            0.0 if np.round(value, 3) == 0 else float(format(float(value), ".3f"))
            for value in coefficients.flat
        ]
    ).reshape(coefficients.shape)


class SINDy:
    def __init__(
        self, config: SINDyConfig | None = None, *, state_names: list[str] | None = None
    ):
        self.config = config or SINDyConfig()
        self.state_names = None if state_names is None else list(state_names)
        self.reset()

    def reset(self):
        self.model = None
        self.search_results = []
        self.search_summary = {}
        self._coefficients = None
        self._powers = None

    def fit(self, times, states) -> dict[str, Any]:
        self.reset()
        times = np.asarray(times, dtype=float)
        states = np.asarray(states, dtype=float)
        if (
            times.ndim != 1
            or states.ndim != 2
            or states.shape[0] != len(times)
            or states.shape[1] < 1
            or len(times) < 3
        ):
            raise ValueError(
                "SINDy requires times [T] and states [T,D], T >= 3 and D >= 1"
            )
        if (
            not np.isfinite(times).all()
            or not np.isfinite(states).all()
            or not np.all(np.diff(times) > 0)
        ):
            raise ValueError(
                "SINDy requires finite inputs and strictly increasing times"
            )
        names = (
            self.state_names
            if self.state_names is not None
            else [f"x_{i}" for i in range(states.shape[1])]
        )
        if (
            len(names) != states.shape[1]
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n for n in names)
        ):
            raise ValueError(
                "state_names must contain one unique nonempty name per state"
            )
        _require_pysindy()
        grid = self.config.parameter_grid()
        selected = grid[0]
        # Preserve the reference wrapper's parameter application: grid updates
        # change declared attributes, while numerical components retain their
        # initialization settings through candidate fitting and final refit.
        effective = self.config.initial_parameters()
        self.search_summary = {
            "selection_metric": "validation_derivative_r2_variance_weighted",
            "num_candidates": len(grid),
            "validation_derivative_r2": None,
        }
        if self.config.optimize_hyperparams:
            split = math.floor((1 - self.config.hyper_opt_eval_fraction) * len(times))
            if split < 3 or len(times) - split < 2:
                raise ValueError(
                    "SINDy needs >= 3 training and >= 2 validation samples"
                )
            self.search_summary.update(
                training_samples=split, validation_samples=len(times) - split
            )
            if self.config.grid_search_n_jobs == 1:
                records = [
                    _trial(i, params, times, states, split, effective)
                    for i, params in enumerate(grid)
                ]
            else:
                from joblib import Parallel, delayed, parallel_config

                with parallel_config(backend="loky", inner_max_num_threads=1):
                    records = Parallel(n_jobs=self.config.grid_search_n_jobs)(
                        delayed(_trial)(i, params, times, states, split, effective)
                        for i, params in enumerate(grid)
                    )
            self.search_results = records
            valid = [r for r in records if r["status"] == "success"]
            self.search_summary["num_valid_candidates"] = len(valid)
            if not valid:
                raise ValueError(
                    "SINDy search has no valid candidate; inspect search_results.json"
                )
            best = max(valid, key=lambda r: r["validation_derivative_r2"])
            selected = best["parameters"]
            self.search_summary.update(
                selected_candidate_id=best["candidate_id"],
                validation_derivative_r2=best["validation_derivative_r2"],
            )
        self.search_summary["selected_grid_parameters"] = selected
        self.search_summary["selected_parameters"] = effective
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator = _estimator(effective)
            estimator.fit(states, t=times)
        coefficients = np.asarray(estimator.coefficients(), dtype=float)
        if not np.isfinite(coefficients).all():
            raise ValueError(
                "SINDy full-trajectory refit produced non-finite coefficients"
            )
        coefficients = _equation_coefficients(coefficients)
        self.search_summary["refit_warnings"] = sorted({str(w.message) for w in caught})
        self.search_summary["refit_samples"] = len(times)
        model = {
            "schema_version": 1,
            "method": "sindy",
            "state_names": names,
            "powers": estimator.feature_library.powers_.tolist(),
            "coefficients": coefficients.tolist(),
            "selected_parameters": effective,
            "selected_grid_parameters": selected,
            "config": self.config.to_dict(),
            "model_format": "sysbiosr-model-v2",
            "dependency_versions": {
                name: version(name)
                for name in ("pysindy", "numpy", "scipy", "scikit-learn")
            },
        }
        self.model = copy.deepcopy(model)
        self._coefficients = coefficients.copy()
        self._powers = np.asarray(model["powers"], dtype=int)
        return copy.deepcopy(model)

    @classmethod
    def from_model(cls, model: dict[str, Any]):
        if model.get("schema_version") != 1 or model.get("method") != "sindy":
            raise ValueError("unsupported SINDy model schema")
        config = SINDyConfig(**model["config"])
        if model.get("model_format") != "sysbiosr-model-v2":
            raise ValueError(
                "unsupported model format. Expected model_format='sysbiosr-model-v2'."
            )
        names = model["state_names"]
        powers = np.asarray(model["powers"], dtype=float)
        coefficients = np.asarray(model["coefficients"], dtype=float)
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("invalid SINDy state names")
        if (
            powers.ndim != 2
            or powers.shape[1] != len(names)
            or len(powers) == 0
            or coefficients.shape != (len(names), len(powers))
        ):
            raise ValueError("invalid SINDy coefficient/power dimensions")
        if (
            not np.isfinite(powers).all()
            or np.any(powers < 0)
            or np.any(powers != np.floor(powers))
            or np.any(powers.sum(axis=1) > config.polynomial_degree)
        ):
            raise ValueError("invalid SINDy polynomial powers")
        if not np.isfinite(coefficients).all() or len(np.unique(powers, axis=0)) != len(
            powers
        ):
            raise ValueError("invalid SINDy coefficients or duplicate powers")
        # These scientific records also prevent reusing earlier search results
        # with the same config/version but different parameter application.
        declared = model.get("selected_grid_parameters")
        if not isinstance(declared, dict) or declared not in config.parameter_grid():
            raise ValueError(
                "missing or invalid grid parameter record; rerun SINDy fitting"
            )
        if model.get("selected_parameters") != config.initial_parameters():
            raise ValueError("SINDy effective parameters mismatch; rerun fitting")
        if not np.array_equal(coefficients, _equation_coefficients(coefficients)):
            raise ValueError(
                "SINDy coefficients must match three-decimal equations; rerun fitting"
            )
        result = cls(config, state_names=names)
        result.model = copy.deepcopy(model)
        result._powers = powers.astype(int)
        result._coefficients = coefficients.copy()
        return result

    def equation(self) -> str:
        if self.model is None:
            raise RuntimeError("SINDy has no fitted model")
        equations = []
        for i, coefficients in enumerate(self._coefficients):
            terms = []
            for coefficient, powers in zip(coefficients, self._powers):
                if coefficient == 0:
                    continue
                factors = [
                    f"x_{j}" if p == 1 else f"x_{j}**{p}"
                    for j, p in enumerate(powers)
                    if p
                ]
                terms.append(" * ".join([format(float(coefficient), ".3f"), *factors]))
            equations.append(f"dx_{i}/dt = " + (" + ".join(terms) if terms else "0"))
        return " | ".join(equations)

    def rhs(self, state) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SINDy has no fitted model")
        state = np.asarray(state, dtype=float)
        if (
            state.shape != (self._coefficients.shape[0],)
            or not np.isfinite(state).all()
        ):
            raise ValueError(
                "SINDy RHS requires a finite state vector of the fitted dimension"
            )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = self._coefficients @ np.prod(
                state[None, :] ** self._powers, axis=1
            )
        if not np.isfinite(result).all():
            raise ValueError("SINDy RHS is non-finite")
        return result

    def predict(self, times, y0) -> np.ndarray:
        from scipy.integrate import solve_ivp

        if self.model is None:
            raise RuntimeError("SINDy has no fitted model")
        times = np.asarray(times, dtype=float)
        y0 = np.asarray(y0, dtype=float)
        if (
            times.ndim != 1
            or not len(times)
            or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)
        ):
            raise ValueError("prediction times must be a finite increasing vector")
        self.rhs(y0)
        if len(times) == 1:
            return y0[None, :].copy()
        deadline = (
            None
            if self.config.rollout_timeout is None
            else time.monotonic() + self.config.rollout_timeout
        )

        def rhs(t, state):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("SINDy rollout exceeded rollout_timeout")
            return self.rhs(state)

        solution = solve_ivp(
            rhs,
            (times[0], times[-1]),
            y0,
            t_eval=times,
            method="RK45",
            rtol=self.config.rollout_rtol,
            atol=self.config.rollout_atol,
        )
        if (
            not solution.success
            or solution.y.shape != (len(y0), len(times))
            or not np.isfinite(solution.y).all()
        ):
            raise RuntimeError(f"SINDy rollout failed: {solution.message}")
        return solution.y.T
