"""SINDy-PI for rational ODEs under the shared benchmark protocol.

The original MATLAB repository is a scientific reference, not a dependency.
Scientific reference: https://github.com/dynamicslab/SINDy-PI.
Rollout adapter follows the existing ODEFormer/SysBioSR integration pattern.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
import copy
import math
import time
import numpy as np


@dataclass(frozen=True)
class SINDyPIConfig:
    polynomial_degree: int = 2
    implicit_denominator_degree: int = 1
    include_constant: bool = True
    threshold: float = 0.05
    threshold_grid: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.2)
    stlsq_iterations: int = 20
    normalize_library: bool = True
    optimize_hyperparams: bool = True
    hyper_opt_eval_fraction: float = 0.3
    sparsity_selection_tolerance: float = 0.05
    min_denominator_abs: float = 1e-8
    max_lhs_candidates: int | None = None
    finite_difference_order: int = 2
    smoother_window_length: int | None = None
    drop_boundary_points: bool = True
    ode_integrator: str = "solve_ivp"
    rollout_rtol: float = 1e-3
    rollout_atol: float = 1e-6
    rollout_timeout: float | None = 1.0

    def __post_init__(self):
        for key in ("polynomial_degree", "implicit_denominator_degree"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if type(self.stlsq_iterations) is not int or self.stlsq_iterations < 1:
            raise ValueError("stlsq_iterations must be a positive integer")
        for key in (
            "include_constant",
            "normalize_library",
            "optimize_hyperparams",
            "drop_boundary_points",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.polynomial_degree == 0 and not self.include_constant:
            raise ValueError("the numerator library must not be empty")
        if type(
            self.finite_difference_order
        ) is not int or self.finite_difference_order not in (1, 2):
            raise ValueError(
                "finite_difference_order must be 1 or 2 (numpy.gradient edge order)"
            )
        if self.max_lhs_candidates is not None and (
            type(self.max_lhs_candidates) is not int or self.max_lhs_candidates < 1
        ):
            raise ValueError("max_lhs_candidates must be null or positive integer")
        window = self.smoother_window_length
        if window is not None and (
            type(window) is not int or window < 3 or window % 2 == 0
        ):
            raise ValueError(
                "smoother_window_length must be null or an odd integer >= 3"
            )
        if (
            not isinstance(self.threshold_grid, (list, tuple))
            or not self.threshold_grid
        ):
            raise ValueError("threshold_grid must be a nonempty list")
        for value in (
            self.threshold,
            *self.threshold_grid,
            self.sparsity_selection_tolerance,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    "thresholds and sparsity_selection_tolerance must be finite and nonnegative"
                )
        if len(set(self.threshold_grid)) != len(self.threshold_grid):
            raise ValueError("threshold_grid must be unique")
        object.__setattr__(self, "threshold_grid", tuple(self.threshold_grid))
        for key in (
            "hyper_opt_eval_fraction",
            "min_denominator_abs",
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
                or value <= 0
            ):
                raise ValueError(f"{key} must be positive and finite")
        if self.hyper_opt_eval_fraction >= 1:
            raise ValueError("hyper_opt_eval_fraction must be between 0 and 1")
        if self.ode_integrator != "solve_ivp":
            raise ValueError("SINDy-PI uses solve_ivp")

    def to_dict(self):
        data = asdict(self)
        data["threshold_grid"] = list(self.threshold_grid)
        return data


def polynomial_powers(dimension, degree, include_constant=True):
    powers = [[0] * dimension] if include_constant else []
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(dimension), order):
            row = [0] * dimension
            for i in indices:
                row[i] += 1
            powers.append(row)
    return powers


def evaluate_library(states, powers):
    x = np.asarray(states, dtype=float)
    values = np.ones((len(x), len(powers)), dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        for j, row in enumerate(powers):
            for i, exponent in enumerate(row):
                if exponent:
                    values[:, j] *= x[:, i] ** exponent
    if not np.isfinite(values).all():
        raise ValueError("nonfinite polynomial library")
    return values


class SINDyPIModel:
    """Full-precision rational RHS; expression text is never parsed for rollout."""

    def __init__(self, numerator, denominator, state_names, config):
        if (
            not isinstance(state_names, list)
            or not state_names
            or any(not isinstance(s, str) or not s for s in state_names)
            or len(set(state_names)) != len(state_names)
        ):
            raise ValueError("state_names must be unique nonempty strings")
        self.state_names = list(state_names)
        self.config = config
        dim = len(state_names)
        self.numerator_powers = polynomial_powers(
            dim, config.polynomial_degree, config.include_constant
        )
        self.denominator_powers = polynomial_powers(
            dim, config.implicit_denominator_degree
        )
        self.numerator = np.asarray(numerator, dtype=float).copy()
        self.denominator = np.asarray(denominator, dtype=float).copy()
        if self.numerator.shape != (
            dim,
            len(self.numerator_powers),
        ) or self.denominator.shape != (dim, len(self.denominator_powers)):
            raise ValueError(
                "SINDy-PI coefficient dimensions do not match libraries/states"
            )
        if (
            not np.isfinite(self.numerator).all()
            or not np.isfinite(self.denominator).all()
        ):
            raise ValueError("SINDy-PI coefficients must be finite")
        if np.any(np.all(self.denominator == 0, axis=1)):
            raise ValueError("SINDy-PI denominator polynomial is zero")

    def rhs(self, state):
        x = np.asarray(state, dtype=float)
        if x.shape != (len(self.state_names),) or not np.isfinite(x).all():
            raise ValueError("invalid state vector")
        numerator = (
            self.numerator @ evaluate_library(x[None, :], self.numerator_powers)[0]
        )
        denominator = (
            self.denominator @ evaluate_library(x[None, :], self.denominator_powers)[0]
        )
        if not np.isfinite(denominator).all() or np.any(
            np.abs(denominator) < self.config.min_denominator_abs
        ):
            raise ValueError("SINDy-PI denominator below min_denominator_abs")
        result = numerator / denominator
        if not np.isfinite(result).all():
            raise ValueError("nonfinite SINDy-PI RHS")
        return result

    def equation(self):
        def expression(coefficients, powers):
            terms = []
            for value, row in zip(coefficients, powers):
                if value == 0:
                    continue
                factors = [
                    f"x_{i}" if power == 1 else f"x_{i}**{power}"
                    for i, power in enumerate(row)
                    if power
                ]
                terms.append(
                    f"({repr(float(value))})"
                    + ("*" + "*".join(factors) if factors else "")
                )
            return " + ".join(terms) or "0"

        return " | ".join(
            f"({expression(n,self.numerator_powers)})/({expression(d,self.denominator_powers)})"
            for n, d in zip(self.numerator, self.denominator)
        )

    def predict(self, times, y0):
        from scipy.integrate import solve_ivp

        t = np.asarray(times, dtype=float)
        y0 = np.asarray(y0, dtype=float)
        if (
            t.ndim != 1
            or not len(t)
            or not np.isfinite(t).all()
            or not np.all(np.diff(t) > 0)
        ):
            raise ValueError("prediction times must be finite and strictly increasing")
        deadline = (
            None
            if self.config.rollout_timeout is None
            else time.monotonic() + self.config.rollout_timeout
        )

        def rhs(_, state):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("SINDy-PI rollout exceeded rollout_timeout")
            return self.rhs(state)

        rhs(t[0], y0)
        if len(t) == 1:
            return y0[None, :].copy()
        solution = solve_ivp(
            rhs,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="RK45",
            rtol=self.config.rollout_rtol,
            atol=self.config.rollout_atol,
        )
        if (
            not solution.success
            or solution.y.shape != (len(y0), len(t))
            or not np.isfinite(solution.y).all()
        ):
            raise RuntimeError(f"SINDy-PI rollout failed: {solution.message}")
        return solution.y.T

    @classmethod
    def from_model(cls, model):
        if (
            not isinstance(model, dict)
            or type(model.get("schema_version")) is not int
            or model["schema_version"] != 1
            or model.get("method") != "sindy_pi"
        ):
            raise ValueError("unsupported SINDy-PI model schema")
        config = SINDyPIConfig(**model["config"])
        if model.get("model_format") != "sysbiosr-model-v2":
            raise ValueError(
                "unsupported model format. Expected model_format='sysbiosr-model-v2'."
            )
        result = cls(
            model["numerator_coefficients"],
            model["denominator_coefficients"],
            model["state_names"],
            config,
        )
        for key, expected in (
            ("numerator_powers", result.numerator_powers),
            ("denominator_powers", result.denominator_powers),
        ):
            raw = model[key]
            if (
                not isinstance(raw, list)
                or any(
                    not isinstance(row, list) or any(type(v) is not int for v in row)
                    for row in raw
                )
                or raw != expected
            ):
                raise ValueError(f"invalid {key}")
        return result


class SINDyPI:
    def __init__(self, config=None, *, state_names=None):
        self.config = config or SINDyPIConfig()
        self.state_names = state_names
        self.reset()

    def reset(self):
        self.model = None
        self.predictor = None
        self.search_results = []
        self.search_summary = {}

    def _differentiate(self, t, x):
        if len(t) < 3:
            raise ValueError("each differentiation segment needs at least 3 samples")
        x = np.asarray(x, dtype=float).copy()
        if self.config.smoother_window_length is not None:
            from scipy.signal import savgol_filter

            window = self.config.smoother_window_length
            if window > len(t):
                raise ValueError("smoothing window exceeds segment length")
            x = savgol_filter(
                x, window_length=window, polyorder=min(3, window - 1), axis=0
            )
        dx = np.gradient(x, t, axis=0, edge_order=self.config.finite_difference_order)
        if self.config.drop_boundary_points:
            x, dx = x[1:-1], dx[1:-1]
        if not np.isfinite(x).all() or not np.isfinite(dx).all():
            raise ValueError("nonfinite differentiated trajectory")
        return x, dx

    def _sparsify(self, design, target, threshold):
        if not np.isfinite(design).all() or not np.isfinite(target).all():
            raise ValueError("nonfinite regression inputs")
        scales = np.ones(design.shape[1])
        if self.config.normalize_library:
            scales = np.linalg.norm(design, axis=0)
            if not np.isfinite(scales).all():
                raise ValueError("nonfinite library norms")
            scales = np.where(scales <= 1e-14, 1.0, scales)
        normalized = design / scales
        coefficients = np.linalg.lstsq(normalized, target, rcond=None)[0]
        for _ in range(self.config.stlsq_iterations):
            small = np.abs(coefficients) < threshold
            coefficients[small] = 0.0
            if np.all(small):
                break
            coefficients[~small] = np.linalg.lstsq(
                normalized[:, ~small], target, rcond=None
            )[0]
        coefficients = coefficients / scales
        if not np.isfinite(coefficients).all():
            raise ValueError("nonfinite STLSQ coefficients")
        return coefficients

    def _fit_coefficients(self, phi, psi, dx, lhs, threshold):
        mask = np.arange(psi.shape[1]) != lhs
        dx_features = dx[:, None] * psi
        design = np.column_stack([phi, dx_features[:, mask]])
        coefficients = self._sparsify(design, dx_features[:, lhs], threshold)
        numerator = coefficients[: phi.shape[1]]
        denominator = np.zeros(psi.shape[1])
        denominator[lhs] = 1.0
        denominator[mask] = -coefficients[phi.shape[1] :]
        return numerator, denominator, int(np.count_nonzero(coefficients)) + 1

    def _score(self, numerator, denominator, phi, psi, dx):
        n = phi @ numerator
        d = psi @ denominator
        if (
            not np.isfinite(n).all()
            or not np.isfinite(d).all()
            or np.any(np.abs(d) < self.config.min_denominator_abs)
        ):
            raise ValueError("invalid denominator or numerator on validation samples")
        pred = n / d
        if not np.isfinite(pred).all():
            raise ValueError("nonfinite validation derivative")
        ss_res = float(np.sum((dx - pred) ** 2))
        mse = ss_res / len(dx)
        if not math.isfinite(mse):
            raise ValueError("nonfinite validation MSE")
        ss_tot = float(np.sum((dx - np.mean(dx)) ** 2))
        r2 = (
            (1.0 if ss_res <= 1e-14 else 0.0)
            if ss_tot <= 1e-14
            else 1.0 - ss_res / ss_tot
        )
        return mse, r2 if math.isfinite(r2) else None

    def fit(self, times, states):
        self.reset()
        self.search_summary = {
            "status": "running",
            "stage": "prepare",
            "state_selections": [],
        }
        try:
            return self._fit(times, states)
        except Exception as exc:
            self.search_summary.update(
                status="failed", error=f"{type(exc).__name__}: {exc}"
            )
            self.model = None
            self.predictor = None
            raise

    def _fit(self, times, states):
        t = np.asarray(times, dtype=float)
        x = np.asarray(states, dtype=float)
        if (
            t.ndim != 1
            or x.ndim != 2
            or len(t) != len(x)
            or len(t) < 3
            or x.shape[1] < 1
            or not np.isfinite(t).all()
            or not np.isfinite(x).all()
            or not np.all(np.diff(t) > 0)
        ):
            raise ValueError("expected finite increasing times [T] and states [T,D]")
        names = (
            [f"x_{i}" for i in range(x.shape[1])]
            if self.state_names is None
            else list(self.state_names)
        )
        if (
            len(names) != x.shape[1]
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("invalid state_names")
        split = int(round(len(t) * (1 - self.config.hyper_opt_eval_fraction)))
        if min(split, len(t) - split) < 3:
            raise ValueError(
                "insufficient samples for separate train/validation differentiation"
            )
        train, dtrain = self._differentiate(t[:split], x[:split])
        val, dval = self._differentiate(t[split:], x[split:])
        npowers = polynomial_powers(
            x.shape[1], self.config.polynomial_degree, self.config.include_constant
        )
        dpowers = polynomial_powers(x.shape[1], self.config.implicit_denominator_degree)
        phi, psi = evaluate_library(train, npowers), evaluate_library(train, dpowers)
        vphi, vpsi = evaluate_library(val, npowers), evaluate_library(val, dpowers)
        self.search_summary.update(
            raw_training_samples=split,
            raw_validation_samples=len(t) - split,
            training_samples=len(train),
            validation_samples=len(val),
        )
        thresholds = (
            self.config.threshold_grid
            if self.config.optimize_hyperparams
            else (self.config.threshold,)
        )
        lhs_indices = range(
            len(dpowers)
            if self.config.max_lhs_candidates is None
            else min(self.config.max_lhs_candidates, len(dpowers))
        )
        self.search_summary["stage"] = "search"
        selected = []
        self.search_summary["state_selections"] = selected
        for state in range(x.shape[1]):
            candidates = []
            for threshold in thresholds:
                for lhs in lhs_indices:
                    record = {
                        "state_index": state,
                        "lhs_index": lhs,
                        "threshold": threshold,
                        "status": "invalid",
                    }
                    try:
                        n, d, active = self._fit_coefficients(
                            phi, psi, dtrain[:, state], lhs, threshold
                        )
                        mse, r2 = self._score(n, d, vphi, vpsi, dval[:, state])
                        record.update(
                            status="success",
                            validation_mse=mse,
                            validation_r2=r2,
                            active_terms=active,
                        )
                        candidates.append(record)
                    except (
                        ValueError,
                        FloatingPointError,
                        np.linalg.LinAlgError,
                    ) as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    self.search_results.append(record)
            if not candidates:
                raise ValueError(f"no valid SINDy-PI candidate for state {state}")
            best = min(c["validation_mse"] for c in candidates)
            eligible = [
                c
                for c in candidates
                if c["validation_mse"]
                <= best * (1 + self.config.sparsity_selection_tolerance)
            ]
            winner = min(
                eligible, key=lambda c: (c["active_terms"], c["validation_mse"])
            )
            selected.append(dict(winner))
        self.search_summary["stage"] = "refit"
        full, dfull = self._differentiate(t, x)
        fphi, fpsi = evaluate_library(full, npowers), evaluate_library(full, dpowers)
        numerators = []
        denominators = []
        for state, selection in enumerate(selected):
            n, d, active = self._fit_coefficients(
                fphi,
                fpsi,
                dfull[:, state],
                selection["lhs_index"],
                selection["threshold"],
            )
            self._score(n, d, fphi, fpsi, dfull[:, state])
            numerators.append(n.tolist())
            denominators.append(d.tolist())
            selection["refit_active_terms"] = active
        predictor = SINDyPIModel(numerators, denominators, names, self.config)
        self.search_summary.update(
            status="success",
            stage="complete",
            refit_samples=len(full),
            state_selections=selected,
            candidates_evaluated=len(self.search_results),
        )
        self.model = {
            "schema_version": 1,
            "method": "sindy_pi",
            "state_names": names,
            "numerator_powers": npowers,
            "denominator_powers": dpowers,
            "numerator_coefficients": numerators,
            "denominator_coefficients": denominators,
            "selection": selected,
            "config": self.config.to_dict(),
            "model_format": "sysbiosr-model-v2",
        }
        self.predictor = predictor
        return copy.deepcopy(self.model)

    def equation(self):
        if self.predictor is None:
            raise RuntimeError("SINDy-PI is not fitted")
        return self.predictor.equation()

    def predict(self, times, y0):
        if self.predictor is None:
            raise RuntimeError("SINDy-PI is not fitted")
        return self.predictor.predict(times, y0)

    @staticmethod
    def from_model(model):
        return SINDyPIModel.from_model(model)
