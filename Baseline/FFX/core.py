"""Benchmark-profile FFX orchestration and portable numeric models.

Upstream solver: https://github.com/ffx-org/ffx.
Worker and rollout interfaces follow the shared SysBioSR adapters.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np


@dataclass(frozen=True)
class FFXConfig:
    finite_difference_order: int = 2
    smoother_window_length: int | None = None
    optimize_hyperparams: bool = True
    hyper_opt_eval_fraction: float = 0.3
    sorting_metric: str = "r2"
    ode_integrator: str = "solve_ivp"
    rollout_rtol: float = 1e-3
    rollout_atol: float = 1e-6
    rollout_timeout: float | None = 1.0
    fit_timeout_seconds: float | None = None

    def __post_init__(self):
        if (
            type(self.finite_difference_order) is not int
            or self.finite_difference_order < 1
        ):
            raise ValueError("finite_difference_order must be a positive integer")
        w = self.smoother_window_length
        if w is not None and (type(w) is not int or w < 5 or w % 2 != 1):
            raise ValueError(
                "smoother_window_length must be null or an odd integer >= 5"
            )
        if type(self.optimize_hyperparams) is not bool:
            raise ValueError("optimize_hyperparams must be boolean")
        if self.sorting_metric != "r2" or self.ode_integrator != "solve_ivp":
            raise ValueError("this FFX profile uses rollout r2 and solve_ivp")
        for k in (
            "hyper_opt_eval_fraction",
            "rollout_rtol",
            "rollout_atol",
            "rollout_timeout",
            "fit_timeout_seconds",
        ):
            v = getattr(self, k)
            if v is None and k in ("rollout_timeout", "fit_timeout_seconds"):
                continue
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
            ):
                raise ValueError(f"{k} must be positive and finite")
        if self.hyper_opt_eval_fraction >= 1:
            raise ValueError("hyper_opt_eval_fraction must be below 1")

    def to_dict(self):
        return asdict(self)


def _finite(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("model numbers must be finite real numbers")
    return float(value)


def validate_basis(basis, dimension):
    if not isinstance(basis, dict):
        raise ValueError("invalid FFX basis")
    kind = basis.get("type")
    if kind == "power":
        i = basis["state"]
        _finite(basis["exponent"])
        if type(i) is not int or not 0 <= i < dimension:
            raise ValueError("invalid state index")
    elif kind in ("abs", "log10"):
        validate_basis(basis["base"], dimension)
    elif kind == "product":
        validate_basis(basis["left"], dimension)
        validate_basis(basis["right"], dimension)
    else:
        raise ValueError(f"unsupported FFX basis: {kind}")


def basis_value(basis, x):
    kind = basis["type"]
    if kind == "power":
        return x[basis["state"]] ** basis["exponent"]
    if kind == "product":
        return basis_value(basis["left"], x) * basis_value(basis["right"], x)
    value = basis_value(basis["base"], x)
    if kind == "abs":
        return abs(value)
    if value <= 0:
        raise ValueError("FFX log10 requires positive input")
    return np.log10(value)


def basis_expression(basis):
    kind = basis["type"]
    if kind == "power":
        name = f"x_{basis['state']}"
        return (
            name
            if basis["exponent"] == 1
            else f"({name})**({repr(float(basis['exponent']))})"
        )
    if kind == "product":
        return (
            f"({basis_expression(basis['left'])})*({basis_expression(basis['right'])})"
        )
    return f"{kind}({basis_expression(basis['base'])})"


class FFXModel:
    """Frozen numerical components, independent of FFX/PySINDy installations."""

    def __init__(self, components, state_names, config=None):
        if (
            not isinstance(state_names, list)
            or not state_names
            or any(not isinstance(n, str) or not n for n in state_names)
            or len(set(state_names)) != len(state_names)
        ):
            raise ValueError("invalid state_names")
        if not isinstance(components, list) or len(components) != len(state_names):
            raise ValueError("component count mismatch")
        for component in components:
            _finite(component["offset"])
            for side in ("numerator", "denominator"):
                if not isinstance(component[side], list):
                    raise ValueError("invalid FFX terms")
                for term in component[side]:
                    _finite(term["coefficient"])
                    validate_basis(term["basis"], len(state_names))
        self.components = copy.deepcopy(components)
        self.state_names = list(state_names)
        self.config = config or FFXConfig()

    def equation(self):
        expressions = []
        for c in self.components:

            def terms(side):
                return [
                    f"({repr(float(t['coefficient']))})*({basis_expression(t['basis'])})"
                    for t in c[side]
                ]

            n = " + ".join([repr(float(c["offset"]))] + terms("numerator"))
            expressions.append(
                f"({n})/(1.0 + {' + '.join(terms('denominator'))})"
                if c["denominator"]
                else n
            )
        return " | ".join(expressions)

    def rhs(self, state):
        x = np.asarray(state, dtype=float)
        if x.shape != (len(self.state_names),) or not np.isfinite(x).all():
            raise ValueError("invalid state vector")
        values = []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for c in self.components:
                n = float(c["offset"])
                d = 1.0
                for t in c["numerator"]:
                    n += t["coefficient"] * basis_value(t["basis"], x)
                for t in c["denominator"]:
                    d += t["coefficient"] * basis_value(t["basis"], x)
                if not np.isfinite(d) or d == 0:
                    raise ValueError("invalid FFX denominator")
                values.append(n / d)
        result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite FFX RHS")
        return result

    def predict(self, times, y0):
        from scipy.integrate import solve_ivp

        t = np.asarray(times, dtype=float)
        initial = np.asarray(y0, dtype=float)
        if (
            t.ndim != 1
            or not len(t)
            or not np.isfinite(t).all()
            or not np.all(np.diff(t) > 0)
        ):
            raise ValueError("prediction times must be finite and increasing")
        deadline = (
            None
            if self.config.rollout_timeout is None
            else time.monotonic() + self.config.rollout_timeout
        )

        def rhs(_, x):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("FFX rollout timeout")
            return self.rhs(x)

        rhs(t[0], initial)
        if len(t) == 1:
            return initial[None, :].copy()
        sol = solve_ivp(
            rhs,
            (t[0], t[-1]),
            initial,
            t_eval=t,
            rtol=self.config.rollout_rtol,
            atol=self.config.rollout_atol,
            method="RK45",
        )
        if (
            not sol.success
            or sol.y.shape != (len(initial), len(t))
            or not np.isfinite(sol.y).all()
        ):
            raise RuntimeError(f"FFX rollout failed: {sol.message}")
        return sol.y.T

    @classmethod
    def from_model(cls, model):
        if (
            not isinstance(model, dict)
            or type(model.get("schema_version")) is not int
            or model["schema_version"] != 1
            or model.get("method") != "ffx"
        ):
            raise ValueError("unsupported FFX model schema")
        cfg = FFXConfig(**model["config"])
        if model.get("model_format") != "sysbiosr-model-v2":
            raise ValueError(
                "unsupported model format. Expected model_format='sysbiosr-model-v2'."
            )
        return cls(model["components"], model["state_names"], cfg)


class FFX:
    def __init__(self, config=None, *, state_names=None, output_dir=None):
        self.config = config or FFXConfig()
        self.state_names = state_names
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.reset()

    def reset(self):
        self.model = None
        self.predictor = None
        self.component_fronts = []
        self.search_summary = {}
        self.worker_log = None

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()

    def fit(self, times, states):
        self.reset()
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
        FFXModel(
            [
                {"offset": 0.0, "numerator": [], "denominator": []}
                for _ in range(x.shape[1])
            ],
            names,
            self.config,
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sysbiosr-ffx-") as scratch:
            scratch = Path(scratch)
            np.savez(scratch / "inputs.npz", times=t, states=x)
            (scratch / "request.json").write_text(
                json.dumps({"config": self.config.to_dict(), "state_names": names})
            )
            log_path = (
                scratch / "worker.log"
                if self.output_dir is None
                else self.output_dir / "worker.log"
            )
            if self.output_dir is not None:
                self.worker_log = log_path
            env = {
                **os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "Baseline.FFX.worker", str(scratch)],
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=os.name == "posix",
                )
                try:
                    process.wait(timeout=self.config.fit_timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    self._stop(process)
                    self.search_summary = {
                        "status": "timeout",
                        "reason": "fit_timeout_seconds",
                    }
                    raise TimeoutError(
                        "FFX worker exceeded fit_timeout_seconds"
                    ) from exc
                except BaseException:
                    self._stop(process)
                    raise
            reply_path = scratch / "reply.json"
            if not reply_path.is_file():
                self.search_summary = {
                    "status": "failed",
                    "worker_exit_code": process.returncode,
                }
                raise RuntimeError(
                    f"FFX worker exited {process.returncode}: {log_path.read_text(errors='replace')[-3000:]}"
                )
            reply = json.loads(reply_path.read_text())
            self.search_summary = reply.get("search_summary", {})
            self.component_fronts = reply.get("component_fronts", [])
            if process.returncode or reply.get("status") != "success":
                error = reply.get("error", "FFX worker failed")
                if reply.get("status") == "timeout":
                    raise TimeoutError(error)
                raise RuntimeError(error)
            self.predictor = FFXModel.from_model(reply["model"])
            self.model = reply["model"]
        return copy.deepcopy(self.model)

    def equation(self):
        if self.predictor is None:
            raise RuntimeError("FFX is not fitted")
        return self.predictor.equation()

    def predict(self, times, y0):
        if self.predictor is None:
            raise RuntimeError("FFX is not fitted")
        return self.predictor.predict(times, y0)

    @staticmethod
    def from_model(model):
        return FFXModel.from_model(model)
