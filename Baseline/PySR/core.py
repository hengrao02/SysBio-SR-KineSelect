"""Portable PySR equations and an isolated trajectory-first fitting interface.

Upstream projects: https://github.com/MilesCranmer/PySR and
https://github.com/sdascoli/odeformer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np


@dataclass(frozen=True)
class PySRConfig:
    finite_difference_order: int = 2
    smoother_window_length: int | None = None
    niterations: int = 50
    binary_operators: tuple[str, ...] = ("plus", "sub", "mult", "pow", "div")
    unary_operators: tuple[str, ...] = ("neg",)
    loss: str = "loss(x, y) = (x - y)^2"
    populations: int = 31
    population_size: int = 27
    maxsize: int = 30
    ncycles_per_iteration: int = 380
    precision: int = 32
    optimize_hyperparams: bool = True
    hyper_opt_eval_fraction: float = 0.3
    sorting_metric: str = "r2"
    random_seed: int = 0
    deterministic: bool = True
    parallelism: str = "serial"
    ode_integrator: str = "solve_ivp"
    rollout_rtol: float = 1e-3
    rollout_atol: float = 1e-6
    rollout_timeout: float | None = 1.0
    fit_timeout_seconds: float | None = None

    def __post_init__(self):
        for name in (
            "finite_difference_order",
            "niterations",
            "populations",
            "population_size",
            "maxsize",
            "ncycles_per_iteration",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.population_size < 15 or self.maxsize < 7:
            raise ValueError(
                "population_size must be >= 15 and maxsize >= 7 for the PySR defaults"
            )
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be a uint32 integer")
        for name in ("optimize_hyperparams", "deterministic"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not self.deterministic or self.parallelism != "serial":
            raise ValueError(
                "this PySR profile requires deterministic: true and parallelism: serial"
            )
        if type(self.precision) is not int or self.precision not in (32, 64):
            raise ValueError("precision must be 32 or 64")
        for name, allowed in (
            ("binary_operators", {"plus", "sub", "mult", "pow", "div"}),
            ("unary_operators", {"neg"}),
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, (list, tuple))
                or any(not isinstance(v, str) or v not in allowed for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(
                    f"{name} must contain unique supported operators"
                )
            object.__setattr__(self, name, tuple(values))
        if not self.binary_operators:
            raise ValueError("at least one binary operator is required")
        if (
            self.loss != "loss(x, y) = (x - y)^2"
            or self.sorting_metric != "r2"
            or self.ode_integrator != "solve_ivp"
        ):
            raise ValueError(
                "this profile requires derivative MSE, rollout r2 and solve_ivp"
            )
        window = self.smoother_window_length
        if window is not None and (
            type(window) is not int or window < 5 or window % 2 == 0
        ):
            raise ValueError(
                "smoother_window_length must be null or an odd integer >= 5"
            )
        for name in (
            "hyper_opt_eval_fraction",
            "rollout_rtol",
            "rollout_atol",
            "rollout_timeout",
            "fit_timeout_seconds",
        ):
            value = getattr(self, name)
            if value is None and name in ("rollout_timeout", "fit_timeout_seconds"):
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if self.hyper_opt_eval_fraction >= 1:
            raise ValueError("hyper_opt_eval_fraction must be between zero and one")

    def to_dict(self):
        values = asdict(self)
        for name in ("binary_operators", "unary_operators"):
            values[name] = list(values[name])
        return values

    def derivative_candidates(self):
        orders = list(dict.fromkeys([self.finite_difference_order, 2, 3, 4]))
        windows = list(dict.fromkeys([self.smoother_window_length, None, 15]))
        return [(order, window) for order in orders for window in windows]


def validate_inputs(times, states, state_names=None):
    t = np.asarray(times, dtype=float)
    x = np.asarray(states, dtype=float)
    if (
        t.ndim != 1
        or x.ndim != 2
        or x.shape[0] != len(t)
        or len(t) < 3
        or x.shape[1] < 1
    ):
        raise ValueError("PySR requires times [T], states [T,D], T >= 3 and D >= 1")
    if (
        not np.isfinite(t).all()
        or not np.isfinite(x).all()
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("PySR inputs must be finite with strictly increasing times")
    names = (
        [f"x_{i}" for i in range(x.shape[1])]
        if state_names is None
        else list(state_names)
    )
    if (
        len(names) != x.shape[1]
        or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("state_names must contain one unique name per state")
    return t, x, names


class PySRModel:
    """Run the selected expression vector without PySR, Julia or a pickle."""

    def __init__(self, expressions, state_names, config=None):
        try:
            import numexpr as ne
        except ImportError as exc:
            raise ImportError(
                "PySR equation prediction requires numexpr: python -m pip install numexpr"
            ) from exc
        if (
            not isinstance(state_names, list)
            or not state_names
            or any(not isinstance(n, str) or not n for n in state_names)
            or len(set(state_names)) != len(state_names)
        ):
            raise ValueError("invalid PySR state names")
        if (
            not isinstance(expressions, list)
            or len(expressions) != len(state_names)
            or any(not isinstance(e, str) or not e.strip() for e in expressions)
        ):
            raise ValueError("one nonempty RHS expression is required per state")
        self.expressions = list(expressions)
        self.state_names = list(state_names)
        self.config = config or PySRConfig()
        allowed = {f"x_{i}": i for i in range(len(state_names))}
        self._compiled = []
        for expression in expressions:
            compiled = ne.NumExpr(expression)
            if any(name not in allowed for name in compiled.input_names):
                raise ValueError("PySR expression contains an unknown state symbol")
            self._compiled.append(
                (compiled, [allowed[name] for name in compiled.input_names])
            )

    def equation(self):
        return " | ".join(self.expressions)

    def rhs(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (len(self.state_names),) or not np.isfinite(state).all():
            raise ValueError("invalid PySR state vector")
        values = [
            compiled(*[state[i] for i in indices])
            for compiled, indices in self._compiled
        ]
        if any(np.iscomplexobj(v) for v in values):
            raise ValueError("PySR RHS must be real")
        result = np.asarray(values, dtype=float).reshape(-1)
        if result.shape != state.shape or not np.isfinite(result).all():
            raise ValueError("PySR RHS must be finite and match the state dimension")
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
            raise ValueError("prediction times must be finite and strictly increasing")
        deadline = (
            None
            if self.config.rollout_timeout is None
            else time.monotonic() + self.config.rollout_timeout
        )

        def rhs(time_value, state):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("PySR rollout exceeded rollout_timeout")
            return self.rhs(state)

        rhs(t[0], initial)
        if len(t) == 1:
            return initial[None, :].copy()
        solution = solve_ivp(
            rhs,
            (t[0], t[-1]),
            initial,
            t_eval=t,
            method="RK45",
            rtol=self.config.rollout_rtol,
            atol=self.config.rollout_atol,
        )
        if (
            not solution.success
            or solution.y.shape != (len(initial), len(t))
            or not np.isfinite(solution.y).all()
        ):
            raise RuntimeError(f"PySR rollout failed: {solution.message}")
        return solution.y.T

    @classmethod
    def from_model(cls, model):
        if (
            not isinstance(model, dict)
            or model.get("model_format") != "sysbiosr-model-v2"
            or model.get("method") != "pysr"
            or model.get("schema_version") != 2
        ):
            raise ValueError(
                "unsupported model format. Expected a dictionary with "
                "method='pysr', model_format='sysbiosr-model-v2', and schema_version=2."
            )
        config = PySRConfig(**model["config"])
        return cls(model["rhs_expressions"], model["state_names"], config)


class PySR:
    """Controller; all Julia training happens in a fresh worker process."""

    def __init__(
        self,
        config=None,
        *,
        state_names=None,
        output_dir=None,
        julia_executable=None,
        julia_project=None,
    ):
        self.config = config or PySRConfig()
        self.state_names = state_names
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.julia_executable = julia_executable
        self.julia_project = julia_project
        self.reset()

    def reset(self):
        self.model = None
        self.predictor = None
        self.hall_of_fame = []
        self.search_summary = {}
        self.candidates = []
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
        t, x, names = validate_inputs(times, states, self.state_names)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sysbiosr-pysr-") as scratch:
            scratch = Path(scratch)
            np.savez(scratch / "inputs.npz", times=t, states=x)
            request = {"config": self.config.to_dict(), "state_names": names}
            (scratch / "request.json").write_text(json.dumps(request))
            log_path = (
                scratch / "worker.log"
                if self.output_dir is None
                else self.output_dir / "worker.log"
            )
            env = dict(os.environ)
            env.update(
                PYTHON_JULIACALL_THREADS="1",
                PYTHON_JULIACALL_HANDLE_SIGNALS="yes",
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                TMPDIR=str(scratch),
            )
            for key, value in (
                ("PYTHON_JULIAPKG_EXE", self.julia_executable),
                ("PYTHON_JULIAPKG_PROJECT", self.julia_project),
            ):
                if value is not None:
                    env[key] = str(Path(value).expanduser().resolve())
            if self.output_dir is not None:
                self.worker_log = log_path
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "Baseline.PySR.worker", str(scratch)],
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
                        "PySR worker exceeded fit_timeout_seconds"
                    ) from exc
                except BaseException:
                    self._stop(process)
                    raise
            if self.output_dir is not None:
                self.worker_log = log_path
            reply_path = scratch / "reply.json"
            if not reply_path.is_file():
                raise RuntimeError(
                    f"PySR worker exited with code {process.returncode}: {log_path.read_text(errors='replace')[-3000:]}"
                )
            reply = json.loads(reply_path.read_text())
            self.search_summary = reply.get("search_summary", {})
            self.hall_of_fame = reply.get("hall_of_fame", [])
            if process.returncode != 0 or reply.get("status") != "success":
                raise RuntimeError(
                    "PySR fit failed: "
                    + str(reply.get("error", f"worker exit {process.returncode}"))
                )
            predictor = PySRModel.from_model(reply["model"])
            self.predictor = predictor
            self.model = reply["model"]
            self.candidates = reply.get("candidates", [predictor.equation()])
        return copy.deepcopy(self.model)

    def predict(self, times, y0):
        if self.predictor is None:
            raise RuntimeError("PySR has no fitted model")
        return self.predictor.predict(times, y0)
