"""ODEFormer inference protocol and portable physical-scale equations.

Upstream implementation: https://github.com/sdascoli/odeformer.
Integration details are summarized in the root README.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import copy
import math
import time
import numpy as np


@dataclass(frozen=True)
class ODEFormerConfig:
    checkpoint_path: str = "~/.cache/sysbiosr/odeformer/odeformer.pt"
    device: str = "cuda:0"
    beam_size: int = 50
    beam_temperature: float = 0.1
    max_input_points: int = 10000
    max_generated_output_len: int = 200
    rescale: bool = True
    sort_metric: str = "r2"
    random_seed: int = 0
    rollout_rtol: float = 1e-3
    rollout_atol: float = 1e-6
    rollout_timeout: float | None = 1.0

    def __post_init__(self):
        for key in ("beam_size", "max_input_points", "max_generated_output_len"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_generated_output_len < 3:
            raise ValueError("max_generated_output_len must be at least 3")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be an integer in [0, 2**32)")
        if type(self.rescale) is not bool or self.sort_metric != "r2":
            raise ValueError("rescale must be boolean; this profile uses rollout r2")
        if not isinstance(self.device, str) or not (
            self.device == "cpu"
            or self.device == "cuda"
            or (self.device.startswith("cuda:") and self.device[5:].isdigit())
        ):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        if not isinstance(self.checkpoint_path, (str, Path)) or not str(
            self.checkpoint_path
        ):
            raise ValueError(
                "checkpoint_path must name the official ODEFormer checkpoint"
            )
        object.__setattr__(self, "checkpoint_path", str(self.checkpoint_path))
        for key in (
            "beam_temperature",
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

    def bind_checkpoint(self):
        path = Path(self.checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"ODEFormer checkpoint not found: {path}; set checkpoint_path"
            )
        return replace(self, checkpoint_path=str(path))

    def to_dict(self):
        result = asdict(self)
        result.pop("checkpoint_path")
        return result


def prefix_expression(prefix):
    """Render a prefix tree with explicit parentheses, preserving numeric tokens."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("empty ODEFormer prefix")
    tokens = iter(prefix.split(","))
    binary = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}
    unary = {
        "abs",
        "sqrt",
        "log",
        "exp",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
    }

    def parse():
        token = next(tokens)
        if token in binary:
            return f"({parse()}){binary[token]}({parse()})"
        if token in unary:
            return f"{token}({parse()})"
        if token == "inv":
            return f"1/({parse()})"
        if token in {"pow2", "pow3"}:
            return f"({parse()})**{token[-1]}"
        if token == "id":
            return parse()
        if token == "t" or (token.startswith("x_") and token[2:].isdigit()):
            return token
        if token in {"e", "pi", "euler_gamma"}:
            return repr(
                float({"e": np.e, "pi": np.pi, "euler_gamma": np.euler_gamma}[token])
            )
        if not math.isfinite(float(token)):
            raise ValueError("nonfinite expression constant")
        return f"({token})"

    try:
        expression = parse()
    except (StopIteration, RecursionError) as exc:
        raise ValueError("malformed ODEFormer prefix") from exc
    if next(tokens, None) is not None:
        raise ValueError("unused ODEFormer prefix tokens")
    return expression


class ODEFormerModel:
    """Frozen equation vector; no checkpoint, Torch or external repository required."""

    def __init__(self, prefixes, state_names, config=None):
        import numexpr as ne

        if (
            not isinstance(state_names, list)
            or not state_names
            or any(not isinstance(n, str) or not n for n in state_names)
            or len(set(state_names)) != len(state_names)
        ):
            raise ValueError("state_names must be unique nonempty strings")
        if not isinstance(prefixes, list) or len(prefixes) != len(state_names):
            raise ValueError("one complete equation is required per state")
        self.state_names = list(state_names)
        self.prefixes = list(prefixes)
        self.config = config or ODEFormerConfig()
        self.expressions = [prefix_expression(p) for p in prefixes]
        self._compiled = [ne.NumExpr(e) for e in self.expressions]
        allowed = {f"x_{i}" for i in range(len(state_names))} | {"t"}
        if any(set(c.input_names) - allowed for c in self._compiled):
            raise ValueError("expression refers to an unknown state")

    def equation(self):
        return " | ".join(self.expressions)

    def rhs(self, state, time_value=0.0):
        state = np.asarray(state, dtype=float)
        if (
            state.shape != (len(self.state_names),)
            or not np.isfinite(state).all()
            or not math.isfinite(time_value)
        ):
            raise ValueError("invalid state or time")
        variables = {f"x_{i}": value for i, value in enumerate(state)}
        variables["t"] = time_value
        values = [c(*[variables[n] for n in c.input_names]) for c in self._compiled]
        if any(np.iscomplexobj(v) for v in values):
            raise ValueError("complex ODEFormer RHS")
        result = np.asarray(values, dtype=float).reshape(-1)
        if result.shape != state.shape or not np.isfinite(result).all():
            raise ValueError("nonfinite or incomplete ODEFormer RHS")
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
                raise TimeoutError("ODEFormer rollout exceeded rollout_timeout")
            return self.rhs(state, time_value)

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
            raise RuntimeError(f"ODEFormer rollout failed: {solution.message}")
        return solution.y.T

    @classmethod
    def from_model(cls, model):
        if (
            not isinstance(model, dict)
            or type(model.get("schema_version")) is not int
            or model["schema_version"] != 1
            or model.get("method") != "odeformer"
        ):
            raise ValueError("unsupported ODEFormer model schema")
        config = ODEFormerConfig(**model["config"])
        if model.get("model_format") != "sysbiosr-model-v2":
            raise ValueError(
                "unsupported model format. Expected model_format='sysbiosr-model-v2'."
            )
        result = cls(model["prefixes"], model["state_names"], config)
        if result.expressions != model["rhs_expressions"]:
            raise ValueError("ODEFormer expression/prefix mismatch")
        return result


class ODEFormer:
    def __init__(self, config=None, *, state_names=None):
        self.config = config or ODEFormerConfig()
        self.state_names = state_names
        self.reset()

    def reset(self):
        self.model = None
        self.predictor = None
        self.search_results = []
        self.search_summary = {}

    def fit(self, times, states):
        self.reset()
        self.search_summary = {"status": "running", "stage": "prepare"}
        try:
            return self._fit(times, states)
        except Exception as exc:
            self.model = None
            self.predictor = None
            self.search_summary.update(
                status="failed", error=f"{type(exc).__name__}: {exc}"
            )
            raise

    def _fit(self, times, states):
        from .inference.backend import generate_candidates
        from sklearn.metrics import r2_score

        t, x = np.asarray(times, dtype=float), np.asarray(states, dtype=float)
        if (
            t.ndim != 1
            or x.ndim != 2
            or len(t) != len(x)
            or len(t) < 2
            or not 1 <= x.shape[1] <= 6
            or not np.isfinite(t).all()
            or not np.isfinite(x).all()
            or not np.all(np.diff(t) > 0)
        ):
            raise ValueError(
                "expected finite increasing times [T] and states [T,D], T >= 2, 1 <= D <= 6"
            )
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
        self.config = self.config.bind_checkpoint()
        self.search_summary["stage"] = "decode"
        candidates, scaling = generate_candidates(t, x, self.config)
        self.search_summary.update(
            stage="selection", scaling=scaling, candidates_evaluated=len(candidates)
        )
        scores = []
        for index, candidate in enumerate(candidates):
            record = {
                **candidate,
                "index": index,
                "status": "invalid",
                "rollout_r2": None,
            }
            try:
                if candidate.get("error"):
                    raise ValueError(candidate["error"])
                predictor = ODEFormerModel(candidate["prefixes"], names, self.config)
                prediction = predictor.predict(t, x[0])
                score = float(r2_score(x, prediction, multioutput="variance_weighted"))
                if not math.isfinite(score):
                    raise ValueError("nonfinite rollout R2")
                record.update(
                    status="success",
                    rollout_r2=score,
                    rhs_expressions=predictor.expressions,
                )
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                SyntaxError,
                KeyError,
            ) as exc:
                record.update(error=f"{type(exc).__name__}: {exc}")
            except TimeoutError as exc:
                record.update(status="timeout", error=f"TimeoutError: {exc}")
            scores.append(
                record["rollout_r2"] if record["status"] == "success" else -np.inf
            )
            self.search_results.append(record)
        if not any(math.isfinite(s) for s in scores):
            if self.search_results and all(
                r["status"] == "timeout" for r in self.search_results
            ):
                raise TimeoutError("all ODEFormer candidate rollouts timed out")
            raise ValueError("no valid ODEFormer candidate rollout")
        selected_index = int(np.argsort(scores)[-1])
        selected = self.search_results[selected_index]
        self.predictor = ODEFormerModel(selected["prefixes"], names, self.config)
        self.model = {
            "schema_version": 1,
            "method": "odeformer",
            "prefixes": selected["prefixes"],
            "rhs_expressions": self.predictor.expressions,
            "state_names": names,
            "selected_index": selected_index,
            "scaling": scaling,
            "config": self.config.to_dict(),
            "model_format": "sysbiosr-model-v2",
        }
        self.search_summary.update(
            status="success",
            stage="complete",
            selected_index=selected_index,
            selected_rollout_r2=selected["rollout_r2"],
        )
        return copy.deepcopy(self.model)

    def equation(self):
        if self.predictor is None:
            raise RuntimeError("ODEFormer is not fitted")
        return self.predictor.equation()

    def predict(self, times, y0):
        if self.predictor is None:
            raise RuntimeError("ODEFormer is not fitted")
        return self.predictor.predict(times, y0)

    @staticmethod
    def from_model(model):
        return ODEFormerModel.from_model(model)
