"""Isolated PySR fit and full-rollout final candidate selection.

Upstream projects: https://github.com/MilesCranmer/PySR and
https://github.com/sdascoli/odeformer.
"""

from __future__ import annotations
from importlib.metadata import version
from itertools import product
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
from .core import PySRConfig, PySRModel, validate_inputs


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def normalize_hof(hof, dimension):
    frames = hof if isinstance(hof, list) else [hof]
    if len(frames) != dimension:
        raise ValueError("PySR HOF output dimension mismatch")
    outputs = []
    for frame in frames:
        if frame is None or frame.empty or "score" not in frame:
            raise ValueError("PySR returned an empty or invalid HOF")
        frame = frame.sort_values("score", ascending=False, kind="mergesort")
        records = []
        for _, row in frame.iterrows():
            expression = str(row.get("sympy_format", row.get("equation", ""))).strip()
            if not expression:
                continue
            score = float(row["score"])
            records.append(
                {
                    "expression": expression,
                    "complexity": int(row["complexity"]),
                    "loss": _number(row["loss"]),
                    "score": _number(score),
                    "score_kind": (
                        "nan"
                        if np.isnan(score)
                        else (
                            "+inf"
                            if score == np.inf
                            else ("-inf" if score == -np.inf else "finite")
                        )
                    ),
                }
            )
        if not records:
            raise ValueError("PySR returned no usable HOF expressions")
        outputs.append(records)
    return outputs


def rollout_score(expressions, times, states, names, config):
    from sklearn.metrics import r2_score

    try:
        prediction = PySRModel(expressions, names, config).predict(times, states[0])
        score = float(r2_score(states, prediction, multioutput="variance_weighted"))
        if not np.isfinite(score):
            raise ValueError("non-finite rollout R2")
        return score, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def choose_final(hof, times, states, names, config):
    sizes = tuple(len(rows) for rows in hof)
    count = math.prod(sizes)
    if not sizes or not count:
        raise ValueError("PySR HOF has no complete equation vector")
    policy = "component_score_first"
    selected_index = 0
    selected_score = None
    selected_error = None
    evaluations = valid = timeouts = 0
    if config.optimize_hyperparams and count > 1:
        policy = "training_rollout_r2_full"
        # Keep numeric scores/codes only, not the Cartesian expression vectors.
        scores = np.empty(count, dtype=float)
        failures = np.zeros(count, dtype=np.uint8)
        for ordinal, idx in enumerate(product(*(range(size) for size in sizes))):
            expressions = [rows[i]["expression"] for rows, i in zip(hof, idx)]
            score, error = rollout_score(expressions, times, states, names, config)
            if score is not None and np.isfinite(score) and error is None:
                scores[ordinal] = score
                valid += 1
            else:
                scores[ordinal] = -np.inf
                if error and error.startswith("TimeoutError:"):
                    failures[ordinal] = 1
                    timeouts += 1
                else:
                    failures[ordinal] = 2
            evaluations += 1
        # Same first item as ODEFormer's np.argsort(scores)[::-1], including ties.
        # Invalid candidates use -inf here, instead of its metric's failure 0.
        selected_index = int(np.argsort(scores)[-1])
        if failures[selected_index] == 0:
            selected_score = float(scores[selected_index])
        else:
            selected_error = (
                "rollout_timeout"
                if failures[selected_index] == 1
                else "invalid_or_nonfinite_rollout"
            )
    remaining = selected_index
    indices = []
    for size in reversed(sizes):
        remaining, position = divmod(remaining, size)
        indices.append(position)
    indices.reverse()
    winner = {
        "hof_indices": indices,
        "expressions": [rows[i]["expression"] for rows, i in zip(hof, indices)],
        "rollout_r2": selected_score,
        "error": selected_error,
    }
    return winner, {
        "policy": policy,
        "total_combinations": count,
        "rollout_evaluations": evaluations,
        "valid_rollouts": valid,
        "invalid_rollouts": evaluations - valid,
        "timeout_rollouts": timeouts,
        "all_invalid": evaluations > 0 and valid == 0,
        "candidate_details": "selected_only",
        "candidates": [winner],
    }


def native_fit(factory, times, states, config, fd_order, window, run_id):
    from pysindy import FiniteDifference, SmoothedFiniteDifference

    diff = (
        FiniteDifference(order=fd_order)
        if window is None
        else SmoothedFiniteDifference(
            order=fd_order,
            smoother_kws={"window_length": window, "polyorder": 3, "axis": 0},
        )
    )
    derivatives = np.asarray(diff._differentiate(states, times), dtype=float)
    if derivatives.shape != states.shape or not np.isfinite(derivatives).all():
        raise ValueError("invalid estimated derivatives")
    model = factory(
        niterations=config.niterations,
        binary_operators=list(config.binary_operators),
        unary_operators=list(config.unary_operators),
        elementwise_loss=config.loss,
        populations=config.populations,
        population_size=config.population_size,
        maxsize=config.maxsize,
        ncycles_per_iteration=config.ncycles_per_iteration,
        precision=config.precision,
        random_state=config.random_seed,
        deterministic=config.deterministic,
        parallelism=config.parallelism,
        temp_equation_file=True,
        delete_tempfiles=True,
        run_id=run_id,
        progress=False,
        verbosity=0,
        update_verbosity=0,
        input_stream="devnull",
    )
    model.fit(
        X=states,
        y=derivatives[:, 0] if states.shape[1] == 1 else derivatives,
        variable_names=[f"x_{i}" for i in range(states.shape[1])],
    )
    return normalize_hof(model.get_hof(), states.shape[1])


def fit_protocol(factory, times, states, names, config, summary):
    selected = (config.finite_difference_order, config.smoother_window_length)
    summary.update(
        status="running", random_seed=config.random_seed, hyperparameter_trials=[]
    )
    if config.optimize_hyperparams:
        split = math.floor((1 - config.hyper_opt_eval_fraction) * len(times))
        if split < 3 or len(times) - split < 2:
            raise ValueError(
                "insufficient training/validation samples for PySR selection"
            )
        summary.update(training_samples=split, validation_samples=len(times) - split)
        best_score = -np.inf
        for i, (order, window) in enumerate(config.derivative_candidates()):
            record = {
                "finite_difference_order": order,
                "smoother_window_length": window,
                "status": "failed",
                "rollout_r2": None,
                "random_seed": config.random_seed,
            }
            try:
                hof = native_fit(
                    factory,
                    times[:split],
                    states[:split],
                    config,
                    order,
                    window,
                    f"search_{i}",
                )
                candidate = [rows[0]["expression"] for rows in hof]
                score, error = rollout_score(
                    candidate, times[split:], states[split:], names, config
                )
                record.update(
                    expressions=candidate,
                    rollout_r2=score,
                    error=error,
                    status="success" if score is not None else "invalid_rollout",
                )
                if score is not None and score > best_score:
                    selected, best_score = (order, window), score
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            summary["hyperparameter_trials"].append(record)
        summary["selection_status"] = (
            "selected" if np.isfinite(best_score) else "fallback_no_finite_score"
        )
    else:
        summary["selection_status"] = "fixed_parameters"
    summary["selected_derivative_settings"] = {
        "finite_difference_order": selected[0],
        "smoother_window_length": selected[1],
    }
    hof = native_fit(factory, times, states, config, *selected, "refit")
    winner, selection = choose_final(hof, times, states, names, config)
    summary.update(
        status="success", refit_samples=len(times), final_selection=selection
    )
    # Compile the selected vector before declaring fit success.
    PySRModel(winner["expressions"], names, config)
    model = {
        "schema_version": 2,
        "method": "pysr",
        "rhs_expressions": winner["expressions"],
        "state_names": names,
        "hof_indices": winner["hof_indices"],
        "selected_derivative_settings": summary["selected_derivative_settings"],
        "selection_policy": selection["policy"],
        "config": config.to_dict(),
        "model_format": "sysbiosr-model-v2",
    }
    return model, hof, [" | ".join(r["expressions"]) for r in selection["candidates"]]


def main():
    directory = Path(sys.argv[1])
    reply = {"status": "failed", "search_summary": {}}
    try:
        request = json.loads((directory / "request.json").read_text())
        config = PySRConfig(**request["config"])
        arrays = np.load(directory / "inputs.npz", allow_pickle=False)
        times, states, names = validate_inputs(
            arrays["times"], arrays["states"], request["state_names"]
        )
        if version("pysr") != "1.5.10" or version("pysindy") != "2.1.0":
            raise ImportError(
                "PySR fitting requires pysr==1.5.10 and pysindy==2.1.0; install '.[pysr]'"
            )
        from pysr import PySRRegressor
        from pysr.julia_import import jl

        model, hof, candidates = fit_protocol(
            PySRRegressor, times, states, names, config, reply["search_summary"]
        )
        model["dependency_versions"] = {
            name: version(name)
            for name in (
                "pysr",
                "pysindy",
                "numpy",
                "scipy",
                "scikit-learn",
                "numexpr",
                "juliacall",
            )
        }
        model["dependency_versions"].update(
            julia=str(jl.VERSION),
            symbolic_regression=str(jl.seval("string(pkgversion(SymbolicRegression))")),
        )
        reply.update(
            status="success", model=model, hall_of_fame=hof, candidates=candidates
        )
    except Exception as exc:
        reply["search_summary"]["status"] = "failed"
        reply.update(
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        if isinstance(exc, ImportError):
            reply[
                "error"
            ] += "; install optional dependencies with python -m pip install '.[pysr]'"
        print(reply["traceback"], file=sys.stderr)
    (directory / "reply.json").write_text(
        json.dumps(reply, indent=2, allow_nan=False) + "\n"
    )
    return 0 if reply["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
