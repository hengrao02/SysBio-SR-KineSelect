"""Serial benchmark-profile FFX fitting and rollout selection.

Upstream projects: https://github.com/ffx-org/ffx and
https://github.com/sdascoli/odeformer.
"""

from __future__ import annotations
from itertools import product
from pathlib import Path
import json
import math
import signal
import sys
import traceback
import importlib.metadata
import numpy as np
from .core import FFXConfig, FFXModel


class FilteredBasis(ValueError):
    pass


def export_basis(base):
    from ffx.core.bases import SimpleBase, OperatorBase, ProductBase
    from ffx.core.constants import OP_ABS, OP_LOG10, OP_MAX0, OP_MIN0, OP_GTH, OP_LTH

    if isinstance(base, SimpleBase):
        return {
            "type": "power",
            "state": int(base.var),
            "exponent": float(base.exponent),
        }
    if isinstance(base, ProductBase):
        return {
            "type": "product",
            "left": export_basis(base.base1),
            "right": export_basis(base.base2),
        }
    if isinstance(base, OperatorBase):
        if base.nonlin_op in (OP_MAX0, OP_MIN0, OP_GTH, OP_LTH):
            raise FilteredBasis("the benchmark profile excludes max/min and hinge bases")
        if base.nonlin_op in (OP_ABS, OP_LOG10):
            return {
                "type": "abs" if base.nonlin_op == OP_ABS else "log10",
                "base": export_basis(base.simple_base),
            }
    raise ValueError(f"unsupported native FFX basis: {type(base).__name__}")


def export_model(model):
    from ffx.core.models import ConstantModel, FFXModel as Native

    if isinstance(model, ConstantModel):
        return {"offset": float(model.constant), "numerator": [], "denominator": []}
    if not isinstance(model, Native):
        raise ValueError("unsupported native FFX model")
    return {
        "offset": float(model.coefs_n[0]),
        "numerator": [
            {"coefficient": float(c), "basis": export_basis(b)}
            for c, b in zip(model.coefs_n[1:], model.bases_n)
        ],
        "denominator": [
            {"coefficient": float(c), "basis": export_basis(b)}
            for c, b in zip(model.coefs_d, model.bases_d)
        ],
    }


def differentiate(times, states, order, window):
    import pysindy as ps

    diff = (
        ps.FiniteDifference(order=order)
        if window is None
        else ps.SmoothedFiniteDifference(
            order=order, smoother_kws={"window_length": window}
        )
    )
    dx = np.asarray(diff._differentiate(states.copy(), times), dtype=float)
    if dx.shape != states.shape or not np.isfinite(dx).all():
        raise ValueError("nonfinite/incomplete FFX derivatives")
    return dx


def fit_fronts(times, states, order, window):
    import ffx

    dx = differentiate(times, states, order, window)
    fronts = []
    counts = []
    for i in range(states.shape[1]):
        old_handler = (
            signal.getsignal(signal.SIGALRM) if hasattr(signal, "SIGALRM") else None
        )
        try:
            native = ffx.run(
                states,
                dx[:, i],
                states,
                dx[:, i],
                varnames=[f"x_{j}" for j in range(states.shape[1])],
            )
        except SystemExit as exc:
            raise RuntimeError(f"FFX backend exited: {exc.code}") from exc
        finally:
            if old_handler is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
        rows = []
        filtered = 0
        for native_index in reversed(range(len(native))):
            try:
                component = export_model(native[native_index])
            except FilteredBasis:
                filtered += 1
                continue
            # Validate component state references against the full system dimension.
            FFXModel(
                [component] * states.shape[1],
                [f"x_{j}" for j in range(states.shape[1])],
            )
            rows.append(
                {
                    "native_index": native_index,
                    "component": component,
                    "native_num_bases": int(native[native_index].numBases()),
                    "native_complexity": int(native[native_index].complexity()),
                }
            )
        if not rows:
            raise ValueError(f"no retained FFX model for state {i}")
        fronts.append(rows)
        counts.append(
            {
                "native_models": len(native),
                "filtered_models": filtered,
                "retained_models": len(rows),
            }
        )
    return fronts, counts


def score(components, times, states, names, config):
    from sklearn.metrics import r2_score

    try:
        prediction = FFXModel(components, names, config).predict(times, states[0])
        value = float(r2_score(states, prediction, multioutput="variance_weighted"))
        if not math.isfinite(value):
            raise ValueError("nonfinite rollout R2")
        return value, None
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def choose_final(fronts, times, states, names, config):
    sizes = [len(f) for f in fronts]
    total = math.prod(sizes)
    if not sizes or not total:
        raise ValueError("no complete FFX equation vector")
    if not config.optimize_hyperparams:
        return [0] * len(fronts), {
            "policy": "source_first_component_combination",
            "combinations": total,
            "evaluated": 0,
        }
    scores = np.full(total, -np.inf)
    invalid = timeouts = 0
    for ordinal, indices in enumerate(product(*(range(n) for n in sizes))):
        value, error = score(
            [f[j]["component"] for f, j in zip(fronts, indices)],
            times,
            states,
            names,
            config,
        )
        if value is None:
            invalid += 1
            timeouts += int(error.startswith("TimeoutError:"))
        else:
            scores[ordinal] = value
    if invalid == total:
        if timeouts == total:
            raise TimeoutError("all FFX candidate rollouts timed out")
        raise ValueError("no valid FFX candidate rollout")
    ordinal = int(np.argsort(scores)[-1])
    indices = list(np.unravel_index(ordinal, tuple(sizes)))
    return [int(i) for i in indices], {
        "policy": "training_rollout_r2_full",
        "combinations": total,
        "evaluated": total,
        "invalid": invalid,
        "timeouts": timeouts,
        "selected_ordinal": ordinal,
        "selected_r2": float(scores[ordinal]),
    }


def train(times, states, names, config, summary):
    selected = (config.finite_difference_order, config.smoother_window_length)
    summary.update(status="running", stage="search", grid=[])
    if config.optimize_hyperparams:
        split = int(np.floor((1 - config.hyper_opt_eval_fraction) * len(times)))
        if min(split, len(times) - split) < 3:
            raise ValueError("insufficient train/validation samples")
        orders = list(dict.fromkeys([2, 3, 4, config.finite_difference_order]))
        windows = list(dict.fromkeys([None, 15, config.smoother_window_length]))
        best = -np.inf
        summary.update(training_samples=split, validation_samples=len(times) - split)
        for order, window in product(orders, windows):
            record = {
                "finite_difference_order": order,
                "smoother_window_length": window,
                "status": "failed",
                "validation_r2": None,
            }
            try:
                fronts, counts = fit_fronts(
                    times[:split], states[:split], order, window
                )
                value, error = score(
                    [f[0]["component"] for f in fronts],
                    times[split:],
                    states[split:],
                    names,
                    config,
                )
                record["front_counts"] = counts
                if value is None:
                    record["error"] = error
                else:
                    record.update(status="success", validation_r2=value)
                    if value > best:
                        best = value
                        selected = (order, window)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            summary["grid"].append(record)
        if not math.isfinite(best):
            raise ValueError("no valid FFX cross-validation setting")
    summary.update(
        stage="refit",
        selected_preprocessing={
            "finite_difference_order": selected[0],
            "smoother_window_length": selected[1],
        },
    )
    fronts, counts = fit_fronts(times, states, *selected)
    summary.update(stage="selection", refit_samples=len(times), front_counts=counts)
    indices, selection = choose_final(fronts, times, states, names, config)
    components = [f[j]["component"] for f, j in zip(fronts, indices)]
    model = {
        "schema_version": 1,
        "method": "ffx",
        "components": components,
        "state_names": names,
        "front_indices": indices,
        "selected_preprocessing": summary["selected_preprocessing"],
        "config": config.to_dict(),
        "model_format": "sysbiosr-model-v2",
    }
    summary.update(status="success", stage="complete", selection=selection)
    return model, fronts


def main():
    scratch = Path(sys.argv[1])
    summary = {}
    fronts = []
    try:
        for name, version in [("ffx", "2.1.0"), ("pysindy", "2.1.0")]:
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise ImportError(
                    'FFX fitting requires: python -m pip install ".[ffx]"'
                ) from exc
            if installed != version:
                raise ImportError(
                    f"FFX requires {name}=={version}; install sysbiosr[ffx]"
                )
        request = json.loads((scratch / "request.json").read_text())
        data = np.load(scratch / "inputs.npz")
        config = FFXConfig(**request["config"])
        model, fronts = train(
            data["times"], data["states"], request["state_names"], config, summary
        )
        reply = {
            "status": "success",
            "model": model,
            "component_fronts": fronts,
            "search_summary": summary,
        }
    except Exception as exc:
        traceback.print_exc()
        status = "timeout" if isinstance(exc, TimeoutError) else "failed"
        summary.update(status=status, error=f"{type(exc).__name__}: {exc}")
        reply = {
            "status": status,
            "error": summary["error"],
            "search_summary": summary,
            "component_fronts": fronts,
        }
    (scratch / "reply.json").write_text(json.dumps(reply, allow_nan=False))
    return 0 if reply["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
