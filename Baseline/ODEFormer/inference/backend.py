"""Pretrained sampling and scaling; the only checkpoint-loading boundary.

Upstream implementation: https://github.com/sdascoli/odeformer.
"""

from contextlib import contextmanager
import pickle
import random
import types
import numpy as np


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("odeformer."):
            module = "Baseline.ODEFormer.inference." + module[len("odeformer.") :]
        return super().find_class(module, name)


def load_checkpoint(config):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'ODEFormer inference requires: python -m pip install ".[odeformer]"'
        ) from exc
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA requested ({config.device}) but no CUDA device is available"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index unavailable: {config.device}")
    # The original public checkpoint is a serialized model, not a state_dict.
    # Remap only ODEFormer classes locally; never modify sys.path/sys.modules.
    reader = types.ModuleType("odeformer_checkpoint_pickle")
    reader.Unpickler = _CheckpointUnpickler
    reader.load = pickle.load
    model = torch.load(
        config.checkpoint_path,
        map_location=device,
        weights_only=False,
        pickle_module=reader,
    )
    from .model.model_wrapper import ModelWrapper

    if not isinstance(model, ModelWrapper) or model.env.params.use_two_hot:
        raise ValueError("expected the official ODEFormer sampling checkpoint")
    model.to(device).eval()
    model.device = device
    model.beam_type = "sampling"
    model.beam_size = config.beam_size
    model.beam_temperature = config.beam_temperature
    model.max_generated_output_len = config.max_generated_output_len
    model.random_seed = config.random_seed
    model.average_trajectories = False
    return model


@contextmanager
def inference_rng(seed, device):
    import torch

    py_state, np_state = random.getstate(), np.random.get_state()
    devices = (
        []
        if device.type == "cpu"
        else [device.index if device.index is not None else torch.cuda.current_device()]
    )
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def generate_candidates(times, states, config):
    from .model.scaler import Scaler

    model = load_checkpoint(config)
    if states.shape[1] > model.env.params.max_dimension:
        raise ValueError("trajectory exceeds checkpoint state dimension")
    scaler = Scaler(time_range=[1, 10], feature_scale=1) if config.rescale else None
    if scaler is not None:
        scaled_t, scaled_x = scaler.fit_transform(times, states)
        a_t, b_t, scale = scaler.get_params()
    else:
        scaled_t, scaled_x = times.copy(), states.copy()
        a_t, b_t, scale = 1.0, 0.0, np.ones(states.shape[1])
    if (
        not np.isfinite(scaled_t).all()
        or not np.isfinite(scaled_x).all()
        or not np.isfinite(scale).all()
    ):
        raise ValueError("nonfinite ODEFormer input scaling")
    with inference_rng(config.random_seed, model.device):
        order = np.random.permutation(len(times))
        inputs = [
            [
                [scaled_t[i], scaled_x[i]]
                for i in order[start : start + config.max_input_points]
            ]
            for start in range(0, len(order), config.max_input_points)
        ]
        outputs = model(inputs)
    if len(outputs) != len(inputs):
        raise ValueError("incomplete ODEFormer decoder output")
    candidates = []
    for bag, trees in enumerate(outputs):
        if not trees:
            candidates.append(
                {"bag": bag, "prefixes": None, "error": "no decoded equation"}
            )
        for tree in trees:
            record = {"bag": bag, "prefixes": None}
            try:
                if tree is None or len(tree.nodes) != states.shape[1]:
                    raise ValueError("decoded equation dimension mismatch")
                record["raw_prefixes"] = [n.prefix() for n in tree.nodes]
                if scaler is not None:
                    tree = scaler.rescale_function(model.env, tree, a_t, b_t, scale)
                    if tree is None:
                        raise ValueError("could not restore physical scales")
                    try:
                        tree = model.env.simplifier.simplify_tree(tree)
                    except (ValueError, TimeoutError, ArithmeticError, RecursionError):
                        record["simplification"] = "kept_unsimplified"
                if tree is None or len(tree.nodes) != states.shape[1]:
                    raise ValueError("invalid rescaled equation")
                record["prefixes"] = [n.prefix() for n in tree.nodes]
            except (ValueError, ArithmeticError, AttributeError, IndexError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            candidates.append(record)
    scaling = {
        "time_scale": float(a_t),
        "time_shift": float(b_t),
        "state_scale": scale.tolist(),
        "bags": len(inputs),
        "beam_size": config.beam_size,
    }
    return candidates, scaling
