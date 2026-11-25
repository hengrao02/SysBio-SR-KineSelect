#!/usr/bin/env python3
"""Replay one reference trajectory without changing the released data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sysbiosr_reference import load_reference_model, simulate_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-dir", type=Path, required=True)
    parser.add_argument("--trajectory", choices=("default", "perturbed"), default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    system = args.system_dir.resolve()
    output = args.output_dir.resolve()
    dataset = system.parents[1]
    if output == dataset or dataset in output.parents:
        parser.error("simulation output must be outside the released dataset")
    model = load_reference_model(system)
    metadata = json.loads((system / "traj/trajectory_metadata.json").read_text())
    spec = next(r for r in metadata["trajectories"] if r["name"] == args.trajectory)
    observed = np.genfromtxt(system / "traj" / spec["file"], delimiter=",", names=True)
    times = np.linspace(float(observed["time"][0]), float(observed["time"][-1]), len(observed))
    values = simulate_reference(model, times, y0=spec["initial_condition"], consts=metadata["default_consts"])
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{system.name}_{args.trajectory}.csv"
    with target.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", *model.state_names])
        writer.writerows(np.column_stack((times, values)))
    print(target)


if __name__ == "__main__":
    main()
