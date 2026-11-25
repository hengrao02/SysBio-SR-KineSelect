# SysBioSR and KineSelect

[![GitHub Code](https://img.shields.io/badge/GitHub-Code-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/hengrao02/SysBio-SR-KineSelect) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/HengRao/SysBio-SR) [![Benchmark: 198 systems](https://img.shields.io/badge/Benchmark-198%20systems-18766B?style=flat-square)](Dataset/SysBioSR/benchmark/README.md) [![Reaction evaluation: 161 systems](https://img.shields.io/badge/Reaction%20evaluation-161%20systems-296B73?style=flat-square)](Dataset/SysBioSR/benchmark/benchmark.yaml) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)

SysBioSR is a symbolic regression benchmark with 198 biological ODE systems. KineSelect infers stoichiometrically structured reaction models from time series. Seven baselines use the same fitting and evaluation protocol.

## Run KineSelect on one system

From the repository root, create a Python 3.10+ environment and run the two-state photosynthesis example (`BIOMD0000000386`).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements.lock -e .
python run.py Configs/examples/kineselectsr.yaml --results-root Results/quickstart
```

The example fits the default trajectory and simulates the same model from a perturbed initial condition. Under `Results/quickstart/`, each trajectory directory contains the scores in `summary.json`, the predictions in `generated_trajectory.csv` and a comparison plot in `overlay.png`.

## Repository structure

```text
SysBio-SR-KineSelect/
├── run.py                         # Run experiments from YAML configs
├── Configs/
│   ├── examples/
│   ├── smoke/
│   └── benchmark/                 # Formal run configs
├── Dataset/SysBioSR/benchmark/    # Data and evaluation protocol
├── Methods/KineSelectSR/          # KineSelect implementation
├── Baseline/                      # Comparison methods
├── DataAdapters/                  # Trajectory loading
├── Wrappers/                      # fit() and predict()
├── Runners/                       # Single and batch runs
├── Metrics/
│   ├── trajectory_metrics.py
│   └── reaction_annotation_metrics.py
├── Scripts/aggregate_results.py
├── Utils/
├── pyproject.toml
└── requirements.lock
```

## Benchmark and evaluation

Fit and select models using `traj_default.csv`. Start each scored simulation at the first observed state of its target trajectory. Reserve `traj_perturbed.csv`, reference equations, reference models and reaction annotations for evaluation. All 198 systems are eligible for trajectory scoring. The 95 Main and 66 Challenge systems also support reaction scoring for methods that return explicit reactions. The remaining 37 systems form the Stress partition. [Benchmark files, protocol and data sources](Dataset/SysBioSR/benchmark/README.md) · [System membership](Dataset/SysBioSR/benchmark/benchmark.yaml)

To score your own trajectory, save a CSV with the benchmark time grid and named state columns. This example scores the quick-start prediction. Replace `prediction_file` with your CSV for the same system and trajectory.

```python
from pathlib import Path

import numpy as np
import pandas as pd
from Metrics import compute_trajectory_metrics

system = "BIOMD0000000386"
data = Path("Dataset/SysBioSR/benchmark/main") / system / "traj"
run = Path("Results/quickstart/kineselectsr-example/kineselectsr/sysbiosr-1.0.0-rc.2/main")
prediction_file = run / system / "perturbed/seed-42/generated_trajectory.csv"
observed = pd.read_csv(data / "traj_perturbed.csv", index_col="time")
predicted = pd.read_csv(prediction_file, index_col="time")
np.testing.assert_allclose(predicted.index, observed.index, rtol=1e-12, atol=1e-12)
states = observed.columns.tolist()
scores = compute_trajectory_metrics(observed.to_numpy(), predicted[states].to_numpy(), states)
print(scores["overall"])
```

## Reproduce the benchmark

Run the three KineSelect partition configs with their formal search budgets, then aggregate the perturbed-trajectory and reaction scores.

```bash
for partition in main challenge stress; do
  python run.py "Configs/benchmark/kineselectsr/$partition.yaml" --results-root Results/paper
done

python Scripts/aggregate_results.py --results-root Results/paper \
  --dataset-root Dataset/SysBioSR --experiment-id sysbiosr-paper \
  --method kineselectsr --seed 42 --output-dir Results/paper-aggregate
```

`Results/paper-aggregate/summary.json` reports accuracy and coverage. `run_index.jsonl` records the status of each system, including missing and failed runs.

## Baselines

Each method has three partition configs under `Configs/benchmark/<method>/` and a small execution check under `Configs/smoke/`. For methods with an extra, install it in the same environment, for example `python -m pip install -c requirements.lock -e '.[sindy]'`.

| Method key | Installation | Method reference |
| --- | --- | --- |
| `reactive_sindy` | Base install | [Reactive SINDy](https://doi.org/10.1063/1.5066099) |
| `sisr` | Base install | [SISR](https://github.com/rxhernandez/SISR) |
| `sindy_pi` | Base install | [SINDy-PI](https://github.com/dynamicslab/SINDy-PI) |
| `sindy` | `.[sindy]` | [PySINDy](https://github.com/dynamicslab/pysindy) |
| `pysr` | `.[pysr]` | [PySR](https://github.com/MilesCranmer/PySR) |
| `odeformer` | `.[odeformer]` | [ODEFormer](https://github.com/sdascoli/odeformer) |
| `ffx` | `.[ffx]` | [FFX](https://github.com/ffx-org/ffx) |

PySR initializes its Julia backend on first use. For ODEFormer, obtain the upstream `odeformer.pt` checkpoint and place it at `~/.cache/sysbiosr/odeformer/odeformer.pt`, or set `checkpoint_path` in the config. The formal ODEFormer configs use `cuda:0` and accept a `device` override in YAML.
