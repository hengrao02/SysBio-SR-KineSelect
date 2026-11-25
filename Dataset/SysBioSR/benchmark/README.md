# SysBioSR Benchmark

[![GitHub Code](https://img.shields.io/badge/GitHub-Code-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/hengrao02/SysBio-SR-KineSelect) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/HengRao/SysBio-SR)

SysBioSR is a benchmark for symbolic regression on biological time series. It pairs simulated trajectories from 198 literature-derived ODE models with their reference equations and reaction annotations. Together, these models comprise 661 state equations.

## Benchmark partitions

| Partition | Systems | Kinetic scope | Reaction evaluation |
| --- | ---: | --- | --- |
| **Main** | 95 | Mass-action kinetics only | Included |
| **Challenge** | 66 | Michaelis–Menten or Hill kinetics | Included |
| **Stress** | 37 | Complex non-canonical kinetic forms | Not included |

All 198 systems support trajectory evaluation. The 161 systems in the Main and Challenge partitions also support reaction-level evaluation.

A model’s most complex kinetic law determines its partition. Challenge and Stress models can also contain simpler reactions. [System membership and evaluation protocol](benchmark.yaml)

## 📦 Directory structure

```text
Dataset/SysBioSR/benchmark/
├── README.md
├── benchmark.yaml
├── simulate_reference.py
├── sysbiosr_reference/
└── {main,challenge,stress}/
    └── <system_id>/
        ├── <source_model>.xml
        ├── equation.json
        ├── model.py
        ├── reaction_annotation.json
        └── traj/
            ├── traj_default.csv
            ├── traj_perturbed.csv
            └── trajectory_metadata.json
```

## 🧪 Evaluation protocol

SysBioSR evaluates symbolic regression methods along two complementary axes:

- **Trajectory fidelity:** methods fit the default trajectory and are evaluated through in-distribution reconstruction and out-of-distribution simulation from perturbed initial conditions.
- **Mechanism recovery:** reference reaction annotations are used for post-hoc evaluation on Main and Challenge systems.

Each scored simulation starts from the first observed state of the trajectory being evaluated. Model fitting and selection use only the default trajectory. Perturbed trajectories, reference equations, reference models and reaction annotations are reserved for evaluation.

## ⚙️ Standalone reference simulation

The standalone simulator requires NumPy and SciPy. Run both examples from the directory containing `benchmark.yaml`. From the GitHub repository root, enter it with `cd Dataset/SysBioSR/benchmark`.

```bash
python -m pip install numpy scipy
python simulate_reference.py \
  --system-dir main/BIOMD0000000006 \
  --trajectory perturbed \
  --output-dir ../simulation-output
```

The command writes `../simulation-output/BIOMD0000000006_perturbed.csv`. If that file already exists, choose another output directory outside the benchmark. Reference replay uses the initial condition recorded in the trajectory metadata and a uniform time grid. Benchmark scoring starts from the first observed state of the trajectory being evaluated.

From the same directory, use the Python API to supply an initial state or model constants.

```python
from sysbiosr_reference import load_reference_model, simulate_reference

model = load_reference_model("main/BIOMD0000000006")
values = simulate_reference(model, [0.0, 0.1, 0.2], y0=[0.0, 0.0])
```

Keep benchmark data and reference model sources unchanged when running experiments.

## 🧬 Data provenance

SysBioSR is constructed from literature-derived biological models collected through BioModels [1] and corresponding trajectories from SysBio-Traj [2], with ODEbase [3] serving as an additional reference during system collection and equation curation. The models are converted into standardized executable ODE representations with canonical state ordering and generated default and perturbed trajectories. Original model identifiers, publications and attribution remain in the bundled XML files.

## 📚 References

[1] Li, C., Donizelli, M., Rodriguez, N., et al. “BioModels Database: An Enhanced, Curated and Annotated Resource for Published Quantitative Kinetic Models.” *BMC Systems Biology* 4, 92 (2010). https://doi.org/10.1186/1752-0509-4-92

[2] Rao, H., Zhang, J. Z., Gu, Y., et al. “A Regime-Aware Trajectory Prediction Framework for 1000+ Systems Biology Models.” *International Conference on Machine Learning* (2026). https://openreview.net/forum?id=sI3UUkXJxs

[3] Lüders, C., Sturm, T., and Radulescu, O. “ODEbase: A Repository of ODE Systems for Systems Biology.” *Bioinformatics Advances* 2(1), vbac027 (2022). https://doi.org/10.1093/bioadv/vbac027
