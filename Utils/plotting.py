from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_trajectory_overlay_plot(
    times: np.ndarray,
    fit_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    state_names: list[str],
    output_path: str | Path,
    *,
    title: str = "Trajectory fit vs generated-equation rollout",
) -> Path:
    times = np.asarray(times, dtype=float)
    fit_trajectory = np.asarray(fit_trajectory, dtype=float)
    pred_trajectory = np.asarray(pred_trajectory, dtype=float)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_states = fit_trajectory.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(10, max(3, 2.8 * n_states)), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times, fit_trajectory[:, i], label="fit trajectory", linewidth=2)
        ax.plot(times, pred_trajectory[:, i], "--", label="generated equation", linewidth=1.5)
        ax.set_ylabel(state_names[i])
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend()

    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
