from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EquationResult:
    """Standardized equation output container for SysBioSR wrappers."""

    best: str | None = None
    candidates: list[str | None] = field(default_factory=list)
    status: str = "not_fitted"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseODESRWrapper(ABC):
    """Minimal unified interface for ODE symbolic regression methods.

    All concrete wrappers should accept trajectory-first inputs at the outer API:
    - single-trajectory mode:
        times: np.ndarray shape [T]
        trajectories: np.ndarray shape [T, D]
    - multi-trajectory mode:
        times: list[np.ndarray]
        trajectories: list[np.ndarray]

    Internal implementations are free to convert trajectories into derivatives,
    prompts, conditions, surrogate targets, etc.
    """

    method_name: str = "base"

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._equation_result = EquationResult()
        self._raw_outputs: dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        times: np.ndarray | list[np.ndarray],
        trajectories: np.ndarray | list[np.ndarray],
        **kwargs: Any,
    ) -> EquationResult:
        """Fit the model on one or more trajectories and return standardized equations."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        times: np.ndarray | list[np.ndarray],
        y0: np.ndarray | list[np.ndarray] | None = None,
        **kwargs: Any,
    ) -> np.ndarray | list[np.ndarray] | None:
        """Roll out the learned symbolic ODE on the given time grid."""
        raise NotImplementedError

    def get_equations(self) -> EquationResult:
        """Return the last standardized equation result."""
        return self._equation_result

    def get_raw_outputs(self) -> dict[str, Any]:
        """Return method-specific raw outputs for debugging or deeper analysis."""
        return self._raw_outputs

    def is_fitted(self) -> bool:
        return self._is_fitted

    @staticmethod
    def validate_single_trajectory(times: np.ndarray, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, dtype=float).reshape(-1)
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2:
            raise ValueError(f"trajectory must be 2D [T, D], got shape {trajectory.shape}")
        if times.shape[0] != trajectory.shape[0]:
            raise ValueError(
                f"time length {times.shape[0]} does not match trajectory length {trajectory.shape[0]}"
            )
        return times, trajectory

    @classmethod
    def validate_trajectories(
        cls,
        times: np.ndarray | list[np.ndarray],
        trajectories: np.ndarray | list[np.ndarray],
    ) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
        if isinstance(times, list) or isinstance(trajectories, list):
            if not isinstance(times, list) or not isinstance(trajectories, list):
                raise TypeError("times and trajectories must either both be lists or both be arrays")
            if len(times) != len(trajectories):
                raise ValueError(f"len(times)={len(times)} does not match len(trajectories)={len(trajectories)}")
            checked_t, checked_y = [], []
            for t, y in zip(times, trajectories):
                t_i, y_i = cls.validate_single_trajectory(t, y)
                checked_t.append(t_i)
                checked_y.append(y_i)
            return checked_t, checked_y
        return cls.validate_single_trajectory(times, trajectories)

    def _mark_fitted(
        self,
        *,
        best: str | None,
        candidates: list[str | None] | None = None,
        status: str = "success",
        metadata: dict[str, Any] | None = None,
        raw_outputs: dict[str, Any] | None = None,
    ) -> EquationResult:
        self._is_fitted = status == "success"
        self._equation_result = EquationResult(
            best=best,
            candidates=[] if candidates is None else list(candidates),
            status=status,
            metadata={} if metadata is None else dict(metadata),
        )
        self._raw_outputs = {} if raw_outputs is None else dict(raw_outputs)
        return self._equation_result
