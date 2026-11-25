"""Trajectory-first adapter for the independent SISR baseline."""

from __future__ import annotations
from typing import Any
import numpy as np
from .base_ode_sr_wrapper import BaseODESRWrapper, EquationResult


class SISRWrapper(BaseODESRWrapper):
    method_name = "sisr"

    def __init__(self, *, state_names: list[str] | None = None, **config: Any) -> None:
        super().__init__()
        from Baseline.SISR import SISR, SISRConfig

        self.estimator = SISR(SISRConfig(**config), state_names=state_names)
        self.config = self.estimator.config
        self.fit_result = None

    def fit(
        self,
        times: np.ndarray | list[np.ndarray],
        trajectories: np.ndarray | list[np.ndarray],
        **kwargs: Any,
    ) -> EquationResult:
        self._is_fitted = False
        self._equation_result = EquationResult()
        self._raw_outputs = {}
        self.fit_result = None
        if kwargs:
            raise TypeError(f"unsupported SISR fit overrides: {sorted(kwargs)}")
        if isinstance(times, list) or isinstance(trajectories, list):
            if not isinstance(times, list) or not isinstance(trajectories, list):
                raise TypeError("times and trajectories must both be lists or arrays")
            if len(times) != 1 or len(trajectories) != 1:
                raise NotImplementedError("SISR fits exactly one trajectory")
            times, trajectories = times[0], trajectories[0]
        result = self.estimator.fit(times, trajectories)
        self.fit_result = result
        return self._mark_fitted(
            best=result.equation,
            candidates=[result.equation] if result.equation else [],
            status=result.status,
            metadata={**result.metadata, "selected_model": result.model},
            raw_outputs={"selected_model": result.model, "pareto_front": result.front},
        )

    def predict(
        self,
        times: np.ndarray | list[np.ndarray],
        y0: np.ndarray | list[np.ndarray] | None = None,
        **kwargs: Any,
    ) -> np.ndarray | list[np.ndarray]:
        if kwargs:
            raise TypeError(f"unsupported SISR prediction options: {sorted(kwargs)}")
        if not self.is_fitted():
            raise RuntimeError("SISRWrapper must be fitted before predict")
        if isinstance(times, list):
            if not isinstance(y0, list) or len(times) != len(y0):
                raise ValueError("list prediction requires matching time/y0 lists")
            return [self.estimator.predict(t, initial) for t, initial in zip(times, y0)]
        if y0 is None:
            raise ValueError("prediction requires y0")
        return self.estimator.predict(times, y0)
