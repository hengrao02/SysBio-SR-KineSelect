"""Trajectory-first adapter for pretrained ODEFormer."""

from __future__ import annotations
from typing import Any
import numpy as np
from .base_ode_sr_wrapper import BaseODESRWrapper, EquationResult


class ODEFormerWrapper(BaseODESRWrapper):
    method_name = "odeformer"

    def __init__(self, *, state_names: list[str] | None = None, **config: Any):
        super().__init__()
        from Baseline.ODEFormer import ODEFormer, ODEFormerConfig

        self.estimator = ODEFormer(ODEFormerConfig(**config), state_names=state_names)
        self.config = self.estimator.config

    def fit(self, times, trajectories, **kwargs) -> EquationResult:
        self._is_fitted = False
        self._equation_result = EquationResult()
        self._raw_outputs = {}
        self.estimator.reset()
        if kwargs:
            raise TypeError(f"unsupported ODEFormer fit overrides: {sorted(kwargs)}")
        if isinstance(times, list) or isinstance(trajectories, list):
            if not isinstance(times, list) or not isinstance(trajectories, list):
                raise TypeError("times and trajectories must both be lists or arrays")
            if len(times) != 1 or len(trajectories) != 1:
                raise NotImplementedError("ODEFormer fits exactly one trajectory")
            times, trajectories = times[0], trajectories[0]
        model = self.estimator.fit(times, trajectories)
        self.config = self.estimator.config
        equation = self.estimator.equation()
        return self._mark_fitted(
            best=equation,
            candidates=[equation],
            status="success",
            metadata={
                "selected_model": model,
                "search_summary": self.estimator.search_summary,
            },
            raw_outputs={
                "selected_model": model,
                "search_results": self.estimator.search_results,
            },
        )

    def predict(self, times, y0=None, **kwargs) -> np.ndarray | list[np.ndarray]:
        if kwargs:
            raise TypeError(
                f"unsupported ODEFormer prediction options: {sorted(kwargs)}"
            )
        if not self.is_fitted():
            raise RuntimeError("ODEFormerWrapper must be fitted before predict")
        if isinstance(times, list):
            if not isinstance(y0, list) or len(times) != len(y0):
                raise ValueError("list prediction requires matching time/y0 lists")
            return [self.estimator.predict(t, initial) for t, initial in zip(times, y0)]
        if y0 is None:
            raise ValueError("prediction requires y0")
        return self.estimator.predict(times, y0)
