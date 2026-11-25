"""Unified wrapper for the independent Reactive SINDy baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from Baseline.ReactiveSINDy import (
    ReactiveSINDy,
    ReactiveSINDyConfig,
    ReactiveSINDyFitResult,
)
from .base_ode_sr_wrapper import BaseODESRWrapper, EquationResult


class ReactiveSINDyWrapper(BaseODESRWrapper):
    """Adapt the array-only Reactive SINDy core to the shared ODE-SR API."""

    method_name = "reactive_sindy"

    def __init__(
        self,
        *,
        state_names: list[str],
        library_mode: str = "enumerated",
        max_reaction_order: int = 2,
        include_fission: bool = True,
        include_double_conversion: bool = True,
        deduplicate_reactions: bool = True,
        derivative_method: str = "numpy_gradient",
        derivative_edge_order: int = 2,
        short_trajectory_edge_order: int = 1,
        drop_boundary_points: bool = True,
        normalize_library: bool = True,
        library_normalization: str = "column_l2",
        library_zero_norm_threshold: float = 1.0e-14,
        library_scale_fit: str = "full_effective_trajectory",
        library_scale_refit: str = "reuse_full_effective_scale",
        derivative_block_policy: str = "differentiate_full_then_split_effective",
        optimize_hyperparams: bool = True,
        hyper_opt_eval_fraction: float | None = 0.3,
        hyper_opt_split: str = "chronological",
        hyper_opt_train_steps_rounding: str = "python_round_half_to_even",
        alpha_scale_grid: list[float] | tuple[float, ...] | None = None,
        l1_ratio_grid: list[float] | tuple[float, ...] | None = None,
        alpha: float | None = None,
        l1_ratio: float = 1.0,
        alpha_max_normalizer: str = "num_scalar_rows",
        objective_normalizer: str = "num_time_steps",
        final_refit_derivative_blocks: str = "full_effective_trajectory",
        rate_threshold: float = 1.0e-8,
        relative_rate_threshold: float = 1.0e-4,
        sparsity_selection_tolerance: float = 0.05,
        optimizer_method: str = "L-BFGS-B",
        optimizer_max_iter: int = 5000,
        optimizer_tol: float = 1.0e-10,
        ode_integrator: str = "solve_ivp",
        ode_method: str = "RK45",
        ode_rtol: float = 1.0e-3,
        ode_atol: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.state_names = [str(name) for name in state_names]
        self.config = ReactiveSINDyConfig(
            library_mode=str(library_mode),
            max_reaction_order=int(max_reaction_order),
            include_fission=bool(include_fission),
            include_double_conversion=bool(include_double_conversion),
            deduplicate_reactions=bool(deduplicate_reactions),
            derivative_method=str(derivative_method),
            derivative_edge_order=int(derivative_edge_order),
            short_trajectory_edge_order=int(short_trajectory_edge_order),
            drop_boundary_points=bool(drop_boundary_points),
            normalize_library=bool(normalize_library),
            library_normalization=str(library_normalization),
            library_zero_norm_threshold=float(library_zero_norm_threshold),
            library_scale_fit=str(library_scale_fit),
            library_scale_refit=str(library_scale_refit),
            derivative_block_policy=str(derivative_block_policy),
            optimize_hyperparams=bool(optimize_hyperparams),
            hyper_opt_eval_fraction=(
                None
                if hyper_opt_eval_fraction is None
                else float(hyper_opt_eval_fraction)
            ),
            hyper_opt_split=str(hyper_opt_split),
            hyper_opt_train_steps_rounding=str(hyper_opt_train_steps_rounding),
            alpha_scale_grid=(
                ReactiveSINDyConfig.alpha_scale_grid
                if alpha_scale_grid is None
                else tuple(float(value) for value in alpha_scale_grid)
            ),
            l1_ratio_grid=(
                ReactiveSINDyConfig.l1_ratio_grid
                if l1_ratio_grid is None
                else tuple(float(value) for value in l1_ratio_grid)
            ),
            alpha=None if alpha is None else float(alpha),
            l1_ratio=float(l1_ratio),
            alpha_max_normalizer=str(alpha_max_normalizer),
            objective_normalizer=str(objective_normalizer),
            final_refit_derivative_blocks=str(final_refit_derivative_blocks),
            rate_threshold=float(rate_threshold),
            relative_rate_threshold=float(relative_rate_threshold),
            sparsity_selection_tolerance=float(sparsity_selection_tolerance),
            optimizer_method=str(optimizer_method),
            optimizer_max_iter=int(optimizer_max_iter),
            optimizer_tol=float(optimizer_tol),
            ode_integrator=str(ode_integrator),
            ode_method=str(ode_method),
            ode_rtol=float(ode_rtol),
            ode_atol=float(ode_atol),
        )
        self.estimator = ReactiveSINDy(self.state_names, self.config)
        self.fit_result: ReactiveSINDyFitResult | None = None

    def fit(
        self,
        times: np.ndarray | list[np.ndarray],
        trajectories: np.ndarray | list[np.ndarray],
        **kwargs: Any,
    ) -> EquationResult:
        if kwargs:
            keys = ", ".join(sorted(kwargs))
            raise TypeError(
                f"ReactiveSINDyWrapper.fit() does not accept overrides: {keys}"
            )
        times, trajectories = self.validate_trajectories(times, trajectories)
        if isinstance(times, list) or isinstance(trajectories, list):
            if not isinstance(times, list) or not isinstance(trajectories, list):
                raise TypeError(
                    "times and trajectories must both be lists or both be arrays"
                )
            if len(times) != 1 or len(trajectories) != 1:
                raise NotImplementedError(
                    "the release Reactive SINDy baseline fits one trajectory"
                )
            times, trajectories = times[0], trajectories[0]

        result = self.estimator.fit(
            np.asarray(times, dtype=float), np.asarray(trajectories, dtype=float)
        )
        self.fit_result = result
        selected_reactions = result.selected_reactions_payload()
        metadata = {
            "state_names": self.state_names,
            "selected_model": self.estimator.to_model(),
            "algorithm_contract": self.config.to_dict(),
            "num_candidate_reactions": len(result.reactions),
            "num_active_reactions": len(selected_reactions),
            "selected_reactions": selected_reactions,
            "best_substituted": list(result.equation_parts),
            "column_scales": result.column_scales.tolist(),
            "hyperparameter_records": list(result.hyperparameter_records),
            "selected_alpha": result.selected_alpha,
            "selected_l1_ratio": result.selected_l1_ratio,
            **result.metadata,
        }
        return self._mark_fitted(
            best=result.equation,
            candidates=[result.equation],
            status="success",
            metadata=metadata,
            raw_outputs={
                "rates": result.rates,
                "candidate_library": result.candidate_library_payload(),
                "selected_reactions": selected_reactions,
                "hyperparameter_records": list(result.hyperparameter_records),
            },
        )

    def predict(
        self,
        times: np.ndarray | list[np.ndarray],
        y0: np.ndarray | list[np.ndarray] | None = None,
        **kwargs: Any,
    ) -> np.ndarray | list[np.ndarray]:
        if not self.is_fitted() or self.fit_result is None:
            raise RuntimeError("ReactiveSINDyWrapper must be fitted before predict()")
        unsupported = sorted(set(kwargs) - {"ode_integrator"})
        if unsupported:
            raise TypeError(f"unsupported prediction options: {unsupported}")
        backend = kwargs.get("ode_integrator", self.config.ode_integrator)
        if backend != "solve_ivp":
            raise ValueError(
                "the release Reactive SINDy baseline only supports solve_ivp"
            )

        if isinstance(times, list):
            if y0 is None or not isinstance(y0, list):
                raise TypeError("list-valued times require list-valued y0")
            if len(times) != len(y0):
                raise ValueError("times and y0 list lengths differ")
            return [
                self.estimator.predict(
                    np.asarray(time_grid, dtype=float), np.asarray(initial, dtype=float)
                )
                for time_grid, initial in zip(times, y0)
            ]
        if y0 is None or isinstance(y0, list):
            raise ValueError("array-valued prediction requires one y0 array")
        return self.estimator.predict(
            np.asarray(times, dtype=float), np.asarray(y0, dtype=float)
        )


__all__ = ["ReactiveSINDyWrapper"]
