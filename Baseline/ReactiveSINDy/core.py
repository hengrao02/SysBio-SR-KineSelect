"""Independent Reactive SINDy implementation used by the release baseline.

The algorithm in this module is deliberately array-only: it has no dataset,
annotation, XML, repository, or result-directory access.  Candidate reactions
are enumerated blindly from the observed state dimension.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


class ReactiveSINDyError(RuntimeError):
    """Base error for the released Reactive SINDy implementation."""


class ReactiveSINDyRolloutError(ReactiveSINDyError):
    """Raised when a fitted reaction system cannot be integrated safely."""


@dataclass(frozen=True)
class ReactiveSINDyConfig:
    """Frozen numerical contract for the release Reactive SINDy baseline."""

    library_mode: str = "enumerated"
    max_reaction_order: int = 2
    include_fission: bool = True
    include_double_conversion: bool = True
    deduplicate_reactions: bool = True

    derivative_method: str = "numpy_gradient"
    derivative_edge_order: int = 2
    short_trajectory_edge_order: int = 1
    drop_boundary_points: bool = True
    normalize_library: bool = True
    library_normalization: str = "column_l2"
    library_zero_norm_threshold: float = 1.0e-14
    library_scale_fit: str = "full_effective_trajectory"
    library_scale_refit: str = "reuse_full_effective_scale"
    derivative_block_policy: str = "differentiate_full_then_split_effective"

    optimize_hyperparams: bool = True
    hyper_opt_eval_fraction: float | None = 0.3
    hyper_opt_split: str = "chronological"
    hyper_opt_train_steps_rounding: str = "python_round_half_to_even"
    alpha_scale_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0)
    l1_ratio_grid: tuple[float, ...] = (1.0,)
    alpha: float | None = None
    l1_ratio: float = 1.0
    alpha_max_normalizer: str = "num_scalar_rows"
    objective_normalizer: str = "num_time_steps"
    final_refit_derivative_blocks: str = "full_effective_trajectory"

    rate_threshold: float = 1.0e-8
    relative_rate_threshold: float = 1.0e-4
    sparsity_selection_tolerance: float = 0.05
    optimizer_method: str = "L-BFGS-B"
    optimizer_max_iter: int = 5000
    optimizer_tol: float = 1.0e-10

    ode_integrator: str = "solve_ivp"
    ode_method: str = "RK45"
    ode_rtol: float = 1.0e-3
    ode_atol: float = 1.0e-6

    def __post_init__(self) -> None:
        fixed_values = {
            "library_mode": (self.library_mode, "enumerated"),
            "derivative_method": (self.derivative_method, "numpy_gradient"),
            "library_normalization": (self.library_normalization, "column_l2"),
            "library_scale_fit": (
                self.library_scale_fit,
                "full_effective_trajectory",
            ),
            "library_scale_refit": (
                self.library_scale_refit,
                "reuse_full_effective_scale",
            ),
            "derivative_block_policy": (
                self.derivative_block_policy,
                "differentiate_full_then_split_effective",
            ),
            "hyper_opt_split": (self.hyper_opt_split, "chronological"),
            "hyper_opt_train_steps_rounding": (
                self.hyper_opt_train_steps_rounding,
                "python_round_half_to_even",
            ),
            "alpha_max_normalizer": (self.alpha_max_normalizer, "num_scalar_rows"),
            "objective_normalizer": (self.objective_normalizer, "num_time_steps"),
            "final_refit_derivative_blocks": (
                self.final_refit_derivative_blocks,
                "full_effective_trajectory",
            ),
            "ode_integrator": (self.ode_integrator, "solve_ivp"),
        }
        for key, (actual, expected) in fixed_values.items():
            if actual != expected:
                raise ValueError(f"{key} must be {expected!r}, got {actual!r}")
        if self.max_reaction_order not in {1, 2}:
            raise ValueError("max_reaction_order must be 1 or 2")
        if self.derivative_edge_order != 2 or self.short_trajectory_edge_order != 1:
            raise ValueError(
                "the release derivative contract requires edge orders 2 and 1"
            )
        if self.library_zero_norm_threshold <= 0:
            raise ValueError("library_zero_norm_threshold must be positive")
        if self.optimize_hyperparams:
            fraction = self.hyper_opt_eval_fraction
            if fraction is None or not 0.0 < float(fraction) < 1.0:
                raise ValueError(
                    "hyper_opt_eval_fraction must lie in (0, 1) when optimization is enabled"
                )
            if not self.alpha_scale_grid or not self.l1_ratio_grid:
                raise ValueError("hyperparameter grids must be non-empty")
        if any(value < 0 for value in self.alpha_scale_grid):
            raise ValueError("alpha_scale_grid values must be non-negative")
        if any(not 0.0 <= value <= 1.0 for value in self.l1_ratio_grid):
            raise ValueError("l1_ratio_grid values must lie in [0, 1]")
        if self.alpha is not None and self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio must lie in [0, 1]")
        if self.rate_threshold < 0 or self.relative_rate_threshold < 0:
            raise ValueError("rate thresholds must be non-negative")
        if self.sparsity_selection_tolerance < 0:
            raise ValueError("sparsity_selection_tolerance must be non-negative")
        if self.optimizer_method != "L-BFGS-B":
            raise ValueError("the release optimizer contract requires L-BFGS-B")
        if self.optimizer_max_iter <= 0 or self.optimizer_tol <= 0:
            raise ValueError("optimizer limits and tolerance must be positive")
        if self.ode_rtol <= 0 or self.ode_atol <= 0:
            raise ValueError("ODE tolerances must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["alpha_scale_grid"] = list(self.alpha_scale_grid)
        data["l1_ratio_grid"] = list(self.l1_ratio_grid)
        return data


@dataclass(frozen=True)
class ReactionBasis:
    label: str
    reactants: tuple[int, ...]
    products: tuple[int, ...]
    stoichiometry: tuple[int, ...]

    def propensity(self, trajectory: np.ndarray) -> np.ndarray:
        values = np.ones(trajectory.shape[0], dtype=float)
        for idx in self.reactants:
            values *= trajectory[:, idx]
        return values

    def scalar_propensity(self, state: np.ndarray) -> float:
        value = 1.0
        for idx in self.reactants:
            value *= float(state[idx])
        return value

    def signature(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.reactants, self.stoichiometry

    def payload(
        self, *, candidate_index: int | None = None, rate: float | None = None
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "label": self.label,
            "reactants": [f"x_{idx}" for idx in self.reactants],
            "products": [f"x_{idx}" for idx in self.products],
            "stoichiometry": list(self.stoichiometry),
        }
        if candidate_index is not None:
            data["candidate_index"] = int(candidate_index)
        if rate is not None:
            data["rate"] = float(rate)
        return data


@dataclass(frozen=True)
class ReactiveSINDyFitResult:
    state_names: tuple[str, ...]
    reactions: tuple[ReactionBasis, ...]
    rates: np.ndarray
    equation_parts: tuple[str, ...]
    hyperparameter_records: tuple[dict[str, Any], ...]
    selected_alpha: float
    selected_l1_ratio: float
    column_scales: np.ndarray
    metadata: dict[str, Any]

    @property
    def equation(self) -> str:
        return " | ".join(self.equation_parts)

    def candidate_library_payload(self) -> list[dict[str, Any]]:
        return [
            reaction.payload(candidate_index=index)
            for index, reaction in enumerate(self.reactions)
        ]

    def selected_reactions_payload(self) -> list[dict[str, Any]]:
        return [
            reaction.payload(candidate_index=index, rate=float(rate))
            for index, (reaction, rate) in enumerate(zip(self.reactions, self.rates))
            if rate > 0
        ]


class ReactiveSINDy:
    """Non-negative sparse regression over vector-valued reaction bases."""

    def __init__(
        self,
        state_names: list[str] | tuple[str, ...],
        config: ReactiveSINDyConfig | None = None,
    ) -> None:
        self.state_names = tuple(str(name) for name in state_names)
        if not self.state_names:
            raise ValueError("state_names must be non-empty")
        if len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state_names must be unique")
        self.config = config or ReactiveSINDyConfig()
        self.fit_result: ReactiveSINDyFitResult | None = None

    @staticmethod
    def _format_float(value: float) -> str:
        text = f"{float(value):.12g}"
        return "0" if text == "-0" else text

    @staticmethod
    def _format_reaction(reactants: tuple[int, ...], products: tuple[int, ...]) -> str:
        left = " + ".join(f"x_{idx}" for idx in reactants) if reactants else "0"
        right = " + ".join(f"x_{idx}" for idx in products) if products else "0"
        return f"{left} -> {right}"

    def _make_reaction(
        self, reactants: tuple[int, ...], products: tuple[int, ...]
    ) -> ReactionBasis | None:
        reactants = tuple(sorted(reactants))
        products = tuple(sorted(products))
        stoichiometry = [0] * len(self.state_names)
        for idx in reactants:
            stoichiometry[idx] -= 1
        for idx in products:
            stoichiometry[idx] += 1
        if all(value == 0 for value in stoichiometry):
            return None
        return ReactionBasis(
            label=self._format_reaction(reactants, products),
            reactants=reactants,
            products=products,
            stoichiometry=tuple(stoichiometry),
        )

    def enumerate_library(self) -> tuple[ReactionBasis, ...]:
        n_states = len(self.state_names)
        reactions: list[ReactionBasis] = []

        def append(reactants: tuple[int, ...], products: tuple[int, ...]) -> None:
            reaction = self._make_reaction(reactants, products)
            if reaction is not None:
                reactions.append(reaction)

        for i in range(n_states):
            append((), (i,))
            append((i,), ())
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    append((i,), (j,))
        if self.config.include_fission:
            for i in range(n_states):
                for j in range(n_states):
                    for k in range(j, n_states):
                        append((i,), (j, k))
        if self.config.max_reaction_order >= 2:
            for i in range(n_states):
                for j in range(i, n_states):
                    append((i, j), ())
                    for k in range(n_states):
                        append((i, j), (k,))
                    if self.config.include_double_conversion:
                        for k in range(n_states):
                            for ell in range(k, n_states):
                                append((i, j), (k, ell))

        if not self.config.deduplicate_reactions:
            return tuple(reactions)
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        unique: list[ReactionBasis] = []
        for reaction in reactions:
            signature = reaction.signature()
            if signature not in seen:
                seen.add(signature)
                unique.append(reaction)
        if not unique:
            raise ReactiveSINDyError("candidate reaction library is empty")
        return tuple(unique)

    def _validate_fit_inputs(
        self, times: np.ndarray, trajectory: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, dtype=float).reshape(-1)
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2:
            raise ValueError(f"trajectory must be 2D [T, D], got {trajectory.shape}")
        if len(times) != trajectory.shape[0]:
            raise ValueError("time and trajectory lengths differ")
        if trajectory.shape[1] != len(self.state_names):
            raise ValueError("trajectory state dimension does not match state_names")
        if len(times) < 2:
            raise ValueError("at least two time points are required")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(trajectory)):
            raise ValueError("fit inputs must be finite")
        if not np.all(np.diff(times) > 0):
            raise ValueError("fit times must be strictly increasing")
        return times, trajectory

    def differentiate_block(
        self,
        times: np.ndarray,
        trajectory: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.asarray(times, dtype=float).reshape(-1)
        trajectory = np.asarray(trajectory, dtype=float)
        edge_order = (
            self.config.derivative_edge_order
            if len(times) > 2
            else self.config.short_trajectory_edge_order
        )
        derivative = np.gradient(trajectory, times, axis=0, edge_order=edge_order)
        if self.config.drop_boundary_points and len(times) > 2:
            return times[1:-1], trajectory[1:-1], derivative[1:-1]
        return times, trajectory, derivative

    @staticmethod
    def build_design(
        trajectory: np.ndarray,
        derivative: np.ndarray,
        reactions: tuple[ReactionBasis, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        n_time, n_state = trajectory.shape
        design = np.zeros((n_time * n_state, len(reactions)), dtype=float)
        target = np.asarray(derivative, dtype=float).reshape(n_time * n_state)
        for column, reaction in enumerate(reactions):
            propensity = reaction.propensity(trajectory)
            stoichiometry = np.asarray(reaction.stoichiometry, dtype=float)
            design[:, column] = (propensity[:, None] * stoichiometry[None, :]).reshape(
                -1
            )
        return design, target

    def fit_column_scales(self, design: np.ndarray) -> np.ndarray:
        if not self.config.normalize_library:
            return np.ones(design.shape[1], dtype=float)
        scales = np.linalg.norm(design, axis=0)
        return np.where(scales <= self.config.library_zero_norm_threshold, 1.0, scales)

    @staticmethod
    def apply_column_scales(design: np.ndarray, scales: np.ndarray) -> np.ndarray:
        return (
            np.asarray(design, dtype=float) / np.asarray(scales, dtype=float)[None, :]
        )

    @staticmethod
    def alpha_max(design: np.ndarray, target: np.ndarray) -> float:
        if design.size == 0:
            return 1.0
        # Match the reference profile: alpha_max is normalized
        # by scalar regression rows, while the solver objective remains
        # normalized by effective time steps.
        value = float(np.max(np.abs(design.T @ target)) / max(1, int(target.size)))
        return value if np.isfinite(value) and value > 0 else 1.0

    def rate_cutoff(self, rates: np.ndarray) -> float:
        rates = np.asarray(rates, dtype=float)
        max_rate = float(np.max(np.abs(rates))) if rates.size else 0.0
        return max(
            self.config.rate_threshold, self.config.relative_rate_threshold * max_rate
        )

    def threshold_rates(self, rates: np.ndarray) -> np.ndarray:
        rates = np.asarray(rates, dtype=float)
        return np.where(rates > self.rate_cutoff(rates), rates, 0.0)

    def solve_rates(
        self,
        design: np.ndarray,
        target: np.ndarray,
        *,
        alpha: float,
        l1_ratio: float,
        n_time_steps: int,
    ) -> tuple[np.ndarray, Any]:
        n_features = design.shape[1]
        normalizer = max(1, int(n_time_steps))
        l1_penalty = float(alpha) * float(l1_ratio)
        l2_penalty = float(alpha) * (1.0 - float(l1_ratio))

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            residual = design @ coefficients - target
            loss = 0.5 * float(residual @ residual) / normalizer
            value = (
                loss
                + l1_penalty * float(np.sum(coefficients))
                + 0.5 * l2_penalty * float(coefficients @ coefficients)
            )
            gradient = (
                (design.T @ residual) / normalizer
                + l1_penalty
                + l2_penalty * coefficients
            )
            return value, gradient

        result = minimize(
            lambda coefficients: objective(coefficients)[0],
            np.zeros(n_features, dtype=float),
            method=self.config.optimizer_method,
            jac=lambda coefficients: objective(coefficients)[1],
            bounds=[(0.0, None)] * n_features,
            tol=self.config.optimizer_tol,
            options={"maxiter": self.config.optimizer_max_iter},
        )
        return np.asarray(result.x, dtype=float), result

    def _prepare_blocks(
        self,
        times: np.ndarray,
        trajectory: np.ndarray,
        reactions: tuple[ReactionBasis, ...],
    ) -> dict[str, Any]:
        # The historical reference results were produced by differentiating
        # the complete default trajectory once, dropping the two global
        # boundary samples, and only then making the chronological
        # train/validation split.  Preserve that exact execution order here;
        # the perturbed trajectory remains evaluation-only.
        effective_times, effective_x, effective_dx = self.differentiate_block(
            times, trajectory
        )
        design, target = self.build_design(effective_x, effective_dx, reactions)
        effective_steps = len(effective_times)
        split_index = effective_steps
        if self.config.optimize_hyperparams and effective_steps >= 3:
            fraction = float(self.config.hyper_opt_eval_fraction)
            proposed = int(round(effective_steps * (1.0 - fraction)))
            if 0 < proposed < effective_steps:
                split_index = min(max(proposed, 1), effective_steps - 1)

        n_states = len(self.state_names)
        split_row = split_index * n_states
        if split_index == effective_steps:
            train_design = design
            train_target = target
            val_design = design
            val_target = target
            validation_times = effective_times
        else:
            train_design = design[:split_row]
            train_target = target[:split_row]
            val_design = design[split_row:]
            val_target = target[split_row:]
            validation_times = effective_times[split_index:]
        return {
            "train_design": train_design,
            "train_target": train_target,
            "validation_design": val_design,
            "validation_target": val_target,
            "full_design": design,
            "full_target": target,
            "train_steps": split_index,
            "validation_steps": len(validation_times),
            "effective_split_index": split_index,
            "full_effective_steps": effective_steps,
            "train_time_range": [
                float(effective_times[0]),
                float(effective_times[split_index - 1]),
            ],
            "validation_time_range": [
                float(validation_times[0]),
                float(validation_times[-1]),
            ],
        }

    def _equation_parts(
        self, rates: np.ndarray, reactions: tuple[ReactionBasis, ...]
    ) -> tuple[str, ...]:
        cutoff = self.rate_cutoff(rates)
        coefficients: list[dict[tuple[int, ...], float]] = [
            dict() for _ in self.state_names
        ]
        for rate, reaction in zip(rates, reactions):
            if rate <= 0:
                continue
            for state_index, stoichiometric_coefficient in enumerate(
                reaction.stoichiometry
            ):
                if stoichiometric_coefficient:
                    key = reaction.reactants
                    coefficients[state_index][key] = coefficients[state_index].get(
                        key, 0.0
                    ) + float(stoichiometric_coefficient) * float(rate)

        parts: list[str] = []
        for state_coefficients in coefficients:
            terms: list[str] = []
            for reactants, coefficient in sorted(
                state_coefficients.items(), key=lambda item: (len(item[0]), item[0])
            ):
                if abs(coefficient) <= cutoff:
                    continue
                magnitude = abs(coefficient)
                factors = [f"x_{idx}" for idx in reactants]
                if factors:
                    prefix = (
                        ""
                        if abs(magnitude - 1.0) <= 1.0e-12
                        else f"{self._format_float(magnitude)}*"
                    )
                    term = f"{prefix}{'*'.join(factors)}"
                else:
                    term = self._format_float(magnitude)
                terms.append(f"{'+' if coefficient > 0 else '-'}{term}")
            if not terms:
                parts.append("0")
            else:
                expression = " ".join(terms)
                parts.append(
                    expression[1:].strip() if expression.startswith("+") else expression
                )
        return tuple(parts)

    def fit(self, times: np.ndarray, trajectory: np.ndarray) -> ReactiveSINDyFitResult:
        times, trajectory = self._validate_fit_inputs(times, trajectory)
        reactions = self.enumerate_library()
        blocks = self._prepare_blocks(times, trajectory, reactions)

        # The reference profile normalized the complete effective design before the
        # chronological split.  This is intentionally not a train-only scale.
        scales = self.fit_column_scales(blocks["full_design"])
        train_design = self.apply_column_scales(blocks["train_design"], scales)
        validation_design = self.apply_column_scales(
            blocks["validation_design"], scales
        )
        train_target = blocks["train_target"]
        validation_target = blocks["validation_target"]

        if self.config.optimize_hyperparams:
            alpha_base = self.alpha_max(train_design, train_target)
            grid = [
                (scale, l1_ratio)
                for scale in self.config.alpha_scale_grid
                for l1_ratio in self.config.l1_ratio_grid
            ]
            candidates = [
                (index, float(scale * alpha_base), float(l1_ratio))
                for index, (scale, l1_ratio) in enumerate(grid)
            ]
        else:
            alpha_base = None
            candidates = [
                (
                    0,
                    float(0.0 if self.config.alpha is None else self.config.alpha),
                    float(self.config.l1_ratio),
                )
            ]

        records: list[dict[str, Any]] = []
        for candidate_index, alpha, l1_ratio in candidates:
            normalized_rates, solver_result = self.solve_rates(
                train_design,
                train_target,
                alpha=alpha,
                l1_ratio=l1_ratio,
                n_time_steps=blocks["train_steps"],
            )
            residual = validation_design @ normalized_rates - validation_target
            validation_mse = (
                float(np.mean(residual**2))
                if solver_result.success and residual.size
                else float("inf")
            )
            physical_rates = normalized_rates / scales
            active_count = int(np.count_nonzero(self.threshold_rates(physical_rates)))
            records.append(
                {
                    "candidate_index": candidate_index,
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "validation_mse": validation_mse,
                    "active_count": active_count,
                    "solver_success": bool(solver_result.success),
                    "solver_message": str(solver_result.message),
                    "solver_nit": getattr(solver_result, "nit", None),
                }
            )

        finite_records = [
            record for record in records if np.isfinite(record["validation_mse"])
        ]
        if not finite_records:
            raise ReactiveSINDyError(
                "no hyperparameter candidate converged successfully"
            )
        best_mse = min(float(record["validation_mse"]) for record in finite_records)
        eligible = [
            record
            for record in finite_records
            if float(record["validation_mse"])
            <= best_mse * (1.0 + self.config.sparsity_selection_tolerance)
        ]
        selected_record = min(
            eligible,
            key=lambda record: (
                int(record["active_count"]),
                float(record["validation_mse"]),
                float(record["alpha"]),
                float(record["l1_ratio"]),
            ),
        )

        refit_design_raw = blocks["full_design"]
        refit_target = blocks["full_target"]
        refit_steps = int(blocks["full_effective_steps"])
        refit_blocks = ["full_effective_trajectory"]
        refit_design = self.apply_column_scales(refit_design_raw, scales)
        normalized_rates, final_solver = self.solve_rates(
            refit_design,
            refit_target,
            alpha=float(selected_record["alpha"]),
            l1_ratio=float(selected_record["l1_ratio"]),
            n_time_steps=refit_steps,
        )
        if not final_solver.success:
            raise ReactiveSINDyError(
                f"final optimization failed: {final_solver.message}"
            )
        rates = self.threshold_rates(normalized_rates / scales)

        metadata = {
            "alpha_base": alpha_base,
            "effective_split_index": int(blocks["effective_split_index"]),
            "train_time_range": blocks["train_time_range"],
            "validation_time_range": blocks["validation_time_range"],
            "train_effective_steps": int(blocks["train_steps"]),
            "validation_effective_steps": int(blocks["validation_steps"]),
            "final_refit_blocks": refit_blocks,
            "final_refit_effective_steps": refit_steps,
            "final_refit_scalar_rows": int(refit_target.size),
            "selected_candidate_index": int(selected_record["candidate_index"]),
            "final_solver_success": bool(final_solver.success),
            "final_solver_message": str(final_solver.message),
            "final_solver_nit": getattr(final_solver, "nit", None),
            "effective_rate_threshold": self.rate_cutoff(rates),
        }
        result = ReactiveSINDyFitResult(
            state_names=self.state_names,
            reactions=reactions,
            rates=rates,
            equation_parts=self._equation_parts(rates, reactions),
            hyperparameter_records=tuple(records),
            selected_alpha=float(selected_record["alpha"]),
            selected_l1_ratio=float(selected_record["l1_ratio"]),
            column_scales=scales,
            metadata=metadata,
        )
        self.fit_result = result
        return result

    def to_model(self):
        if self.fit_result is None:
            raise ReactiveSINDyError("fit must be called before model export")
        result = self.fit_result
        return {
            "model_format": "sysbiosr-model-v2",
            "method": "reactive_sindy",
            "state_names": list(self.state_names),
            "config": self.config.to_dict(),
            "equation": result.equation,
            "equation_parts": list(result.equation_parts),
            "reactions": result.selected_reactions_payload(),
            "rates": result.rates.tolist(),
        }

    @classmethod
    def from_model(cls, model):
        if (
            not isinstance(model, dict)
            or model.get("model_format") != "sysbiosr-model-v2"
            or model.get("method") != "reactive_sindy"
        ):
            raise ValueError(
                "unsupported model format. Expected a dictionary with "
                "method='reactive_sindy' and model_format='sysbiosr-model-v2'."
            )
        names = model["state_names"]
        if not isinstance(names, list) or any(
            not isinstance(n, str) or not n for n in names
        ):
            raise ValueError("invalid state names")
        estimator = cls(names, ReactiveSINDyConfig(**model["config"]))
        full_rates = np.asarray(model["rates"], dtype=float)
        if (
            full_rates.ndim != 1
            or not np.isfinite(full_rates).all()
            or (full_rates < 0).any()
        ):
            raise ValueError("invalid saved reaction rates")
        reactions, rates, indices = [], [], []
        for row in model["reactions"]:

            def states(values):
                result = []
                for value in values:
                    if (
                        not isinstance(value, str)
                        or not value.startswith("x_")
                        or not value[2:].isdigit()
                    ):
                        raise ValueError("invalid saved reaction state")
                    i = int(value[2:])
                    if not 0 <= i < len(names):
                        raise ValueError("saved reaction state out of range")
                    result.append(i)
                return tuple(result)

            reactants, products = states(row["reactants"]), states(row["products"])
            stoich = tuple(
                products.count(i) - reactants.count(i) for i in range(len(names))
            )
            if list(stoich) != row["stoichiometry"]:
                raise ValueError("reaction stoichiometry mismatch")
            i = row["candidate_index"]
            rate = float(row["rate"])
            if (
                type(i) is not int
                or not 0 <= i < len(full_rates)
                or rate <= 0
                or full_rates[i] != rate
            ):
                raise ValueError("saved rate binding mismatch")
            indices.append(i)
            rates.append(rate)
            reactions.append(ReactionBasis(row["label"], reactants, products, stoich))
        if indices != list(np.flatnonzero(full_rates > 0)):
            raise ValueError("saved reaction order or coverage mismatch")
        rates = np.asarray(rates, dtype=float)
        parts = estimator._equation_parts(rates, tuple(reactions))
        if (
            list(parts) != model["equation_parts"]
            or " | ".join(parts) != model["equation"]
        ):
            raise ValueError("saved equation mismatch")
        estimator.fit_result = ReactiveSINDyFitResult(
            tuple(names),
            tuple(reactions),
            rates,
            parts,
            (),
            0.0,
            0.0,
            np.ones(len(rates)),
            {},
        )
        return estimator

    def rhs(self, _time: float, state: np.ndarray) -> np.ndarray:
        if self.fit_result is None:
            raise ReactiveSINDyError("fit must be called before rhs")
        state = np.asarray(state, dtype=float).reshape(-1)
        derivative = np.zeros(len(self.state_names), dtype=float)
        for rate, reaction in zip(self.fit_result.rates, self.fit_result.reactions):
            if rate > 0:
                derivative += (
                    float(rate)
                    * reaction.scalar_propensity(state)
                    * np.asarray(reaction.stoichiometry, dtype=float)
                )
        return derivative

    def predict(self, times: np.ndarray, y0: np.ndarray) -> np.ndarray:
        if self.fit_result is None:
            raise ReactiveSINDyError("fit must be called before predict")
        times = np.asarray(times, dtype=float).reshape(-1)
        y0 = np.asarray(y0, dtype=float).reshape(-1)
        if not len(times):
            raise ValueError("prediction times must be non-empty")
        if len(y0) != len(self.state_names):
            raise ValueError("y0 dimension does not match state_names")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(y0)):
            raise ValueError("prediction inputs must be finite")
        order = np.argsort(times, kind="stable")
        sorted_times = times[order]
        if len(sorted_times) > 1 and not np.all(np.diff(sorted_times) > 0):
            raise ValueError("prediction times must be unique")
        if len(sorted_times) == 1:
            return y0.reshape(1, -1)

        with warnings.catch_warnings(record=True) as caught_warnings:
            solution = solve_ivp(
                self.rhs,
                (float(sorted_times[0]), float(sorted_times[-1])),
                y0,
                t_eval=sorted_times,
                method=self.config.ode_method,
                rtol=self.config.ode_rtol,
                atol=self.config.ode_atol,
            )
        trajectory = np.asarray(solution.y.T, dtype=float)
        if not solution.success:
            raise ReactiveSINDyRolloutError(str(solution.message))
        if caught_warnings:
            messages = "; ".join(str(item.message) for item in caught_warnings)
            raise ReactiveSINDyRolloutError(f"solve_ivp emitted warnings: {messages}")
        if trajectory.shape != (len(sorted_times), len(self.state_names)):
            raise ReactiveSINDyRolloutError(
                f"unexpected rollout shape {trajectory.shape}"
            )
        if not np.all(np.isfinite(trajectory)):
            raise ReactiveSINDyRolloutError("rollout contains non-finite values")
        inverse_order = np.argsort(order, kind="stable")
        return trajectory[inverse_order]


__all__ = [
    "ReactionBasis",
    "ReactiveSINDy",
    "ReactiveSINDyConfig",
    "ReactiveSINDyError",
    "ReactiveSINDyFitResult",
    "ReactiveSINDyRolloutError",
]
