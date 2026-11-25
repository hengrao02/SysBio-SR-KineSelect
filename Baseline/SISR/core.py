"""Independent SISR-NNLS/Pareto baseline (uniform-elitism R2-R12)."""

from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import time
from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from .reaction_mechanism_generator import (
    NNLSReactionMechanismGenerator,
    reaction_system,
)


@dataclass(frozen=True)
class SISRConfig:
    order: int = 2
    include_bias: bool = True
    max_stoichiometric_ratio: int = 2
    max_ratio: float = 5.0
    min_rxns_per_mech: int = 2
    max_rxns_per_mech: int = 12
    island_pop: int = 2000
    island_generations: int = 20
    island_elitism: float = 0.1
    mutation_prob: float = 0.1
    random_seed: int = 42
    verbosity: int = 1
    pareto_rollout_rtol: float = 1e-6
    pareto_rollout_atol: float = 1e-9
    pareto_alpha: float = 1.05
    pareto_eps: float = 1e-12
    rollout_timeout: float | None = None

    def __post_init__(self):
        for key in (
            "order",
            "max_stoichiometric_ratio",
            "min_rxns_per_mech",
            "max_rxns_per_mech",
            "island_pop",
            "island_generations",
            "random_seed",
            "verbosity",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        if self.order < 1 or self.max_stoichiometric_ratio < 1:
            raise ValueError("order and max_stoichiometric_ratio must be positive")
        if not 1 <= self.min_rxns_per_mech <= self.max_rxns_per_mech:
            raise ValueError("require 1 <= min_rxns_per_mech <= max_rxns_per_mech")
        if self.island_pop < 2 or self.island_generations < 1:
            raise ValueError("island_pop >= 2 and island_generations >= 1 are required")
        if not isinstance(self.include_bias, bool):
            raise ValueError("include_bias must be boolean")
        for key in (
            "max_ratio",
            "island_elitism",
            "mutation_prob",
            "pareto_rollout_rtol",
            "pareto_rollout_atol",
            "pareto_alpha",
            "pareto_eps",
        ):
            if isinstance(getattr(self, key), bool) or not math.isfinite(
                float(getattr(self, key))
            ):
                raise ValueError(f"{key} must be finite")
        if not 0 < self.island_elitism <= 1 or not 0 <= self.mutation_prob <= 1:
            raise ValueError(
                "island_elitism must be in (0, 1] and mutation_prob in [0, 1]"
            )
        if self.max_ratio < 1 or self.pareto_alpha < 1 or self.pareto_eps < 0:
            raise ValueError(
                "max_ratio/pareto_alpha must be >= 1; pareto_eps must be >= 0"
            )
        if min(self.pareto_rollout_rtol, self.pareto_rollout_atol) <= 0:
            raise ValueError("rollout tolerances must be positive")
        if (
            self.random_seed < 0
            or self.random_seed + self.max_rxns_per_mech * 1000 + 7919 >= 2**32
        ):
            raise ValueError("derived island seeds must fit uint32")
        if self.verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1 or 2")
        if self.rollout_timeout is not None and (
            not math.isfinite(self.rollout_timeout) or self.rollout_timeout <= 0
        ):
            raise ValueError("rollout_timeout must be null or positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SISRFitResult:
    status: str
    equation: str | None
    equation_parts: list[str]
    metadata: dict[str, Any]
    model: dict[str, Any] | None
    front: dict[str, Any]


class SISR:
    def __init__(
        self, config: SISRConfig | None = None, *, state_names: list[str] | None = None
    ):
        self.config = config or SISRConfig()
        for key, value in self.config.to_dict().items():
            setattr(self, key, value)
        self.state_names = None if state_names is None else list(state_names)
        self.fit_result = None
        self._reaction_mechanism = None
        self._rate_constants = None
        self._dim = 0
        self.front_payload = {
            "candidates": [],
            "pareto_front": [],
            "selected_candidate_id": None,
        }
        self.rollout_failures = []

    def _generator(self, *, size, population, elitism, seed):
        return NNLSReactionMechanismGenerator(
            order=self.order,
            include_bias=self.include_bias,
            max_ratio=self.max_ratio,
            max_stoichiometric_ratio=self.max_stoichiometric_ratio,
            num_generations=self.island_generations,
            num_mech_per_gen=population,
            min_rxns_per_mech=size,
            max_rxns_per_mech=size,
            from_previous_generation=elitism,
            mutation_prob=self.mutation_prob,
            verbosity=self.verbosity,
            seed=seed,
        )

    def fit(self, times: np.ndarray, trajectory: np.ndarray) -> SISRFitResult:
        self.fit_result = None
        self._reaction_mechanism = self._rate_constants = None
        self.front_payload = {
            "candidates": [],
            "pareto_front": [],
            "selected_candidate_id": None,
        }
        self.rollout_failures = []
        t = np.asarray(times, dtype=float)
        y = np.asarray(trajectory, dtype=float)
        if (
            t.ndim != 1
            or y.ndim != 2
            or y.shape[0] != t.size
            or t.size < 2
            or y.shape[1] < 1
        ):
            raise ValueError(
                "fit requires times [T], trajectory [T,D], T >= 2 and D >= 1"
            )
        if (
            not np.all(np.isfinite(t))
            or not np.all(np.isfinite(y))
            or not np.all(np.diff(t) > 0)
        ):
            raise ValueError("fit inputs must be finite and times strictly increasing")
        self._dim = y.shape[1]
        names = self.state_names or [f"x_{i}" for i in range(self._dim)]
        if (
            not all(isinstance(name, str) for name in names)
            or len(names) != self._dim
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "state_names must be unique and match trajectory dimension"
            )
        self.state_names = names
        probe = self._generator(
            size=self.min_rxns_per_mech,
            population=2,
            elitism=1.0,
            seed=self.random_seed,
        )
        probe.N_TARGETS = self._dim
        library_size = len(probe._generate_reactions())
        candidates, records = [], []
        for size in range(self.min_rxns_per_mech, self.max_rxns_per_mech + 1):
            record = {"island_n_rxns": size, "base_elitism": self.island_elitism}
            records.append(record)
            if size > library_size:
                record.update(status="skipped", reason="reaction_library_too_small")
                continue
            population = (
                min(self.island_pop, math.comb(library_size, size))
                if library_size < 30
                else self.island_pop
            )
            if population < 2:
                record.update(
                    status="skipped", reason="fewer_than_two_unique_mechanisms"
                )
                continue
            elitism = min(1.0, max(self.island_elitism, 2.0 / population))
            seed = self.random_seed + size * 1000 + 7919
            record.update(population=population, effective_elitism=elitism, seed=seed)
            generator = self._generator(
                size=size, population=population, elitism=elitism, seed=seed
            )
            started = time.monotonic()
            try:
                generator.fit(y, t)
            except (RuntimeError, IndexError, ValueError) as exc:
                record.update(
                    status="failed", reason=type(exc).__name__, message=str(exc)
                )
            else:
                for entry in generator.best_mechs:
                    candidates.append([*entry, size])
                record.update(
                    status="success", num_candidates=len(generator.best_mechs)
                )
            finally:
                record["elapsed_seconds"] = time.monotonic() - started
        selected = self._pareto_select_from_candidates(
            candidates, t, y, probe._construct_matrix, reaction_system
        )
        derivatives = np.gradient(y, t, axis=0)
        derivative_scales = np.max(np.abs(derivatives), axis=0)
        zero_columns = np.flatnonzero(derivative_scales == 0).tolist()
        concentration_scales = np.max(np.abs(y), axis=0)
        concentration_scales[concentration_scales < 1e-14] = 1.0
        metadata = {
            "state_names": names,
            "model_format": "sysbiosr-model-v2",
            "algorithm_contract": self.config.to_dict(),
            "islands": records,
            "num_candidate_reactions": library_size,
            "zero_derivative_columns": zero_columns,
            "derivative_scales": np.where(
                derivative_scales == 0, 1.0, derivative_scales
            ).tolist(),
            "concentration_scales": concentration_scales.tolist(),
            "rollout_failures": self.rollout_failures,
        }
        model = None
        parts = []
        if selected is None:
            timed_out = bool(self.rollout_failures) and all(
                r["reason"] == "TimeoutError" for r in self.rollout_failures
            )
            metadata["failure_reason"] = (
                "timeout" if timed_out else "no_valid_candidate"
            )
            status = "timeout" if timed_out else "no_solution"
        else:
            (
                self._reaction_mechanism,
                self._rate_constants,
                mse,
                scaled_mse,
                selected_meta,
            ) = selected
            parts = self._format_equations()
            metadata.update(
                selected_meta,
                mse=mse,
                scaled_conc_mse=scaled_mse,
                reaction_mechanism=[r.tolist() for r in self._reaction_mechanism],
                rate_constants=self._rate_constants.tolist(),
                num_reactions=len(self._reaction_mechanism),
            )
            model = {
                "schema_version": 1,
                "method": "sisr",
                "state_names": names,
                "reaction_mechanism": metadata["reaction_mechanism"],
                "rate_constants": metadata["rate_constants"],
                "config": self.config.to_dict(),
                "model_format": "sysbiosr-model-v2",
                "derivative_scales": metadata["derivative_scales"],
                "concentration_scales": metadata["concentration_scales"],
            }
            status = "success"
        metadata["best_substituted"] = parts
        self.front_payload["islands"] = records
        self.front_payload["rollout_failures"] = self.rollout_failures
        self.fit_result = SISRFitResult(
            status,
            " | ".join(parts) if parts else None,
            parts,
            metadata,
            model,
            self.front_payload,
        )
        return self.fit_result

    def _integrate(self, times, y0, mechanism, rates):
        deadline = (
            None
            if self.rollout_timeout is None
            else time.monotonic() + self.rollout_timeout
        )

        def rhs(t, y):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("SISR rollout exceeded rollout_timeout")
            return reaction_system(t, y, mechanism, rates)

        return solve_ivp(
            rhs,
            [float(times[0]), float(times[-1])],
            y0,
            t_eval=times,
            method="LSODA",
            rtol=self.pareto_rollout_rtol,
            atol=self.pareto_rollout_atol,
            min_step=1e-12,
        )

    def predict(self, times: np.ndarray, y0: np.ndarray) -> np.ndarray:
        if self._reaction_mechanism is None:
            raise RuntimeError("SISR must be fitted or restored before prediction")
        t, initial = np.asarray(times, dtype=float), np.asarray(y0, dtype=float)
        if t.ndim != 1 or len(t) < 1 or initial.shape != (self._dim,):
            raise ValueError(
                "prediction requires a time vector and y0 matching state dimension"
            )
        if (
            not np.all(np.isfinite(t))
            or not np.all(np.isfinite(initial))
            or not np.all(np.diff(t) > 0)
        ):
            raise ValueError(
                "prediction inputs must be finite and times strictly increasing"
            )
        if len(t) == 1:
            return initial[None, :].copy()
        sol = self._integrate(
            t, initial, self._reaction_mechanism, self._rate_constants
        )
        if (
            sol.status != 0
            or sol.y.T.shape != (len(t), self._dim)
            or not np.all(np.isfinite(sol.y))
        ):
            raise RuntimeError(f"SISR rollout failed: {sol.message}")
        return sol.y.T

    @classmethod
    def from_model(cls, model: dict[str, Any]) -> SISR:
        if model.get("schema_version") != 1 or model.get("method") != "sisr":
            raise ValueError("unsupported SISR model format")
        names = model.get("state_names")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("model requires unique state_names")
        estimator = cls(SISRConfig(**model["config"]), state_names=names)
        if model.get("model_format") != "sysbiosr-model-v2":
            raise ValueError(
                "unsupported model format. Expected model_format='sysbiosr-model-v2'."
            )
        matrix = np.asarray(model["reaction_mechanism"], dtype=float)
        rates = np.asarray(model["rate_constants"], dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 2 * len(names) or matrix.shape[0] < 1:
            raise ValueError("invalid reaction matrix shape")
        if (
            not np.all(np.isfinite(matrix))
            or np.any(matrix < 0)
            or np.any(matrix != np.floor(matrix))
        ):
            raise ValueError("reaction matrix must contain nonnegative integers")
        if (
            rates.shape != (len(matrix),)
            or not np.all(np.isfinite(rates))
            or np.any(rates < 0)
        ):
            raise ValueError("invalid rate constants")
        estimator._dim = len(names)
        estimator._reaction_mechanism = [r.astype(int) for r in matrix]
        estimator._rate_constants = rates.copy()
        return estimator

    @staticmethod
    def _compute_complexity(
        reaction_mechanism: list[np.ndarray],
        rate_constants: np.ndarray,
        dim: int,
    ) -> int:
        """SISR-style expression-tree node count from the reaction matrix.

        Follows the formula validated against paper Fig. 2.

        Only active reactions (k > 1e-14) are counted.
        """
        total = 0
        term_counts = [0] * dim

        for reaction, k in zip(reaction_mechanism, rate_constants):
            if k < 1e-14:
                continue

            reactants = reaction[:dim]
            products = reaction[dim:]
            stoich = products.astype(int) - reactants.astype(int)
            O_r = int(np.sum(np.abs(reactants)))

            for i in range(dim):
                s_ir = int(stoich[i])
                if s_ir == 0:
                    continue
                a_ir = abs(s_ir)
                stoich_nodes = 1 if a_ir == 1 else a_ir + 1
                total += 1 + stoich_nodes + 1 + O_r
                term_counts[i] += 1

        for c in term_counts:
            if c > 1:
                total += 1

        return total

    @staticmethod
    def _build_pareto_front(candidates: list[dict]) -> list[dict]:
        """Non-dominated sort over (complexity, scaled_conc_mse)."""
        # Filter out NaN/Inf (defence in depth — should be caught upstream)
        valid = [
            c
            for c in candidates
            if np.isfinite(c["scaled_conc_mse"]) and np.isfinite(c["complexity"])
        ]
        if not valid:
            return []

        front: list[dict] = []
        for c in valid:
            dominated = False
            for other in valid:
                if other is c:
                    continue
                if (
                    other["complexity"] <= c["complexity"]
                    and other["scaled_conc_mse"] <= c["scaled_conc_mse"]
                    and (
                        other["complexity"] < c["complexity"]
                        or other["scaled_conc_mse"] < c["scaled_conc_mse"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(c)
        front.sort(key=lambda c: c["complexity"])
        return front

    def _alpha_select(self, front: list[dict]) -> dict | None:
        """Alpha-selection from the Pareto front.

        Select the lowest-complexity candidate whose scaled_conc_mse
        is within alpha of the best on the front.
        """
        if not front:
            return None
        best_mse = min(c["scaled_conc_mse"] for c in front)
        threshold = best_mse * self.pareto_alpha + self.pareto_eps
        eligible = [c for c in front if c["scaled_conc_mse"] <= threshold]
        if eligible:
            eligible.sort(key=lambda c: (c["complexity"], c["scaled_conc_mse"]))
            return eligible[0]
        # Fallback: min scaled_conc_mse
        front_by_mse = sorted(front, key=lambda c: c["scaled_conc_mse"])
        return front_by_mse[0]

    def _format_equations(self) -> list[str]:
        if self._reaction_mechanism is None or self._rate_constants is None:
            return []

        equations = [[] for _ in range(self._dim)]

        for reaction, k in zip(self._reaction_mechanism, self._rate_constants):
            if k < 1e-14:
                continue

            reactants = reaction[: self._dim]
            products = reaction[self._dim :]
            stoich = products - reactants

            prop_parts = []
            for i, order in enumerate(reactants):
                if order == 1:
                    prop_parts.append(f"x_{i}")
                elif order > 1:
                    prop_parts.append(f"x_{i}^{int(order)}")
            prop_str = "*".join(prop_parts) if prop_parts else "1"

            for i in range(self._dim):
                if stoich[i] != 0:
                    coeff = stoich[i] * k
                    sign = "+" if coeff > 0 else "-"
                    equations[i].append(f"{sign} {abs(coeff):.17g}*{prop_str}")

        result = []
        for i, parts in enumerate(equations):
            if not parts:
                result.append(f"dx_{i}/dt = 0")
            else:
                text = " ".join(parts)
                if text.startswith("+ "):
                    text = text[2:]
                result.append(f"dx_{i}/dt = {text}")

        return result

    def _pareto_select_from_candidates(
        self,
        best_mechs: list,
        t: np.ndarray,
        traj: np.ndarray,
        construct_matrix,
        reaction_system,
    ) -> tuple[list[np.ndarray], np.ndarray, float, float, dict] | None:
        """Roll out all candidates, compute complexity, build Pareto front.

        Each ``best_mechs`` entry is ``[mechanism, idx, rate_consts, deriv_mse]``
        with an optional 5th element ``island_n_rxns`` in island mode.

        Returns ``(mechanism, rate_consts, deriv_mse, scaled_conc_mse, pareto_meta)``
        of the selected winner, or ``None`` if no candidate integrated successfully.
        """
        y0 = traj[0, :].copy()
        scale = np.max(np.abs(traj), axis=0, keepdims=True)
        scale[scale < 1e-14] = 1.0
        traj_scaled = traj / scale
        t_eval = np.asarray(t, dtype=float).reshape(-1)

        candidates: list[dict] = []
        candidate_id = 0

        for mech_entry in best_mechs:
            mechanism = mech_entry[0]
            rate_consts = np.asarray(mech_entry[2], dtype=float)
            deriv_mse = float(mech_entry[3])
            island_n = mech_entry[4] if len(mech_entry) > 4 else None

            nonzero_mask = np.abs(rate_consts) > 1e-14
            if not np.any(nonzero_mask):
                continue

            cleaned_mechanism = construct_matrix(mechanism)
            cleaned_coeffs = rate_consts[nonzero_mask]
            cleaned_mechanism = [
                r for r, m in zip(cleaned_mechanism, nonzero_mask) if m
            ]

            try:
                sol = self._integrate(t_eval, y0, cleaned_mechanism, cleaned_coeffs)
                if sol.status != 0:
                    raise RuntimeError(sol.message)

                pred = sol.y.T
                if pred.shape != traj.shape or not np.all(np.isfinite(pred)):
                    raise RuntimeError(
                        "candidate prediction has invalid shape or nonfinite values"
                    )

                pred_scaled = pred / scale
                scaled_mse = float(np.mean((traj_scaled - pred_scaled) ** 2))
                if not np.isfinite(scaled_mse):
                    raise RuntimeError("candidate rollout loss is nonfinite")
                complexity = self._compute_complexity(
                    cleaned_mechanism,
                    cleaned_coeffs,
                    traj.shape[1],
                )
                num_active = int(np.sum(nonzero_mask))

                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "island_n_rxns": island_n,
                        "num_active_reactions": num_active,
                        "derivative_mse": deriv_mse,
                        "scaled_conc_mse": scaled_mse,
                        "complexity": complexity,
                        "reaction_mechanism": [r.tolist() for r in cleaned_mechanism],
                        "rate_constants": cleaned_coeffs.tolist(),
                        "rollout_status": "success",
                    }
                )
                candidate_id += 1
            except Exception as exc:
                self.rollout_failures.append(
                    {
                        "island_n_rxns": island_n,
                        "reason": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue

        if not candidates:
            return None

        # Build Pareto front
        pareto_front = self._build_pareto_front(candidates)

        # Alpha-selection
        selected = self._alpha_select(pareto_front)
        if selected is None:
            return None

        pareto_meta = {
            "selected_complexity": selected["complexity"],
            "selected_scaled_conc_mse": selected["scaled_conc_mse"],
            "selected_derivative_mse": selected["derivative_mse"],
            "selected_island_n_rxns": selected.get("island_n_rxns"),
            "num_pareto_candidates": len(candidates),
            "num_pareto_front_points": len(pareto_front),
            "selection_rule": "alpha",
            "selection_alpha": self.pareto_alpha,
            "selection_eps": self.pareto_eps,
        }

        self.front_payload = {
            "candidates": candidates,
            "pareto_front": [
                {
                    "candidate_id": c["candidate_id"],
                    "complexity": c["complexity"],
                    "scaled_conc_mse": c["scaled_conc_mse"],
                }
                for c in pareto_front
            ],
            "selected_candidate_id": selected["candidate_id"],
        }

        best_mech = [np.array(r) for r in selected["reaction_mechanism"]]
        best_coeffs = np.array(selected["rate_constants"], dtype=float)
        best_deriv_mse = float(selected["derivative_mse"])
        best_scaled_mse = float(selected["scaled_conc_mse"])
        return best_mech, best_coeffs, best_deriv_mse, best_scaled_mse, pareto_meta
