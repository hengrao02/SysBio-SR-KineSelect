from __future__ import annotations
import os
import numpy as np
from Methods.KineSelectSR.formulation import (
    FormulationConfig,
    KineSelectSREvalMetrics,
    KineSelectSRModel,
)
from Methods.KineSelectSR.formulation.complexity import compute_complexity
from Methods.KineSelectSR.formulation.evaluate import evaluate as evaluate_formulation
from Methods.KineSelectSR.sindy.profiler import (
    FullLawSpaceProfiler,
    SINDyKineticOptions,
)
from .individual import KineSelectSRIndividual

_RECOVERABLE_EVAL_ERRORS = (ValueError, RuntimeError, FloatingPointError)


class KineSelectSREvaluator:
    """Evaluates individual reaction skeletons using KineSelectSR-SINDy profiling."""
    
    def __init__(self, t: np.ndarray, X: np.ndarray, options: SINDyKineticOptions):
        self.t = np.asarray(t, dtype=float).reshape(-1)
        self.X = np.asarray(X, dtype=float)
        self.profiler = FullLawSpaceProfiler(self.t, self.X, options)
        self.options = self.profiler.options
        self.cache = {}
        self.last_error: Exception | None = None

    def evaluate(self, individual: KineSelectSRIndividual) -> KineSelectSREvalMetrics:
        """Evaluate a single individual and cache the results."""
        key = individual.reaction_key
        if key in self.cache:
            metrics, selected_variants, rate_constants, diagnostics = self.cache[key]
            individual.metrics = metrics
            individual.selected_variants = selected_variants
            individual.rate_constants = rate_constants.copy() if rate_constants is not None else None
            individual.diagnostics = dict(diagnostics or {})
            return metrics

        try:
            result = self.profiler.evaluate(individual.genes)
            refit_nmse = result.refit_nmse_full_law_space
            if not np.isfinite(refit_nmse):
                raise ValueError("refit_nmse_full_law_space is not finite")
        except _RECOVERABLE_EVAL_ERRORS as exc:
            return self._mark_invalid(individual, key, exc)
        except Exception:
            self.last_error = None
            raise

        self.last_error = None

        # Compute model complexity
        fitted_model = KineSelectSRModel(
            reactions=result.selected_variants,
            rate_constants=result.rate_constants,
            deriv_loss=refit_nmse
        )
        zero_threshold = getattr(self.options, "numerical_zero_threshold", 1e-14)
        comp = compute_complexity(fitted_model, threshold=zero_threshold)
        n_active = int(np.sum(np.abs(result.rate_constants) > zero_threshold))

        metrics = KineSelectSREvalMetrics(
            deriv_scaled_nmse=refit_nmse,
            deriv_scaled_nmse_raw=result.raw_nmse_full_law_space,
            rollout_nmse=float("inf"),
            complexity=comp,
            rate_constants=result.rate_constants,
            n_active_reactions=n_active,
        )

        individual.metrics = metrics
        individual.selected_variants = result.selected_variants
        individual.rate_constants = result.rate_constants.copy()
        individual.diagnostics = dict(result.diagnostics or {})

        self.cache[key] = (
            metrics,
            result.selected_variants,
            result.rate_constants,
            dict(result.diagnostics or {}),
        )
        return metrics

    def evaluate_rollout(
        self,
        individual: KineSelectSRIndividual,
        formulation_config: FormulationConfig,
    ) -> KineSelectSREvalMetrics:
        """Refit fixed SINDy-selected variants and compute rollout metrics."""
        key = individual.reaction_key
        if individual.metrics is None:
            self.evaluate(individual)

        if individual.metrics is None:
            raise RuntimeError("Derivative evaluation did not populate metrics")

        diagnostics = dict(individual.diagnostics or {})
        if diagnostics.get("rollout_evaluated"):
            return individual.metrics

        selected_variants = tuple(individual.selected_variants or ())
        sindy_metrics = individual.metrics
        sindy_rate_constants = (
            None
            if individual.rate_constants is None
            else np.asarray(individual.rate_constants, dtype=float).copy()
        )
        diagnostics.update({
            "rollout_sindy_deriv_scaled_nmse": float(sindy_metrics.deriv_scaled_nmse),
            "rollout_sindy_deriv_scaled_nmse_raw": float(
                sindy_metrics.deriv_scaled_nmse_raw
            ),
            "rollout_sindy_complexity": int(sindy_metrics.complexity),
            "rollout_sindy_n_active_reactions": int(
                sindy_metrics.n_active_reactions
            ),
        })

        try:
            model = KineSelectSRModel(
                reactions=selected_variants,
                deriv_loss=sindy_metrics.deriv_scaled_nmse,
            )
            rollout_metrics = evaluate_formulation(
                model,
                self.t,
                self.X,
                formulation_config,
            )
        except Exception as exc:
            self.last_error = exc
            fallback_metrics = KineSelectSREvalMetrics(
                deriv_scaled_nmse=sindy_metrics.deriv_scaled_nmse,
                deriv_scaled_nmse_raw=sindy_metrics.deriv_scaled_nmse_raw,
                rollout_nmse=float("inf"),
                complexity=sindy_metrics.complexity,
                rate_constants=(
                    np.array([], dtype=float)
                    if sindy_rate_constants is None
                    else sindy_rate_constants.copy()
                ),
                n_active_reactions=sindy_metrics.n_active_reactions,
            )
            diagnostics.update({
                "rollout_evaluated": True,
                "rollout_error": repr(exc),
            })
            individual.metrics = fallback_metrics
            individual.selected_variants = selected_variants
            individual.rate_constants = fallback_metrics.rate_constants.copy()
            individual.diagnostics = diagnostics
            self.cache[key] = (
                fallback_metrics,
                selected_variants,
                individual.rate_constants.copy(),
                dict(diagnostics),
            )
            return fallback_metrics

        self.last_error = None
        diagnostics.update({
            "rollout_evaluated": True,
            "rollout_error": None,
        })
        individual.metrics = rollout_metrics
        individual.selected_variants = selected_variants
        individual.rate_constants = np.asarray(
            rollout_metrics.rate_constants,
            dtype=float,
        ).copy()
        individual.diagnostics = diagnostics
        self.cache[key] = (
            rollout_metrics,
            selected_variants,
            individual.rate_constants.copy(),
            dict(diagnostics),
        )
        return rollout_metrics

    def _mark_invalid(
        self,
        individual: KineSelectSRIndividual,
        key: frozenset,
        exc: Exception,
    ) -> KineSelectSREvalMetrics:
        self.last_error = exc
        if os.environ.get("KINESELECTSR_DEBUG"):
            import sys
            import traceback
            print(
                f"Evaluator caught recoverable error on genes {individual.genes}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()

        failed_rate_constants = np.array([], dtype=float)
        metrics = KineSelectSREvalMetrics(
            deriv_scaled_nmse=float("inf"),
            deriv_scaled_nmse_raw=float("inf"),
            rollout_nmse=float("inf"),
            complexity=0,
            rate_constants=failed_rate_constants,
            n_active_reactions=0,
        )
        individual.metrics = metrics
        individual.selected_variants = ()
        individual.rate_constants = failed_rate_constants.copy()
        individual.diagnostics = {
            "num_candidate_generated": 0,
            "num_ma_candidates": 0,
            "num_mm_candidates": 0,
            "num_hill_candidates": 0,
            "num_selected_reactions": 0,
            "num_selected_ma_reactions": 0,
            "num_selected_mm_reactions": 0,
            "num_selected_hill_reactions": 0,
            "num_selected_none_modifier_terms": 0,
            "num_selected_linear_modifier_terms": 0,
            "error": repr(exc),
        }
        self.cache[key] = (metrics, (), individual.rate_constants, dict(individual.diagnostics))
        return metrics
