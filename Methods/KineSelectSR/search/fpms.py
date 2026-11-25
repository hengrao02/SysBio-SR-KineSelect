"""FPMS DR_KX v1 final selection over an existing global rollout Pareto front.

FPMS deliberately does not evaluate, refit, filter, or rebuild the front.  It
only scores the candidates supplied by the GA final-selection stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, cast

from .individual import (
    KineSelectSRIndividual,
    canonical_candidate_id,
)


FPMS_CRITERIA = "DR_KX"
FPMS_SELECTOR_VERSION = "fpms-dr-kx-v1"
FPMS_AST_COMPLEXITY_VERSION = "kinetic-ast-v1"
FPMS_COST_DIMENSIONS = (
    "log_derivative_loss", "log_rollout_loss", "kinetic_term_count", "ast_complexity",
)
_FIDELITY_DIMENSIONS = (0, 1)
_PARSIMONY_DIMENSIONS = (2, 3)
CostVector = tuple[float, float, float, float]


def fpms_scoring_contract() -> dict:
    """Describe the same fixed dimensions and grouping used by the scorer."""
    return {
        "selector_version": FPMS_SELECTOR_VERSION,
        "criteria": FPMS_CRITERIA,
        "cost_dimensions": list(FPMS_COST_DIMENSIONS),
        "ast_complexity_version": FPMS_AST_COMPLEXITY_VERSION,
        "kinetic_term_count": "active_MM_terms_plus_active_Hill_terms",
        "normalization": "per_dimension_minmax",
        "fidelity_dimensions": list(_FIDELITY_DIMENSIONS),
        "parsimony_dimensions": list(_PARSIMONY_DIMENSIONS),
        "fidelity_cost": "mean(D,R)",
        "parsimony_cost": "mean(K,X)",
        "score": "(1-fidelity_cost)*(1-parsimony_cost)",
        "tie_break": ["descending_score", "ascending_fidelity_cost", "canonical_candidate_id"],
    }


_SUPPORTED_SUBSTRATE_LAWS = {"mass_action", "mm"}
_SUPPORTED_MODIFIER_LAWS = {
    "none",
    "linear",
    "hill_activation",
    "hill_inhibition",
}
_HILL_MODIFIER_LAWS = {"hill_activation", "hill_inhibition"}


@dataclass(frozen=True)
class FPMSScore:
    candidate_index: int
    candidate_id: str
    raw_costs: CostVector
    normalized_costs: CostVector
    fidelity_cost: float
    parsimony_cost: float
    score: float
    rank: int


@dataclass(frozen=True)
class FPMSSelectionResult:
    winner: KineSelectSRIndividual | None
    scores: tuple[FPMSScore, ...]
    degenerate_dimensions: tuple[int, ...]
    fallback_reason: str | None


def _active_kinetic_counts(
    candidate: KineSelectSRIndividual,
    candidate_index: int,
    active_threshold: float,
) -> tuple[int, int]:
    variants = candidate.selected_variants
    rates = candidate.rate_constants
    metrics = candidate.metrics
    if variants is None:
        raise ValueError(f"candidate {candidate_index}: selected_variants is missing")
    if rates is None:
        raise ValueError(f"candidate {candidate_index}: rate_constants is missing")
    if len(variants) != len(rates):
        raise ValueError(
            f"candidate {candidate_index}: selected_variants length {len(variants)} "
            f"does not match rate_constants length {len(rates)}"
        )
    if metrics is None:
        raise ValueError(f"candidate {candidate_index}: metrics is missing")
    if not hasattr(metrics, "n_active_reactions"):
        raise ValueError(
            f"candidate {candidate_index}: metrics.n_active_reactions is missing"
        )

    active_count = 0
    kinetic_term_count = 0
    for reaction_index, (variant, rate) in enumerate(zip(variants, rates)):
        try:
            rate_float = float(rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"candidate {candidate_index}: rate_constants[{reaction_index}] "
                f"cannot be converted to float: {rate!r}"
            ) from exc
        if not math.isfinite(rate_float):
            raise ValueError(
                f"candidate {candidate_index}: rate_constants[{reaction_index}] "
                f"must be finite, got {rate!r}"
            )

        substrate_law = getattr(variant, "substrate_law", None)
        modifier_law = getattr(variant, "modifier_law", None)
        if substrate_law not in _SUPPORTED_SUBSTRATE_LAWS:
            raise ValueError(
                f"candidate {candidate_index}: unsupported substrate_law "
                f"at selected_variants[{reaction_index}]: {substrate_law!r}"
            )
        if modifier_law not in _SUPPORTED_MODIFIER_LAWS:
            raise ValueError(
                f"candidate {candidate_index}: unsupported modifier_law "
                f"at selected_variants[{reaction_index}]: {modifier_law!r}"
            )

        active = abs(rate_float) > active_threshold
        if active:
            active_count += 1
            kinetic_term_count += int(substrate_law == "mm")
            kinetic_term_count += int(modifier_law in _HILL_MODIFIER_LAWS)

    metrics_active = metrics.n_active_reactions
    if active_count != metrics_active:
        raise ValueError(
            f"candidate {candidate_index}: recomputed active reaction count "
            f"{active_count} does not match metrics.n_active_reactions "
            f"{metrics_active!r}"
        )
    return active_count, kinetic_term_count


def _normalize_column(values: Sequence[float]) -> tuple[list[float], bool]:
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return [0.0] * len(values), True
    minimum = min(valid)
    maximum = max(valid)
    if maximum == minimum:
        return [0.0 if math.isfinite(value) else 1.0 for value in values], True
    scale = maximum - minimum
    return [
        (value - minimum) / scale
        if math.isfinite(value)
        else 1.0
        for value in values
    ], False


def fpms_select(
    candidates: Sequence[KineSelectSRIndividual],
    *,
    eps: float = 1e-12,
    active_threshold: float = 1e-14,
) -> FPMSSelectionResult:
    """Select one candidate using the frozen FPMS DR_KX v1 scoring rule."""
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError(f"eps must be finite and > 0, got {eps!r}")
    if not math.isfinite(float(active_threshold)) or float(active_threshold) < 0.0:
        raise ValueError(
            "active_threshold must be finite and >= 0, "
            f"got {active_threshold!r}"
        )
    if not candidates:
        return FPMSSelectionResult(None, (), (), "no_candidates")

    primitive_rows: list[CostVector] = []
    candidate_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        _active_count, kinetic_term_count = _active_kinetic_counts(
            candidate, index, float(active_threshold)
        )
        metrics = candidate.metrics
        assert metrics is not None
        try:
            deriv_loss = float(metrics.deriv_scaled_nmse)
            rollout_loss = float(metrics.rollout_nmse)
            complexity = float(metrics.complexity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"candidate {index}: metrics costs must be numeric"
            ) from exc

        def log_loss(loss: float) -> float:
            if not math.isfinite(loss) or loss < 0.0:
                return float("inf")
            return math.log(loss + float(eps))

        primitive_rows.append((
            log_loss(deriv_loss),
            log_loss(rollout_loss),
            float(kinetic_term_count),
            complexity,
        ))
        candidate_ids.append(canonical_candidate_id(candidate))

    normalized_columns: list[list[float]] = []
    degenerate_dimensions: list[int] = []
    for dimension in range(len(FPMS_COST_DIMENSIONS)):
        normalized, degenerate = _normalize_column(
            [row[dimension] for row in primitive_rows]
        )
        normalized_columns.append(normalized)
        if degenerate:
            degenerate_dimensions.append(dimension)

    score_data: list[tuple[int, tuple[float, ...], float, float, float]] = []
    for index in range(len(candidates)):
        costs = tuple(column[index] for column in normalized_columns)
        fidelity = sum(costs[i] for i in _FIDELITY_DIMENSIONS) / len(_FIDELITY_DIMENSIONS)
        parsimony = sum(costs[i] for i in _PARSIMONY_DIMENSIONS) / len(_PARSIMONY_DIMENSIONS)
        score = (1.0 - fidelity) * (1.0 - parsimony)
        score_data.append((index, costs, fidelity, parsimony, score))

    ranked = sorted(
        score_data,
        key=lambda item: (-item[4], item[2], candidate_ids[item[0]]),
    )
    rank_by_index = {item[0]: rank for rank, item in enumerate(ranked, start=1)}
    scores = tuple(
        FPMSScore(
            candidate_index=index,
            candidate_id=candidate_ids[index],
            raw_costs=primitive_rows[index],
            normalized_costs=cast(CostVector, costs),
            fidelity_cost=fidelity,
            parsimony_cost=parsimony,
            score=score,
            rank=rank_by_index[index],
        )
        for index, costs, fidelity, parsimony, score in score_data
    )
    winner_index = ranked[0][0]
    return FPMSSelectionResult(
        winner=candidates[winner_index],
        scores=scores,
        degenerate_dimensions=tuple(degenerate_dimensions),
        fallback_reason=None,
    )
