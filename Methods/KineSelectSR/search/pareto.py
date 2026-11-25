from __future__ import annotations
import numpy as np
from .individual import KineSelectSRIndividual, canonical_candidate_id

_VALID_OBJECTIVES = {"deriv", "rollout"}


def _validate_objective(objective: str) -> str:
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"objective must be one of {_VALID_OBJECTIVES}, got {objective!r}"
        )
    return objective


def _objective_loss(individual: KineSelectSRIndividual, objective: str) -> float:
    metrics = individual.metrics
    if metrics is None:
        return float("inf")
    if objective == "deriv":
        return float(metrics.deriv_scaled_nmse)
    return float(metrics.rollout_nmse)


def _objective_tuple(
    individual: KineSelectSRIndividual,
    objective: str,
) -> tuple[float, int, int]:
    metrics = individual.metrics
    assert metrics is not None
    return (
        _objective_loss(individual, objective),
        int(metrics.complexity),
        int(metrics.n_active_reactions),
    )


def _dominates(
    loss_a: float,
    complexity_a: int,
    active_a: int,
    loss_b: float,
    complexity_b: int,
    active_b: int,
) -> bool:
    """Return True when A dominates B (all objectives minimized)."""
    if loss_a > loss_b or complexity_a > complexity_b or active_a > active_b:
        return False
    return (
        loss_a < loss_b
        or complexity_a < complexity_b
        or active_a < active_b
    )


def _pareto_front_fast(
    unique_inds: list[KineSelectSRIndividual],
    objective: str,
) -> list[KineSelectSRIndividual]:
    """Build a 3-objective Pareto front in O(n log n + m * C * A) time."""
    if not unique_inds:
        return []

    # Collapse to the best loss per (complexity, n_active) bucket.
    bucket_best: dict[tuple[int, int], KineSelectSRIndividual] = {}
    for ind in unique_inds:
        loss, complexity, active = _objective_tuple(ind, objective)
        key = (complexity, active)
        existing = bucket_best.get(key)
        if (
            existing is None
            or loss < _objective_loss(existing, objective)
            or (
                loss == _objective_loss(existing, objective)
                and canonical_candidate_id(ind) < canonical_candidate_id(existing)
            )
        ):
            bucket_best[key] = ind

    entries = [
        (
            _objective_loss(ind, objective),
            int(ind.metrics.complexity),
            int(ind.metrics.n_active_reactions),
            ind,
        )
        for ind in bucket_best.values()
    ]
    entries.sort(
        key=lambda item: (
            item[0], item[1], item[2], canonical_candidate_id(item[3])
        )
    )

    if not entries:
        return []

    max_complexity = max(complexity for _, complexity, _, _ in entries)
    max_active = max(active for _, _, active, _ in entries)

    # grid[c][a] holds the loss of the accepted front point at (c, a), or inf.
    grid: list[list[float]] = [
        [float("inf")] * (max_active + 1)
        for _ in range(max_complexity + 1)
    ]
    front: list[KineSelectSRIndividual] = []

    for loss, complexity, active, ind in entries:
        dominated = False
        for c2 in range(complexity + 1):
            row = grid[c2]
            for a2 in range(active + 1):
                other_loss = row[a2]
                if not np.isfinite(other_loss):
                    continue
                if _dominates(other_loss, c2, a2, loss, complexity, active):
                    dominated = True
                    break
            if dominated:
                break
        if dominated:
            continue

        survivors: list[KineSelectSRIndividual] = []
        for survivor in front:
            s_loss, s_complexity, s_active = _objective_tuple(survivor, objective)
            if _dominates(loss, complexity, active, s_loss, s_complexity, s_active):
                grid[s_complexity][s_active] = float("inf")
                continue
            survivors.append(survivor)
        front = survivors
        front.append(ind)
        grid[complexity][active] = loss

    front.sort(
        key=lambda item: (
            *_objective_tuple(item, objective), canonical_candidate_id(item)
        )
    )
    return front


def build_pareto_front(
    individuals: list[KineSelectSRIndividual],
    objective: str = "deriv",
) -> list[KineSelectSRIndividual]:
    """Build a Pareto front over (objective loss, complexity, n_active_reactions)."""
    objective = _validate_objective(objective)
    evaluated = [
        ind for ind in individuals
        if ind.metrics is not None and np.isfinite(_objective_loss(ind, objective))
    ]
    if not evaluated:
        return []

    # Deduplicate by reaction_key to avoid redundant Pareto points.
    by_reaction_key: dict[object, KineSelectSRIndividual] = {}
    for ind in evaluated:
        current = by_reaction_key.get(ind.reaction_key)
        if current is None or canonical_candidate_id(ind) < canonical_candidate_id(current):
            by_reaction_key[ind.reaction_key] = ind

    return _pareto_front_fast(list(by_reaction_key.values()), objective)


def alpha_select(
    front: list[KineSelectSRIndividual],
    alpha: float = 1.05,
    eps: float = 1e-12,
    objective: str = "deriv",
) -> KineSelectSRIndividual | None:
    """Select the lowest-complexity candidate within alpha of the best loss."""
    objective = _validate_objective(objective)
    if not front:
        return None

    valid = [
        (ind, _objective_loss(ind, objective))
        for ind in front
        if ind.metrics is not None and np.isfinite(_objective_loss(ind, objective))
    ]
    if not valid:
        return None

    losses = [loss for _, loss in valid]
    best_loss = min(losses)
    threshold = best_loss * alpha + eps

    eligible = [
        ind for ind, loss in valid if loss <= threshold
    ]
    if eligible:
        eligible.sort(
            key=lambda ind: (
                ind.metrics.complexity,
                ind.metrics.n_active_reactions,
                _objective_loss(ind, objective),
                canonical_candidate_id(ind),
            )
        )
        return eligible[0]

    return min(
        (ind for ind, loss in valid if loss == best_loss),
        key=canonical_candidate_id,
    )
