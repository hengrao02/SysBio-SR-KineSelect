"""Small diagnostic counters for KineSelectSR kinetic-law selections."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from Methods.KineSelectSR.formulation import KineSelectSRKineticVariant


HILL_MODIFIER_LAWS = {"hill_activation", "hill_inhibition"}


def kinetic_law_candidate_counts(
    variants: Iterable[KineSelectSRKineticVariant],
) -> dict[str, int]:
    """Count configured SINDy dictionary candidates by kinetic family."""
    records = list(variants)
    return {
        "num_candidate_generated": len(records),
        "num_ma_candidates": sum(1 for item in records if item.substrate_law == "mass_action"),
        "num_mm_candidates": sum(1 for item in records if item.substrate_law == "mm"),
        "num_hill_candidates": sum(1 for item in records if item.modifier_law in HILL_MODIFIER_LAWS),
    }


def kinetic_law_selected_counts(
    variants: Iterable[KineSelectSRKineticVariant],
    rate_constants: Any = None,
    *,
    active_threshold: float = 1e-14,
) -> dict[str, int]:
    """Count active selected variants by substrate and modifier family."""
    records = list(variants)
    if rate_constants is None:
        active_mask = [True] * len(records)
    else:
        ks = np.asarray(rate_constants, dtype=float)
        active_mask = [
            idx < len(ks) and np.isfinite(ks[idx]) and abs(float(ks[idx])) > active_threshold
            for idx in range(len(records))
        ]
    active = [variant for variant, is_active in zip(records, active_mask) if is_active]
    return {
        "num_selected_reactions": len(active),
        "num_selected_ma_reactions": sum(1 for item in active if item.substrate_law == "mass_action"),
        "num_selected_mm_reactions": sum(1 for item in active if item.substrate_law == "mm"),
        "num_selected_hill_reactions": sum(1 for item in active if item.modifier_law in HILL_MODIFIER_LAWS),
        "num_selected_none_modifier_terms": sum(1 for item in active if item.modifier_law == "none"),
        "num_selected_linear_modifier_terms": sum(1 for item in active if item.modifier_law == "linear"),
    }

