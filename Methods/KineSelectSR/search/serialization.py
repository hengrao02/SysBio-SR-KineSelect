from __future__ import annotations

from typing import Any

import numpy as np

from Methods.KineSelectSR.formulation import (
    KineSelectSRKineticVariant,
    KineSelectSRReactionGene,
)

from .individual import KineSelectSRIndividual


DEFAULT_ACTIVE_THRESHOLD = 1e-14
KINETIC_DIAGNOSTIC_COUNT_KEYS = (
    "num_candidate_generated",
    "num_ma_candidates",
    "num_mm_candidates",
    "num_hill_candidates",
    "num_selected_reactions",
    "num_selected_ma_reactions",
    "num_selected_mm_reactions",
    "num_selected_hill_reactions",
    "num_selected_none_modifier_terms",
    "num_selected_linear_modifier_terms",
    "collinearity_columns_before",
    "collinearity_columns_after",
    "collinearity_columns_pruned",
    "collinearity_groups_touched",
)
KINETIC_DIAGNOSTIC_SCALAR_KEYS = (
    "selector_mode",
    "collinearity_pruning_enabled",
    "collinearity_cos_threshold",
    "hill_collinearity_cos_threshold",
    "collinearity_max_abs_cos_pruned",
    "collinearity_pruned_family_counts",
)


def _json_float(value: Any) -> float | None:
    """Coerce a value to a JSON-safe float.

    Returns None for non-finite values (inf/-inf/nan). Standard JSON (RFC 7159)
    has no representation for these, and Python's json with the default
    allow_nan=True emits the non-standard tokens Infinity/NaN, which strict /
    cross-language parsers reject. rollout_nmse in particular is inf whenever a
    rollout is not computed, so emitting it raw produces invalid summary.json.
    """
    if value is None:
        return None
    f = float(value)
    if not np.isfinite(f):
        return None
    return f


def _json_safe(value: Any) -> Any:
    """Recursively coerce diagnostic payloads to strict-JSON-safe values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _json_float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def serialize_gene(gene: KineSelectSRReactionGene, index: int) -> dict[str, Any]:
    """Serialize a GA proposal gene without interpreting it as final kinetics."""
    return {
        "index": int(index),
        "reactants": list(gene.reactants),
        "products": list(gene.products),
        "proposed_modifier_idx": gene.modifier_idx,
    }


def serialize_selected_variant(
    variant: KineSelectSRKineticVariant,
    index: int,
    rate_constant: float | None,
    *,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
) -> dict[str, Any]:
    """Serialize a SINDy-selected kinetic variant with its fitted activity state."""
    k_value = _json_float(rate_constant)
    is_active = k_value is not None and abs(k_value) > active_threshold
    selected_modifier_idx = (
        None if variant.modifier_law == "none" else variant.modifier_idx
    )

    return {
        "index": int(index),
        "reactants": list(variant.reactants),
        "products": list(variant.products),
        "substrate_law": variant.substrate_law,
        "substrate_idx": variant.substrate_idx,
        "substrate_K": variant.substrate_K,
        "modifier_law": variant.modifier_law,
        "selected_modifier_idx": selected_modifier_idx,
        # Backward-compatible alias for older diagnostics code.
        "modifier_idx": selected_modifier_idx,
        "modifier_K": variant.modifier_K,
        "modifier_n": variant.modifier_n,
        "rate_constant": k_value,
        "k": k_value,
        "active": is_active,
    }


def serialize_individual(
    individual: KineSelectSRIndividual,
    *,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
) -> dict[str, Any]:
    """Serialize one evaluated individual into the KineSelectSR result schema."""
    rate_constants = (
        None
        if individual.rate_constants is None
        else np.asarray(individual.rate_constants, dtype=float)
    )

    selected_variants = []
    active_reactions = []
    for idx, variant in enumerate(individual.selected_variants or ()):
        k_value = (
            None
            if rate_constants is None or idx >= len(rate_constants)
            else float(rate_constants[idx])
        )
        record = serialize_selected_variant(
            variant,
            idx,
            k_value,
            active_threshold=active_threshold,
        )
        selected_variants.append(record)
        if record["active"]:
            active_reactions.append(record)

    payload: dict[str, Any] = {
        "rate_constants": _json_safe(
            None if rate_constants is None else rate_constants.tolist()
        ),
        "proposed_genes": [
            serialize_gene(gene, idx) for idx, gene in enumerate(individual.genes)
        ],
        "selected_variants": selected_variants,
        # Legacy field: expose final active selected mechanisms only.
        "reactions": active_reactions,
    }

    if individual.metrics is not None:
        metrics = individual.metrics
        payload.update(
            {
                "deriv_scaled_nmse": _json_float(metrics.deriv_scaled_nmse),
                "deriv_scaled_nmse_raw": _json_float(metrics.deriv_scaled_nmse_raw),
                "raw_nmse_full_law_space": _json_float(metrics.deriv_scaled_nmse_raw),
                "refit_nmse_full_law_space": _json_float(metrics.deriv_scaled_nmse),
                "derivative_nmse": _json_float(metrics.deriv_scaled_nmse),
                "fitness": _json_float(metrics.deriv_scaled_nmse),
                "rollout_nmse": _json_float(metrics.rollout_nmse),
                "complexity": int(metrics.complexity),
                "n_active_reactions": int(metrics.n_active_reactions),
            }
        )

    diagnostics = getattr(individual, "diagnostics", None) or {}
    count_payload = {
        key: int(diagnostics[key])
        for key in KINETIC_DIAGNOSTIC_COUNT_KEYS
        if key in diagnostics and diagnostics[key] is not None
    }
    if count_payload:
        payload.update(count_payload)
    scalar_payload = {
        key: _json_safe(diagnostics[key])
        for key in KINETIC_DIAGNOSTIC_SCALAR_KEYS
        if key in diagnostics and diagnostics[key] is not None
    }
    if scalar_payload:
        payload.update(scalar_payload)
    if count_payload or scalar_payload:
        payload["sindy_diagnostics"] = {**count_payload, **scalar_payload}

    return payload


def serialize_ga_result(
    result: Any,
    *,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
) -> dict[str, Any]:
    """Serialize a KineSelectSRGAResult-like object for external runners."""
    best_individual = getattr(result, "best_individual", None)
    payload: dict[str, Any] = {
        "history": _json_safe(list(getattr(result, "history", []))),
    }
    timing = getattr(result, "timing", None)
    if timing:
        payload["timing"] = _json_safe(timing)
    selection_diagnostics = getattr(result, "selection_diagnostics", None)
    if selection_diagnostics is not None:
        payload["final_selection"] = _json_safe(selection_diagnostics)
    if best_individual is None:
        return public_references(
            payload,
            candidate_labels(
                list(getattr(result, "top_candidates", []) or []),
                str(
                    (getattr(result, "timing", {}) or {}).get(
                        "global_selection_objective", "rollout"
                    )
                ),
            ),
        )
    payload.update(
        serialize_individual(
            best_individual,
            active_threshold=active_threshold,
        )
    )
    top_candidates = getattr(result, "top_candidates", None)
    if top_candidates:
        payload["top_candidates"] = [
            serialize_individual(cand, active_threshold=active_threshold)
            for cand in top_candidates
        ]
    return public_references(
        payload,
        candidate_labels(
            list(getattr(result, "top_candidates", []) or []),
            str(
                (getattr(result, "timing", {}) or {}).get(
                    "global_selection_objective", "rollout"
                )
            ),
        ),
    )


def ordered_candidates(front, objective):
    from .individual import canonical_candidate_id

    def key(ind):
        loss = (
            ind.metrics.rollout_nmse
            if objective == "rollout"
            else ind.metrics.deriv_scaled_nmse
        )
        return (
            float(loss),
            int(ind.metrics.complexity),
            int(ind.metrics.n_active_reactions),
            canonical_candidate_id(ind),
        )

    return sorted(front, key=key)


def candidate_labels(front, objective):
    from .individual import canonical_candidate_id

    return {
        canonical_candidate_id(ind): f"candidate-{i:06d}"
        for i, ind in enumerate(ordered_candidates(front, objective), 1)
    }


def public_references(value, labels):
    """Render algorithm-internal candidate keys as local, readable references."""
    if isinstance(value, str):
        return labels.get(value, value)
    if isinstance(value, dict):
        return {k: public_references(v, labels) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_references(v, labels) for v in value]
    return value
