"""Grouped NNLS solver and KFI variant selection."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from Methods.KineSelectSR.formulation import (
    KineSelectSRModel as KineticModel,
    KineSelectSRKineticVariant as KineticReaction,
    build_design_matrix,
    compute_contributions,
    fit_rate_constants,
    threshold_by_contribution,
)
from .context import TargetContext
from .dictionary import GroupedDictionary


def solve_grouped_nnls(
    dictionary: GroupedDictionary,
    target_ctx: TargetContext,
    l2_lambda: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Solve joint NNLS for all dictionary columns."""
    m = dictionary.columns.shape[1]
    if m == 0:
        return (
            np.array([], dtype=float),
            float(np.linalg.norm(target_ctx.target_norm)),
        )

    A_solve = dictionary.columns
    b_solve = target_ctx.target_norm
    if l2_lambda > 0.0:
        eye_m = np.eye(m, dtype=dictionary.columns.dtype)
        A_solve = np.vstack([dictionary.columns, np.sqrt(l2_lambda) * eye_m])
        b_solve = np.hstack([target_ctx.target_norm, np.zeros(m, dtype=target_ctx.target_norm.dtype)])

    try:
        w_raw, rnorm = nnls(A_solve, b_solve, maxiter=max(1000, 10 * m))
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Grouped NNLS failed: {exc}") from exc

    return w_raw, float(rnorm)


def rank_variants_by_contribution(
    dictionary: GroupedDictionary,
    w_raw: np.ndarray,
    target_ctx: TargetContext,
) -> np.ndarray:
    """Compute per-column contribution fractions."""
    N = len(w_raw)
    if N == 0:
        return np.array([], dtype=float)

    target_norm_val = float(np.linalg.norm(target_ctx.target_norm))
    if target_norm_val < 1e-15:
        return np.zeros(N, dtype=float)

    contributions = np.empty(N, dtype=float)
    for j in range(N):
        contributions[j] = (
            float(np.linalg.norm(dictionary.columns[:, j] * w_raw[j]))
            / target_norm_val
        )

    return contributions


def classify_variant_family(variant: KineticReaction) -> str:
    """Classify a variant into a stable substrate_modifier family name."""
    substrate_names = {
        "mass_action": "MA",
        "mm": "MM",
        "hill_saturation": "HillSat",
    }
    modifier_names = {
        "none": "none",
        "linear": "linear",
        "hill_activation": "HillAct",
        "hill_inhibition": "HillInh",
    }

    substrate = substrate_names.get(variant.substrate_law, variant.substrate_law)
    modifier = modifier_names.get(variant.modifier_law, variant.modifier_law)
    return f"{substrate}_{modifier}"


_FAMILY_COMPLEXITY_RANK = {
    "MA_none": 0,
    "MA_linear": 1,
    "MM_none": 2,
    "MM_linear": 3,
}


def _variant_structural_complexity(variant: KineticReaction) -> int:
    """Return a small AST-like kinetic complexity for Occam tie-breaking."""
    reactant_order = int(sum(variant.reactants))
    if variant.substrate_law == "mass_action":
        substrate_nodes = reactant_order
    elif variant.substrate_law == "mm":
        substrate_nodes = max(reactant_order - 1, 0) + 5
    else:
        substrate_nodes = reactant_order

    connector_nodes = 0 if variant.modifier_law == "none" else 1
    if variant.modifier_law == "none":
        modifier_nodes = 0
    elif variant.modifier_law == "linear":
        modifier_nodes = 1
    elif variant.modifier_law in {"hill_activation", "hill_inhibition"}:
        modifier_nodes = 3 * int(variant.modifier_n or 1) + 2
    else:
        modifier_nodes = 0

    return substrate_nodes + connector_nodes + modifier_nodes


def _variant_family_order(variant: KineticReaction) -> tuple[int, int, int]:
    """Stable family order after structural complexity has been considered."""
    family = classify_variant_family(variant)
    if family in _FAMILY_COMPLEXITY_RANK:
        return (0, _FAMILY_COMPLEXITY_RANK[family], 0)

    substrate_rank = 0 if variant.substrate_law == "mass_action" else 1
    if variant.modifier_law == "hill_activation":
        modifier_rank = 0
    elif variant.modifier_law == "hill_inhibition":
        modifier_rank = 1
    else:
        modifier_rank = 10
    return (1, substrate_rank, modifier_rank)


def _variant_occam_parts(variant: KineticReaction) -> tuple[int, int, int, int, int, str]:
    """Common Occam ordering parts used by contribution and refit selectors."""
    hill_n = int(variant.modifier_n or 0) if variant.modifier_law in {
        "hill_activation",
        "hill_inhibition",
    } else 0
    family_group, substrate_rank, modifier_rank = _variant_family_order(variant)
    return (
        _variant_structural_complexity(variant),
        hill_n,
        family_group,
        substrate_rank,
        modifier_rank,
        classify_variant_family(variant),
    )


def _variant_occam_key(
    dictionary: GroupedDictionary,
    column_idx: int,
    nmse: float,
) -> tuple[object, ...]:
    """Sort key for selecting the simplest near-best kinetic variant."""
    variant = dictionary.column_meta[column_idx].variant
    return (*_variant_occam_parts(variant), float(nmse), int(column_idx))


def _is_hill_variant(variant: KineticReaction) -> bool:
    """Return whether a kinetic variant is a Hill modifier law."""
    return variant.modifier_law in {"hill_activation", "hill_inhibition"}


def prune_collinear_columns_by_occam(
    dictionary: GroupedDictionary,
    *,
    cos_threshold: float,
    hill_cos_threshold: float,
) -> tuple[GroupedDictionary, dict[str, object]]:
    """Prune nearly-collinear dictionary columns, keeping simpler variants.

    Operates per reaction group: within each group, columns are sorted by
    Occam complexity (simplest first), then greedily retained.  A column is
    pruned when its absolute cosine similarity with any already-kept simpler
    column exceeds the threshold.  Hill variants use a more aggressive
    threshold (`hill_cos_threshold`) when either column is a Hill variant.

    Returns the pruned dictionary and a diagnostics dictionary.
    """
    columns = dictionary.columns
    column_meta = dictionary.column_meta
    group_offsets = dictionary.group_offsets
    n_reactions = dictionary.n_reactions
    n_cols_before = int(columns.shape[1])

    if n_cols_before == 0 or n_reactions == 0:
        return dictionary, {
            "collinearity_columns_before": n_cols_before,
            "collinearity_columns_after": n_cols_before,
            "collinearity_columns_pruned": 0,
        }

    kept_global: list[int] = []
    new_group_offsets: list[int] = [0]
    max_cos_pruned: float = 0.0
    pruned_families: dict[str, int] = {}

    for group_idx in range(n_reactions):
        start = int(group_offsets[group_idx])
        end = int(group_offsets[group_idx + 1])
        if end <= start:
            new_group_offsets.append(len(kept_global))
            continue

        # Sort by Occam complexity (simplest first)
        candidate_indices = sorted(
            range(start, end),
            key=lambda idx: _variant_occam_key(dictionary, idx, 0.0),
        )

        kept_group: list[int] = []
        for idx in candidate_indices:
            col_i = columns[:, idx]
            norm_i_sq = float(np.dot(col_i, col_i))
            if norm_i_sq < 1e-30:
                # Degenerate column — drop it
                family = classify_variant_family(column_meta[idx].variant)
                pruned_families[family] = pruned_families.get(family, 0) + 1
                continue

            should_drop = False
            for kept_idx in kept_group:
                col_j = columns[:, kept_idx]
                # Columns are already L2-normalised
                cos_ij = abs(float(np.dot(col_i, col_j)))
                norm_j_sq = float(np.dot(col_j, col_j))
                if norm_i_sq > 1e-30 and norm_j_sq > 1e-30:
                    cos_ij /= float(np.sqrt(norm_i_sq * norm_j_sq))

                vi = column_meta[idx].variant
                vj = column_meta[kept_idx].variant
                threshold = (
                    hill_cos_threshold
                    if _is_hill_variant(vi) or _is_hill_variant(vj)
                    else cos_threshold
                )

                if cos_ij >= threshold:
                    should_drop = True
                    if cos_ij > max_cos_pruned:
                        max_cos_pruned = cos_ij
                    family = classify_variant_family(vi)
                    pruned_families[family] = pruned_families.get(family, 0) + 1
                    break

            if not should_drop:
                kept_group.append(idx)

        kept_global.extend(kept_group)
        new_group_offsets.append(len(kept_global))

    if not kept_global:
        # All columns pruned — return empty dictionary
        empty = GroupedDictionary(
            columns=np.zeros((columns.shape[0], 0), dtype=float),
            column_meta=[],
            group_offsets=[0] * (n_reactions + 1),
            n_reactions=n_reactions,
        )
        return empty, {
            "collinearity_columns_before": n_cols_before,
            "collinearity_columns_after": 0,
            "collinearity_columns_pruned": n_cols_before,
            "collinearity_cos_threshold": cos_threshold,
            "hill_collinearity_cos_threshold": hill_cos_threshold,
            "collinearity_max_abs_cos_pruned": max_cos_pruned,
            "collinearity_pruned_family_counts": pruned_families,
            "collinearity_groups_touched": sum(
                1 for i in range(n_reactions)
                if (group_offsets[i + 1] - group_offsets[i]) > 0
                and (new_group_offsets[i + 1] - new_group_offsets[i])
                < (group_offsets[i + 1] - group_offsets[i])
            ),
        }

    new_columns = columns[:, kept_global]
    new_meta = [column_meta[i] for i in kept_global]
    n_cols_after = int(new_columns.shape[1])

    new_dict = GroupedDictionary(
        columns=new_columns,
        column_meta=new_meta,
        group_offsets=new_group_offsets,
        n_reactions=n_reactions,
    )

    diag: dict[str, object] = {
        "collinearity_columns_before": n_cols_before,
        "collinearity_columns_after": n_cols_after,
        "collinearity_columns_pruned": n_cols_before - n_cols_after,
        "collinearity_cos_threshold": cos_threshold,
        "hill_collinearity_cos_threshold": hill_cos_threshold,
        "collinearity_max_abs_cos_pruned": max_cos_pruned if max_cos_pruned > 0 else None,
        "collinearity_pruned_family_counts": pruned_families,
        "collinearity_groups_touched": sum(
            1 for i in range(n_reactions)
            if (group_offsets[i + 1] - group_offsets[i]) > 0
            and (new_group_offsets[i + 1] - new_group_offsets[i])
            < (group_offsets[i + 1] - group_offsets[i])
        ),
    }

    return new_dict, diag


def select_kfi_variants(
    dictionary: GroupedDictionary,
    contributions: np.ndarray,
    active_threshold: float,
) -> tuple[list[KineticReaction], list[str]]:
    """Select the highest-contribution kinetic variant in each reaction group."""
    selected_reactions = []
    selected_families = []

    for j in range(dictionary.n_reactions):
        start = dictionary.group_offsets[j]
        end = dictionary.group_offsets[j + 1]
        group_size = end - start

        if group_size == 0:
            continue

        group_contribs = contributions[start:end]

        # If all contributions in group are < active_threshold, fall back to MA_none
        if np.all(group_contribs < active_threshold):
            fallback_idx = start
            for idx in range(start, end):
                if classify_variant_family(dictionary.column_meta[idx].variant) == "MA_none":
                    fallback_idx = idx
                    break
            selected_rxn = dictionary.column_meta[fallback_idx].variant
            selected_family = "MA_none"
        else:
            # Select the candidate with the highest contribution score
            local_best_idx = np.argmax(group_contribs)
            best_idx = start + local_best_idx
            selected_rxn = dictionary.column_meta[best_idx].variant
            selected_family = classify_variant_family(selected_rxn)

        selected_reactions.append(selected_rxn)
        selected_families.append(selected_family)

    return selected_reactions, selected_families


def refit_selected_model(
    selected_reactions,
    X,
    target_ctx,
    options,
) -> KineticModel:
    """Refit the selected model using the standard design matrix fitting.

    Args:
        selected_reactions: the chosen ``KineSelectSRKineticVariant`` objects to refit.
        X: trajectory array ``[T, D]``.
        target_ctx: precomputed :class:`TargetContext` for the system.
        options: ``SINDyKineticOptions`` (or a ``FormulationConfig``); anything
            exposing ``to_formulation_config`` is converted, otherwise used as-is.

    Note:
        This is the single canonical KineSelectSR signature. The historical
        ``(stoich_pairs, selected_reactions, X, target_ctx, options)`` 5-arg
        overload was carried over from ``Methods.KineticAwareSR`` and had no
        KineSelectSR caller; it was removed. ``Methods.KineticAwareSR`` keeps its own
        copy with the stoich-pairs path — the two methods are independent.
    """
    opts = options

    if opts is None:
        raise ValueError("options/config must be provided")

    if hasattr(opts, "to_formulation_config"):
        config = opts.to_formulation_config()
    else:
        config = opts

    m = len(selected_reactions)
    if m == 0:
        model = KineticModel(reactions=())
        model.rate_constants = np.array([], dtype=float)
        model.deriv_loss = (
            float(np.linalg.norm(target_ctx.target_norm)) ** 2 / target_ctx.denom
        )
        return model

    model = KineticModel(reactions=tuple(selected_reactions))

    X_dot_max_raw = np.max(np.abs(target_ctx.X_dot), axis=0, keepdims=True)
    A_norm = build_design_matrix(model, X, X_dot_max_raw)

    try:
        k_raw, _ = fit_rate_constants(
            A_norm, target_ctx.target_norm, l2_lambda=config.l2_lambda,
        )
    except RuntimeError:
        k_raw = np.zeros(m, dtype=float)

    contributions = compute_contributions(A_norm, k_raw, target_ctx.target_norm)
    k_active = threshold_by_contribution(
        k_raw, contributions, max_ratio=config.rate_threshold_ratio,
    )
    zero_threshold = getattr(opts, "numerical_zero_threshold", 1e-14)
    k_active = np.where(np.abs(k_active) > zero_threshold, k_active, 0.0)

    residual = target_ctx.target_norm - A_norm @ k_active
    deriv_scaled_nmse = float(np.dot(residual, residual)) / target_ctx.denom

    model.rate_constants = k_active
    model.deriv_loss = deriv_scaled_nmse

    return model
