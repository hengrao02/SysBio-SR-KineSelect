"""Grouped full-law-space dictionary construction for KFI."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from Methods.KineSelectSR.formulation import KineSelectSRKineticVariant as KineticReaction, compute_rate_batch
from .context import TargetContext
from .options import SINDyKineticOptions, resolve_sindy_options_for_X


@dataclass
class ColumnMeta:
    """Metadata for a single dictionary column."""

    reaction_idx: int
    variant: KineticReaction
    original_norm: float


@dataclass
class GroupedDictionary:
    """Dictionary columns grouped by proposed reaction."""

    columns: np.ndarray
    column_meta: list[ColumnMeta] = field(default_factory=list)
    group_offsets: list[int] = field(default_factory=list)
    n_reactions: int = 0


def get_K_grid_for_species(options: SINDyKineticOptions, species_idx: int) -> list[float]:
    """Get the K grid for a specific species index."""
    if options.species_K_grid is not None:
        if species_idx < 0 or species_idx >= len(options.species_K_grid):
            raise ValueError(
                f"species_idx {species_idx} out of range for species_K_grid "
                f"of length {len(options.species_K_grid)}"
            )
        ks = options.species_K_grid[species_idx]
        if not ks:
            raise ValueError(f"species_K_grid[{species_idx}] is empty")
        return list(ks)
    return list(options.K_grid)


def parse_reaction_gene(gene) -> tuple[tuple[int, ...], tuple[int, ...], int | None]:
    """Legacy scalar parser: return the first normalized modifier candidate."""
    reactants, products, modifier_indices = parse_reaction_gene_candidates(gene)
    modifier_idx = modifier_indices[0] if modifier_indices else None
    return reactants, products, modifier_idx


def parse_reaction_gene_candidates(
    gene,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Parse a reaction gene into reactants, products, and modifier candidates.

    This keeps the public formulation/search gene format scalar while allowing
    oracle annotations to provide multiple candidate modifiers. Each candidate
    is later expanded into a separate single-modifier kinetic variant.
    """
    if isinstance(gene, dict):
        reactants = tuple(gene["reactants"])
        products = tuple(gene["products"])
        if "modifier_indices" in gene:
            modifier_indices = _coerce_modifier_indices(
                gene["modifier_indices"], len(reactants), "modifier_indices"
            )
        elif "modifier_idx" in gene:
            modifier_indices = _coerce_modifier_indices(
                gene.get("modifier_idx"), len(reactants), "modifier_idx"
            )
        elif "modifier_mask" in gene:
            modifier_indices = _coerce_modifier_indices(
                gene["modifier_mask"], len(reactants), "modifier_mask", is_mask=True
            )
        elif "modifiers" in gene:
            modifier_indices = _coerce_modifier_indices(
                gene["modifiers"], len(reactants), "modifiers"
            )
        else:
            modifier_indices = ()
    elif isinstance(gene, (tuple, list)):
        if len(gene) == 3:
            reactants, products, modifiers = gene
            reactants = tuple(reactants)
            products = tuple(products)
            modifier_indices = _coerce_modifier_indices(
                modifiers, len(reactants), "tuple modifier field"
            )
        else:
            reactants, products = gene
            reactants = tuple(reactants)
            products = tuple(products)
            modifier_indices = ()
    else:
        reactants = tuple(gene.reactants)
        products = tuple(gene.products)
        if hasattr(gene, "modifier_indices"):
            modifier_indices = _coerce_modifier_indices(
                getattr(gene, "modifier_indices"), len(reactants), "modifier_indices"
            )
        else:
            m = getattr(gene, "modifier_idx", None)
            if m is not None:
                modifier_indices = _coerce_modifier_indices(
                    m, len(reactants), "modifier_idx"
                )
            elif hasattr(gene, "modifier_mask"):
                modifier_indices = _coerce_modifier_indices(
                    getattr(gene, "modifier_mask"),
                    len(reactants),
                    "modifier_mask",
                    is_mask=True,
                )
            elif hasattr(gene, "modifiers"):
                modifier_indices = _coerce_modifier_indices(
                    getattr(gene, "modifiers"), len(reactants), "modifiers"
                )
            else:
                modifier = getattr(gene, "modifier", None)
                modifier_indices = _coerce_modifier_indices(
                    modifier, len(reactants), "modifier"
                )

    return reactants, products, modifier_indices


def canonical_reaction_gene_key(
    gene,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return the hashable SINDy cache key for a reaction gene."""
    return parse_reaction_gene_candidates(gene)


def _coerce_modifier_indices(
    value,
    dim: int,
    field_name: str,
    *,
    is_mask: bool = False,
) -> tuple[int, ...]:
    """Normalize scalar/list/mask modifier inputs to unique integer indices."""
    if value is None:
        return ()

    if is_mask:
        return _modifier_mask_to_indices(value, dim, field_name)

    if isinstance(value, dict):
        return _unique_indices(
            [_extract_modifier_index(value, field_name)], field_name
        )

    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return (int(value),)

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _unique_indices([value.item()], field_name)
        if value.dtype == np.bool_:
            return _modifier_mask_to_indices(value, dim, field_name)
        return _unique_indices(value.tolist(), field_name)

    if isinstance(value, (list, tuple)):
        if not value:
            return ()
        if _looks_like_bool_mask(value):
            return _modifier_mask_to_indices(value, dim, field_name)
        return _unique_indices(value, field_name)

    return _unique_indices([value], field_name)


def _looks_like_bool_mask(value: list | tuple) -> bool:
    return all(isinstance(v, (bool, np.bool_)) for v in value)


def _modifier_mask_to_indices(value, dim: int, field_name: str) -> tuple[int, ...]:
    mask = np.asarray(value)
    if mask.ndim != 1:
        raise ValueError(f"{field_name} must be a 1D modifier mask")
    if mask.shape[0] != dim:
        raise ValueError(
            f"{field_name} length ({mask.shape[0]}) must match species dimension ({dim})"
        )
    return tuple(int(i) for i in np.where(mask.astype(bool))[0])


def _unique_indices(values, field_name: str) -> tuple[int, ...]:
    result: list[int] = []
    seen = set()
    for raw in values:
        idx = _extract_modifier_index(raw, field_name)
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        result.append(idx)
    return tuple(sorted(result))


def _extract_modifier_index(raw, field_name: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in ("species_idx", "modifier_idx", "idx"):
            if key in raw:
                return _extract_modifier_index(raw[key], f"{field_name}.{key}")
        raise ValueError(f"{field_name} modifier dict must include species_idx")
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"{field_name} entries must be integer species indices")
    try:
        return int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{field_name} entries must be integer species indices, got {raw!r}"
        ) from exc


def _unique_preserving_order(laws: list[str]) -> list[str]:
    """Return laws without duplicates while preserving config order."""
    seen = set()
    result = []
    for law in laws:
        if law in seen:
            continue
        seen.add(law)
        result.append(law)
    return result


def _configured_law_space(
    options: SINDyKineticOptions,
) -> tuple[list[str], list[str]]:
    """Return the configured KFI law space in stable order."""
    substrate_laws = _unique_preserving_order(list(options.substrate_laws))
    modifier_laws = _unique_preserving_order(list(options.modifier_laws))
    if not substrate_laws:
        raise ValueError("options.substrate_laws must not be empty")
    if not modifier_laws:
        raise ValueError("options.modifier_laws must not be empty")
    return substrate_laws, modifier_laws


def _enumerate_substrate_candidates(
    reactants: tuple[int, ...],
    options: SINDyKineticOptions,
    substrate_law: str,
) -> list[tuple[str, int | None, float | None]]:
    """Enumerate parameterized substrate-law candidates for one reaction."""
    if substrate_law == "mass_action":
        return [("mass_action", None, None)]

    if substrate_law == "mm":
        candidates: list[tuple[str, int | None, float | None]] = []
        for s_idx, order in enumerate(reactants):
            if order <= 0:
                continue
            for K_val in get_K_grid_for_species(options, s_idx):
                candidates.append(("mm", s_idx, K_val))
        return candidates

    raise ValueError(
        f"Unsupported substrate_law for KineSelectSR dictionary: {substrate_law!r}"
    )


def _enumerate_modifier_candidates(
    reactants: tuple[int, ...],
    modifier_indices: int | tuple[int, ...] | None,
    options: SINDyKineticOptions,
    modifier_law: str,
) -> list[tuple[str, int | None, float | None, int | None]]:
    """Enumerate parameterized modifier-law candidates for one reaction."""
    if modifier_law == "none":
        return [("none", None, None, None)]

    if modifier_indices is None:
        return []

    if isinstance(modifier_indices, (int, np.integer)):
        candidate_indices = (int(modifier_indices),)
    else:
        candidate_indices = tuple(sorted({int(idx) for idx in modifier_indices}))
    if not candidate_indices:
        return []

    valid_indices = []
    for modifier_idx in candidate_indices:
        if modifier_idx < 0 or modifier_idx >= len(reactants):
            continue
        if reactants[modifier_idx] > 0:
            continue
        valid_indices.append(modifier_idx)
    if not valid_indices:
        return []

    if modifier_law == "linear":
        return [("linear", modifier_idx, None, None) for modifier_idx in valid_indices]

    if modifier_law in {"hill_activation", "hill_inhibition"}:
        candidates: list[tuple[str, int | None, float | None, int | None]] = []
        for modifier_idx in valid_indices:
            for K_val in get_K_grid_for_species(options, modifier_idx):
                for n_val in options.hill_n_values:
                    candidates.append((modifier_law, modifier_idx, K_val, n_val))
        return candidates

    raise ValueError(
        f"Unsupported modifier_law for KineSelectSR dictionary: {modifier_law!r}"
    )


def _enumerate_reaction_variants(
    reactants: tuple[int, ...],
    products: tuple[int, ...],
    modifier_indices: int | tuple[int, ...] | None,
    options: SINDyKineticOptions,
) -> list[KineticReaction]:
    """Enumerate configured kinetic variants for one `(R, P, M)` gene."""
    substrate_laws, modifier_laws = _configured_law_space(options)
    variants: list[KineticReaction] = []

    for substrate_law in substrate_laws:
        substrate_candidates = _enumerate_substrate_candidates(
            reactants, options, substrate_law
        )
        for sub_law, sub_idx, sub_K in substrate_candidates:
            for modifier_law in modifier_laws:
                modifier_candidates = _enumerate_modifier_candidates(
                    reactants, modifier_indices, options, modifier_law
                )
                for mod_law, mod_idx, mod_K, mod_n in modifier_candidates:
                    if sub_idx is not None and sub_idx == mod_idx:
                        continue
                    variants.append(
                        KineticReaction(
                            reactants=reactants,
                            products=products,
                            substrate_law=sub_law,
                            substrate_idx=sub_idx,
                            substrate_K=sub_K,
                            modifier_law=mod_law,
                            modifier_idx=mod_idx,
                            modifier_K=mod_K,
                            modifier_n=mod_n,
                        )
                    )

    return variants


def build_gene_column_block(
    gene,
    X: np.ndarray,
    target_ctx: TargetContext,
    options: SINDyKineticOptions,
) -> list[tuple[KineticReaction, np.ndarray, float]]:
    """Build normalized dictionary columns for one reaction gene.

    Returns a list of ``(variant, normalized_column, original_norm)`` tuples.
    The caller assigns reaction-group indices when assembling a grouped
    dictionary.
    """
    run_options = resolve_sindy_options_for_X(options, X)

    return _build_gene_column_block_resolved(gene, X, target_ctx, run_options)


def _build_gene_column_block_resolved(
    gene,
    X: np.ndarray,
    target_ctx: TargetContext,
    options: SINDyKineticOptions,
) -> list[tuple[KineticReaction, np.ndarray, float]]:
    """Build one gene block assuming data-dependent options are already resolved."""
    T, D = X.shape
    X_dot_max_safe = target_ctx.X_dot_max_safe

    reactants, products, modifier_indices = parse_reaction_gene_candidates(gene)
    variants = _enumerate_reaction_variants(
        reactants, products, modifier_indices, options
    )

    stoich = np.array(products, dtype=int) - np.array(reactants, dtype=int)
    block: list[tuple[KineticReaction, np.ndarray, float]] = []

    for variant in variants:
        props = compute_rate_batch(variant, X)

        col = np.zeros(T * D, dtype=float)
        for i in range(D):
            if stoich[i] != 0:
                col[i::D] = props * stoich[i] / X_dot_max_safe[0, i]

        norm = float(np.linalg.norm(col))
        if norm > 1e-15:
            col_normed = col / norm
        else:
            col_normed = np.zeros_like(col)
            norm = 1.0

        block.append((variant, col_normed, norm))

    return block
