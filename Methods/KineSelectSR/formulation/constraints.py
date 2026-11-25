"""Reaction validity constraints and enumeration."""

from __future__ import annotations

import itertools
import numbers
import numpy as np

from .reaction import FormulationConfig, KineSelectSRKineticVariant, KineSelectSRReactionGene


# ---------------------------------------------------------------------------
# Stoichiometric vector validation and overlap prevention
# ---------------------------------------------------------------------------

def has_reactant_product_overlap(
    reactants: tuple[int, ...],
    products: tuple[int, ...],
) -> bool:
    """Return True if any species appears in both reactants and products."""
    return any(r > 0 and p > 0 for r, p in zip(reactants, products))


def has_noncanonical_source_products(
    reactants: tuple[int, ...],
    products: tuple[int, ...],
) -> bool:
    """Return True if a source reaction is not a canonical singleton 0 -> x_i."""
    if sum(reactants) != 0:
        return False
    return sum(products) != 1 or max(products, default=0) != 1


def get_K_grid_for_species(
    config: FormulationConfig,
    species_idx: int,
) -> list[float]:
    """Return the K grid for species_idx, respecting species_K_grid if set."""
    if config.species_K_grid is not None:
        if species_idx < 0 or species_idx >= len(config.species_K_grid):
            raise ValueError(
                f"species_idx {species_idx} out of range for species_K_grid "
                f"of length {len(config.species_K_grid)}"
            )
        ks = config.species_K_grid[species_idx]
        if not ks:
            raise ValueError(f"species_K_grid[{species_idx}] is empty")
        return list(ks)
    return list(config.K_grid)


# ---------------------------------------------------------------------------
# Quantile K grid generation
# ---------------------------------------------------------------------------

def compute_species_quantile_K_grid(
    X: np.ndarray,
    quantiles: list[float] | None = None,
    positive_eps: float = 1e-12,
    k_floor: float = 1e-12,
    positive_rel_eps: float = 1e-6,
    dedup_rtol: float = 0.10,
) -> list[list[float]]:
    """Compute per-species quantile-based K grids from trajectory data."""
    if quantiles is None:
        quantiles = [0.05, 0.50]
    else:
        quantiles = sorted(quantiles)

    _T, D = X.shape
    grids: list[list[float]] = []
    for i in range(D):
        x_col = X[:, i]
        x_max = float(np.max(np.abs(x_col)))
        threshold = max(positive_eps, positive_rel_eps * x_max)
        positive = x_col[x_col > threshold]
        if len(positive) == 0:
            grids.append([k_floor])
        else:
            qs = np.quantile(positive, quantiles)
            deduped: list[float] = []
            for q in qs:
                K = max(float(q), k_floor)
                _append_if_distinct(deduped, K, dedup_rtol)
            grids.append(deduped)
    return grids


def _append_if_distinct(values: list[float], k: float, rtol: float) -> None:
    """Append k to values if not already present (within relative tolerance)."""
    for existing in values:
        if np.isclose(k, existing, rtol=rtol, atol=0.0):
            return
    values.append(k)


# ---------------------------------------------------------------------------
# Gene validation
# ---------------------------------------------------------------------------

def is_valid_gene(
    gene: KineSelectSRReactionGene | tuple | list | dict,
    config: FormulationConfig,
) -> bool:
    """Check whether a reaction gene satisfies structural/stoichiometric constraints."""
    if isinstance(gene, KineSelectSRReactionGene):
        reactants = gene.reactants
        products = gene.products
        modifier_idx = gene.modifier_idx
    elif isinstance(gene, dict):
        reactants = tuple(gene["reactants"])
        products = tuple(gene["products"])
        modifier_idx = gene.get("modifier_idx", None)
    elif isinstance(gene, (tuple, list)):
        if len(gene) == 3:
            reactants, products, modifier_idx = gene
            reactants = tuple(reactants)
            products = tuple(products)
        else:
            reactants, products = gene
            reactants = tuple(reactants)
            products = tuple(products)
            modifier_idx = None
    else:
        return False

    # Stoichiometric order checks
    if sum(reactants) > config.max_reactant_order:
        return False
    if sum(products) > config.max_product_order:
        return False
    if any(x > config.max_stoich_per_species for x in reactants):
        return False
    if any(x > config.max_stoich_per_species for x in products):
        return False

    # Exclude identity reactions
    if reactants == products:
        return False

    # Exclude reactant/product overlap
    if has_reactant_product_overlap(reactants, products):
        return False

    # Exclude non-canonical source reactions
    if has_noncanonical_source_products(reactants, products):
        return False

    # Exclude modifier as reactant
    if modifier_idx is not None:
        if modifier_idx < 0 or modifier_idx >= len(reactants):
            return False
        if reactants[modifier_idx] > 0:
            return False

    return True


# ---------------------------------------------------------------------------
# Variant validation
# ---------------------------------------------------------------------------

def is_valid(reaction: KineSelectSRKineticVariant, config: FormulationConfig) -> bool:
    """Check whether a KineSelectSRKineticVariant satisfies all configuration constraints."""
    # First check the stoichiometric gene constraints
    gene_ok = is_valid_gene(
        KineSelectSRReactionGene(
            reactants=reaction.reactants,
            products=reaction.products,
            modifier_idx=reaction.modifier_idx
        ),
        config
    )
    if not gene_ok:
        return False

    # --- Substrate law constraints ---
    if reaction.substrate_law not in config.substrate_laws:
        return False

    # MM-specific: K must be in species-specific K grid (or global fallback)
    if reaction.substrate_law == "mm":
        if reaction.substrate_K not in get_K_grid_for_species(
            config, reaction.substrate_idx
        ):
            return False

    # --- Modifier law constraints ---
    if reaction.modifier_law not in config.modifier_laws:
        return False

    # Hill-specific: K must be in species-specific K grid, n in hill_n_values
    if reaction.modifier_law in ("hill_activation", "hill_inhibition"):
        if reaction.modifier_K not in get_K_grid_for_species(
            config, reaction.modifier_idx
        ):
            return False
        if reaction.modifier_n not in config.hill_n_values:
            return False

    return True


# ---------------------------------------------------------------------------
# Stoichiometric vector enumeration helper
# ---------------------------------------------------------------------------

def _enumerate_side_vectors(dim: int, max_stoich: int, max_order: int) -> list[tuple[int, ...]]:
    """Enumerate valid stoichiometric vectors for one side."""
    vectors: list[tuple[int, ...]] = []
    for coeffs in itertools.product(range(max_stoich + 1), repeat=dim):
        if sum(coeffs) > max_order:
            continue
        vectors.append(tuple(coeffs))
    return vectors


# ---------------------------------------------------------------------------
# Full enumeration of variants
# ---------------------------------------------------------------------------

def enumerate_valid_reactions(
    dim: int,
    config: FormulationConfig,
) -> list[KineSelectSRKineticVariant]:
    """Enumerate all valid KineSelectSRKineticVariant objects for a given dimension."""
    if dim <= 0:
        return []

    reactant_vectors = _enumerate_side_vectors(
        dim, config.max_stoich_per_species, config.max_reactant_order,
    )
    product_vectors = _enumerate_side_vectors(
        dim, config.max_stoich_per_species, config.max_product_order,
    )

    substrate_candidates: list[tuple[str, int | None, float | None]] = []
    substrate_candidates.append(("mass_action", None, None))
    if "mm" in config.substrate_laws:
        for s_idx in range(dim):
            for K in get_K_grid_for_species(config, s_idx):
                substrate_candidates.append(("mm", s_idx, K))

    modifier_candidates: list[tuple[str, int | None, float | None, int | None]] = []
    modifier_candidates.append(("none", None, None, None))
    if "linear" in config.modifier_laws:
        for m_idx in range(dim):
            modifier_candidates.append(("linear", m_idx, None, None))
    if "hill_activation" in config.modifier_laws:
        for m_idx in range(dim):
            for K in get_K_grid_for_species(config, m_idx):
                for n in config.hill_n_values:
                    modifier_candidates.append(("hill_activation", m_idx, K, n))
    if "hill_inhibition" in config.modifier_laws:
        for m_idx in range(dim):
            for K in get_K_grid_for_species(config, m_idx):
                for n in config.hill_n_values:
                    modifier_candidates.append(("hill_inhibition", m_idx, K, n))

    variants: list[KineSelectSRKineticVariant] = []

    for r_vec in reactant_vectors:
        for p_vec in product_vectors:
            # Skip identity
            if r_vec == p_vec:
                continue

            # Skip reactant/product overlap
            if has_reactant_product_overlap(r_vec, p_vec):
                continue

            # Skip non-canonical source reactions
            if has_noncanonical_source_products(r_vec, p_vec):
                continue

            for sub_law, sub_idx, sub_K in substrate_candidates:
                # MM substrate must be a reactant
                if sub_law == "mm":
                    if r_vec[sub_idx] == 0:
                        continue

                for mod_law, mod_idx, mod_K, mod_n in modifier_candidates:
                    # modifier must NOT be a reactant
                    if mod_idx is not None and r_vec[mod_idx] > 0:
                        continue
                    # substrate and modifier must differ
                    if sub_idx is not None and sub_idx == mod_idx:
                        continue

                    try:
                        rxn = KineSelectSRKineticVariant(
                            reactants=r_vec,
                            products=p_vec,
                            substrate_law=sub_law,
                            substrate_idx=sub_idx,
                            substrate_K=sub_K,
                            modifier_law=mod_law,
                            modifier_idx=mod_idx,
                            modifier_K=mod_K,
                            modifier_n=mod_n,
                        )
                        if is_valid(rxn, config):
                            variants.append(rxn)
                    except (ValueError, TypeError):
                        continue

    return variants
