"""AST node counting for kinetic ODE models."""

from __future__ import annotations

import numpy as np

from .reaction import KineSelectSRModel, validate_model


def compute_complexity(
    model: KineSelectSRModel,
    threshold: float = 1e-14,
) -> int:
    """Compute AST node count for a fitted kinetic model."""
    if model.rate_constants is None:
        raise ValueError("model.rate_constants must be set before computing complexity")

    D = validate_model(model)
    if D == 0:
        return 0

    total = 0
    term_counts = [0] * D

    for rxn, k in zip(model.reactions, model.rate_constants):
        if abs(k) <= threshold:
            continue

        stoich = np.array(rxn.products, dtype=int) - np.array(rxn.reactants, dtype=int)
        kinetic_nodes = _compute_kinetic_nodes(rxn)

        for i in range(D):
            s_ir = int(stoich[i])
            if s_ir == 0:
                continue

            a_ir = abs(s_ir)
            stoich_nodes = 1 if a_ir == 1 else a_ir + 1
            total += 1 + stoich_nodes + 1 + kinetic_nodes  # sign + stoich + k + kinetic
            term_counts[i] += 1

    # Phase 2: '+' operators between terms for the same species
    for c in term_counts:
        if c > 1:
            total += c - 1

    return total


def _compute_kinetic_nodes(rxn) -> int:
    """Compute kinetic_nodes for a single reaction."""
    reactants = rxn.reactants
    O_r = sum(reactants)

    # Substrate part
    if rxn.substrate_law == "mass_action":
        substrate_nodes = O_r
    elif rxn.substrate_law == "mm":
        substrate_nodes = max(O_r - 1, 0) + 5
    else:
        substrate_nodes = O_r  # fallback

    # Modifier connector
    modifier_connector = 1 if rxn.modifier_law != "none" else 0

    # Modifier part
    if rxn.modifier_law == "none":
        modifier_nodes = 0
    elif rxn.modifier_law == "linear":
        modifier_nodes = 1
    elif rxn.modifier_law in ("hill_activation", "hill_inhibition"):
        n = rxn.modifier_n or 1
        modifier_nodes = 3 * n + 2
    else:
        modifier_nodes = 0  # fallback

    return substrate_nodes + modifier_connector + modifier_nodes
