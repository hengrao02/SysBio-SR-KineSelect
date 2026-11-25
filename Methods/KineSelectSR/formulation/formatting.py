"""Convert a KineSelectSRModel to human-readable equation strings."""

from __future__ import annotations

import numpy as np

from .reaction import KineSelectSRModel, KineSelectSRKineticVariant


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_model(
    model: KineSelectSRModel,
    state_names: list[str] | None = None,
    threshold: float = 1e-14,
) -> str:
    """Format a fitted model with symbolic rate constants (k0, k1, ...)."""
    _validate(model)
    names = _resolve_names(model, state_names)
    active = _active_indices(model, threshold)
    terms = _build_terms(model, names, active)
    parts = _assemble_parts(terms, names, active)
    return " | ".join(parts)


def format_model_substituted(
    model: KineSelectSRModel,
    state_names: list[str] | None = None,
    threshold: float = 1e-14,
) -> list[str]:
    """Format a fitted model with numerical rate constants."""
    _validate(model)
    names = _resolve_names(model, state_names)
    active = _active_indices(model, threshold)
    ks = np.asarray(model.rate_constants)
    terms = _build_terms(model, names, active, k_values=ks)
    parts = _assemble_parts(terms, names, active)
    return parts


# ---------------------------------------------------------------------------
# Internal: validation & name resolution
# ---------------------------------------------------------------------------

def _validate(model: KineSelectSRModel) -> None:
    if model.rate_constants is None:
        raise ValueError("model.rate_constants is None; fit rate constants first")


def _resolve_names(model: KineSelectSRModel, state_names: list[str] | None) -> list[str]:
    D = _dimension(model)
    return state_names if state_names is not None else [f"x_{i}" for i in range(D)]


def _active_indices(model: KineSelectSRModel, threshold: float) -> list[int]:
    ks = np.asarray(model.rate_constants)
    return [int(i) for i in np.where(np.abs(ks) > threshold)[0]]


# ---------------------------------------------------------------------------
# Internal: core formatting logic
# ---------------------------------------------------------------------------

def _build_terms(
    model: KineSelectSRModel,
    names: list[str],
    active_indices: list[int],
    k_values: np.ndarray | None = None,
) -> list[list[str]]:
    """Build per-dimension term lists."""
    D = _dimension(model)
    terms: list[list[str]] = [[] for _ in range(D)]

    for r_idx in active_indices:
        rxn = model.reactions[r_idx]
        propensity = _format_propensity(rxn, names)

        if k_values is not None:
            k_val = float(k_values[r_idx])
            k_str = f"{k_val:.6g}"
        else:
            k_str = f"k{r_idx}"

        stoich = tuple(
            int(p) - int(r) for p, r in zip(rxn.products, rxn.reactants)
        )
        for dim in range(D):
            s = stoich[dim]
            if s == 0:
                continue
            if s == 1:
                term = f"{k_str} * {propensity}"
            elif s == -1:
                term = f"-{k_str} * {propensity}"
            else:
                term = f"{s}*{k_str} * {propensity}"
            terms[dim].append(term)

    return terms


def _assemble_parts(
    terms: list[list[str]],
    names: list[str],
    active_indices: list[int],
) -> list[str]:
    """Join per-dimension terms into equation strings."""
    parts: list[str] = []
    for dim, var in enumerate(names):
        if not terms[dim]:
            parts.append(f"d{var}/dt = 0")
        else:
            rhs = " + ".join(terms[dim])
            rhs = rhs.replace("+ -", "- ")
            parts.append(f"d{var}/dt = {rhs}")
    return parts


def _dimension(model: KineSelectSRModel) -> int:
    for rxn in model.reactions:
        return len(rxn.reactants)
    return 0


# ---------------------------------------------------------------------------
# Internal: propensity formatting
# ---------------------------------------------------------------------------

def _format_propensity(rxn: KineSelectSRKineticVariant, names: list[str]) -> str:
    parts: list[str] = []
    substrate_part = _format_substrate_factor(rxn, names)
    if substrate_part:
        parts.append(substrate_part)
    modifier_part = _format_modifier_factor(rxn, names)
    if modifier_part and modifier_part != "1":
        parts.append(modifier_part)
    if not parts:
        return "1"
    return " * ".join(parts)


def _format_substrate_factor(rxn: KineSelectSRKineticVariant, names: list[str]) -> str:
    factors: list[str] = []
    for i, order in enumerate(rxn.reactants):
        order = int(order)
        if order <= 0:
            continue
        var = names[i]
        is_substrate = (
            rxn.substrate_law == "mm" and rxn.substrate_idx == i
        )
        if is_substrate:
            K_str = f"{rxn.substrate_K:.6g}" if rxn.substrate_K is not None else "?"
            if order == 1:
                factors.append(f"{var} / ({K_str} + {var})")
            else:
                factors.append(f"{var}^{order} / ({K_str} + {var})")
        else:
            factors.append(var if order == 1 else f"{var}^{order}")
    return " * ".join(factors) if factors else "1"


def _format_modifier_factor(rxn: KineSelectSRKineticVariant, names: list[str]) -> str:
    law = rxn.modifier_law
    if law == "none":
        return "1"
    idx = rxn.modifier_idx
    if idx is None:
        return "1"
    var = names[idx]
    if law == "linear":
        return var
    K = rxn.modifier_K
    n = rxn.modifier_n
    K_str = f"{K:.6g}" if K is not None else "?"
    n_int = int(n) if n is not None else 1
    if law == "hill_activation":
        if n_int == 1:
            return f"{var} / ({K_str} + {var})"
        return f"{var}^{n_int} / ({K_str}^{n_int} + {var}^{n_int})"
    if law == "hill_inhibition":
        if n_int == 1:
            return f"{K_str} / ({K_str} + {var})"
        return f"{K_str}^{n_int} / ({K_str}^{n_int} + {var}^{n_int})"
    return "1"
