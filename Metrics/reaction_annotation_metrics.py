from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any

import sympy as sp

METRIC_VERSION = "sysbiosr-reaction-1.0.0-rc.1"


@dataclass(frozen=True)
class ReactionOptions:
    strict_kinetic_law: bool = True
    strict_hill_n: bool = False
    strict_K: bool = False
    term_tolerance: float = 1e-12
    projection_mode: str = "term"
    stoich_projection_mode: str = "unit"
    kineselectsr_source_policy: str = "auto"


LAW_ALIASES = {
    "hill": "hill_saturation",
    "hill_act": "hill_activation",
    "hill_activation": "hill_activation",
    "hill_inh": "hill_inhibition",
    "hill_inhibition": "hill_inhibition",
    "hill_saturation": "hill_saturation",
    "mm": "mm",
    "michaelis_menten": "mm",
    "rational": "rational_saturation",
    "rational_saturation": "rational_saturation",
    "mass-action": "mass_action",
    "mass_action": "mass_action",
    "linear": "linear",
    "none": "none",
}

AUTO_KINESELECTSR_PATHS = [
    ("equations", "metadata", "kineselectsr", "selected_variants"),
    ("equations", "metadata", "selected_variants"),
    ("selected_model", "selected_variants"),
    ("selected_model", "reactions"),
    ("model", "selected_variants"),
    ("model", "reactions"),
    ("generated_equation", "metadata", "selected_variants"),
    ("generated_equation", "metadata", "reactions"),
    ("reactions",),
]

KINESELECTSR_SELECTED_VARIANT_PATHS = [path for path in AUTO_KINESELECTSR_PATHS if "selected_variants" in path]
KINESELECTSR_REACTION_PATHS = [path for path in AUTO_KINESELECTSR_PATHS if "reactions" in path]


def normalize_law(law: Any) -> str:
    text = str(law or "unknown").strip()
    return LAW_ALIASES.get(text, LAW_ALIASES.get(text.lower(), text))


def _get_nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _is_active_predicted_reaction(rxn: dict[str, Any], threshold: float = 1e-14) -> bool:
    if "active" in rxn:
        return bool(rxn["active"])
    if "selected" in rxn:
        return bool(rxn["selected"])
    if "is_selected" in rxn:
        return bool(rxn["is_selected"])
    for key in ("rate_constant", "k"):
        if key not in rxn or rxn[key] is None:
            continue
        value = _as_float_or_none(rxn[key])
        return value is not None and abs(value) > threshold
    return True


def _normalize_kineselectsr_reaction(rxn: dict[str, Any]) -> dict[str, Any]:
    reactants = [int(v) for v in rxn.get("reactants", [])]
    products = [int(v) for v in rxn.get("products", [])]
    if "substrates" in rxn or "modifiers" in rxn:
        return {
            "reactants": reactants,
            "products": products,
            "substrates": [
                _normalize_entry_record(entry)
                for entry in (rxn.get("substrates") or [])
                if isinstance(entry, dict)
            ],
            "modifiers": [
                _normalize_entry_record(entry)
                for entry in (rxn.get("modifiers") or [])
                if isinstance(entry, dict)
            ],
        }

    substrates: list[dict[str, Any]] = []
    substrate_law = normalize_law(rxn.get("substrate_law") or "mass_action")
    substrate_idx = _as_int_or_none(rxn.get("substrate_idx"))
    substrate_K = rxn.get("substrate_K")

    for idx, order in enumerate(reactants):
        if order <= 0:
            continue
        if substrate_law == "mm" and substrate_idx == idx:
            if order > 1:
                substrates.append(
                    {"species_idx": idx, "law": "mass_action", "order": order - 1, "K": None, "n": None}
                )
            substrates.append(
                {"species_idx": idx, "law": "mm", "order": 1, "K": substrate_K, "n": None}
            )
        elif substrate_law == "hill_saturation" and substrate_idx == idx:
            # Compact substrate Hill records encode only the Hill factor.
            # Annotation-style x_i * Hill(x_i) terms must be represented with
            # explicit substrates=[mass_action, hill_saturation], which the
            # explicit-record branch above preserves verbatim.
            substrates.append(
                {
                    "species_idx": idx,
                    "law": substrate_law,
                    "order": 1,
                    "K": substrate_K,
                    "n": rxn.get("substrate_n"),
                }
            )
        elif substrate_law == "rational_saturation" and substrate_idx == idx:
            if order > 1:
                substrates.append(
                    {"species_idx": idx, "law": "mass_action", "order": order - 1, "K": None, "n": None}
                )
            substrates.append(
                {
                    "species_idx": idx,
                    "law": substrate_law,
                    "order": 1,
                    "K": substrate_K,
                    "n": rxn.get("substrate_n"),
                }
            )
        else:
            substrates.append(
                {"species_idx": idx, "law": "mass_action", "order": order, "K": None, "n": None}
            )

    modifiers: list[dict[str, Any]] = []
    modifier_law = normalize_law(rxn.get("modifier_law") or "none")
    modifier_idx = _as_int_or_none(rxn.get("selected_modifier_idx", rxn.get("modifier_idx")))
    if modifier_law != "none" and modifier_idx is not None:
        modifiers.append(
            {
                "species_idx": modifier_idx,
                "law": modifier_law,
                "order": 1,
                "K": rxn.get("modifier_K"),
                "n": rxn.get("modifier_n"),
            }
        )

    return {
        "reactants": reactants,
        "products": products,
        "substrates": substrates,
        "modifiers": modifiers,
    }


def _normalize_entry_record(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "species_idx": _as_int_or_none(entry.get("species_idx")),
        "law": normalize_law(entry.get("law")),
        "order": _as_int_or_none(entry.get("order")) or 1,
        "K": entry.get("K"),
        "n": _as_int_or_none(entry.get("n")),
    }


def _extract_first_reaction_list(
    payload: dict[str, Any],
    paths: list[tuple[str, ...]],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    for path in paths:
        value = _get_nested(payload, path)
        if isinstance(value, list):
            records = [dict(item) for item in value if isinstance(item, dict)]
            return records, ".".join(path)
    return None, None


def _extract_kineselectsr_reaction_records(
    payload: dict[str, Any],
    *,
    source_policy: str = "auto",
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if source_policy == "auto":
        paths = AUTO_KINESELECTSR_PATHS
    elif source_policy == "selected_variants":
        paths = KINESELECTSR_SELECTED_VARIANT_PATHS
    elif source_policy == "reactions":
        paths = KINESELECTSR_REACTION_PATHS
    else:
        raise ValueError(
            "kineselectsr_source_policy must be one of: auto, selected_variants, reactions"
        )
    return _extract_first_reaction_list(payload, paths)


def extract_true_reactions_from_annotation(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    reactions = annotation.get("reactions", [])
    if not isinstance(reactions, list):
        raise ValueError("annotation['reactions'] must be a list")
    return [dict(rxn) for rxn in reactions if isinstance(rxn, dict)]


def extract_predicted_reactions(
    payload: dict[str, Any],
    method: str,
    dim: int,
    *,
    term_tolerance: float = 1e-12,
    kineselectsr_source_policy: str = "auto",
) -> tuple[list[dict[str, Any]], str, str | None]:
    pred_rxns, source, error, _metadata = extract_predicted_reactions_with_metadata(
        payload,
        method,
        dim,
        term_tolerance=term_tolerance,
        kineselectsr_source_policy=kineselectsr_source_policy,
    )
    return pred_rxns, source, error


def extract_predicted_reactions_with_metadata(
    payload: dict[str, Any],
    method: str,
    dim: int,
    *,
    term_tolerance: float = 1e-12,
    kineselectsr_source_policy: str = "auto",
) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
    if method == "kineselectsr":
        records, source = _extract_kineselectsr_reaction_records(
            payload,
            source_policy=kineselectsr_source_policy,
        )
        if records is None:
            return (
                [],
                "kineselectsr_explicit",
                "no explicit KineSelectSR reaction list found",
                {
                    "kineselectsr_source_policy": kineselectsr_source_policy,
                    "kineselectsr_native_reaction_count": 0,
                    "kineselectsr_active_reaction_count": 0,
                },
            )
        active_records = [rxn for rxn in records if _is_active_predicted_reaction(rxn)]
        return (
            [_normalize_kineselectsr_reaction(rxn) for rxn in active_records],
            source or "kineselectsr_explicit",
            None,
            {
                "kineselectsr_source_policy": kineselectsr_source_policy,
                "kineselectsr_native_reaction_count": len(records),
                "kineselectsr_active_reaction_count": len(active_records),
            },
        )

    if method in {"sisr", "sisr-nnls-pareto", "reactive_sindy"}:
        try:
            return (
                extract_predicted_reactions_from_generated_equation(
                    payload,
                    dim,
                    term_tolerance=term_tolerance,
                ),
                "generated_equation",
                None,
                {},
            )
        except Exception as exc:
            return [], "generated_equation", repr(exc), {}

    return [], "unsupported_method", f"reaction extraction is not enabled for method={method!r}", {}


def _candidate_generated_equation_values(payload: dict[str, Any]) -> list[Any]:
    paths = [
        ("generated_equation", "best_substituted"),
        ("generated_equation", "best"),
        ("equations", "metadata", "best_substituted"),
        ("equations", "best"),
        ("equations", "metadata", "kineselectsr", "best_substituted"),
    ]
    values = []
    for path in paths:
        value = _get_nested(payload, path)
        if value is not None:
            values.append(value)
    return values


_DX_PREFIX_RE = re.compile(r"^\s*d?x_?(\d+)\s*/\s*dt\s*=\s*(.*)$")


def _equation_value_to_rhs_list(value: Any, dim: int) -> list[tuple[int | None, str]]:
    if isinstance(value, list):
        parts = [str(item) for item in value]
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split("|")]
    else:
        raise ValueError(f"unsupported generated equation value type: {type(value).__name__}")

    out: list[tuple[int | None, str]] = []
    for part in parts:
        if not part:
            continue
        match = _DX_PREFIX_RE.match(part)
        if match:
            out.append((int(match.group(1)), match.group(2).strip()))
        else:
            out.append((None, part.strip()))
    if not out:
        raise ValueError("generated equation is empty")
    if len(out) > dim:
        out = out[:dim]
    return out


def _extract_monomial(term: sp.Expr, x_symbols: list[sp.Symbol]) -> tuple[float, dict[int, int]]:
    powers: dict[int, int] = {}
    coefficient = term
    power_dict = term.as_powers_dict()
    for idx, symbol in enumerate(x_symbols):
        exponent = power_dict.get(symbol, 0)
        if exponent == 0:
            continue
        if not exponent.is_integer:
            raise ValueError(f"non-integer exponent in term {term!s}: {symbol}^{exponent}")
        exponent_int = int(exponent)
        if exponent_int < 0:
            raise ValueError(f"negative exponent in term {term!s}: {symbol}^{exponent_int}")
        powers[idx] = exponent_int
        coefficient = coefficient / (symbol ** exponent_int)

    coefficient = sp.N(sp.simplify(coefficient))
    if coefficient.free_symbols:
        raise ValueError(f"non-numeric coefficient in term {term!s}: {coefficient!s}")
    return float(coefficient), powers


def _term_to_reaction(
    *,
    equation_idx: int,
    coefficient: float,
    powers: dict[int, int],
    dim: int,
    source_term: str,
) -> dict[str, Any]:
    reactants = [0] * dim
    products = [0] * dim
    substrates: list[dict[str, Any]] = []
    modifiers: list[dict[str, Any]] = []

    if coefficient < 0:
        reactants[equation_idx] = 1
        consumed_order = powers.get(equation_idx, 0)
        if consumed_order > 0:
            substrates.append(
                {
                    "species_idx": equation_idx,
                    "law": "mass_action",
                    "order": consumed_order,
                    "K": None,
                    "n": None,
                }
            )
        for species_idx, order in sorted(powers.items()):
            if species_idx == equation_idx:
                continue
            modifiers.append(
                {
                    "species_idx": species_idx,
                    "law": "linear",
                    "order": order,
                    "K": None,
                    "n": None,
                }
            )
    else:
        products[equation_idx] = 1
        for species_idx, order in sorted(powers.items()):
            modifiers.append(
                {
                    "species_idx": species_idx,
                    "law": "linear",
                    "order": order,
                    "K": None,
                    "n": None,
                }
            )

    return {
        "equation_idx": equation_idx,
        "reactants": reactants,
        "products": products,
        "coefficient": abs(coefficient),
        "substrates": substrates,
        "modifiers": modifiers,
        "source_term": source_term,
    }


def extract_predicted_reactions_from_generated_equation(
    payload: dict[str, Any],
    dim: int,
    term_tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    values = _candidate_generated_equation_values(payload)
    if not values:
        raise ValueError("no generated equation field found")

    last_error: Exception | None = None
    x_symbols = [sp.Symbol(f"x_{idx}") for idx in range(dim)]
    local_dict = {f"x_{idx}": symbol for idx, symbol in enumerate(x_symbols)}

    for value in values:
        try:
            rhs_entries = _equation_value_to_rhs_list(value, dim)
            reactions: list[dict[str, Any]] = []
            for position, (maybe_idx, rhs) in enumerate(rhs_entries):
                equation_idx = position if maybe_idx is None else maybe_idx
                if equation_idx < 0 or equation_idx >= dim:
                    raise ValueError(f"equation index out of range: {equation_idx}")
                expr = sp.expand(sp.sympify(rhs.replace("^", "**"), locals=local_dict))
                for term in sp.Add.make_args(expr):
                    coefficient, powers = _extract_monomial(term, x_symbols)
                    if abs(coefficient) < term_tolerance:
                        continue
                    reactions.append(
                        _term_to_reaction(
                            equation_idx=equation_idx,
                            coefficient=coefficient,
                            powers=powers,
                            dim=dim,
                            source_term=str(term),
                        )
                    )
            return reactions
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(f"failed to parse generated equation: {last_error!r}")


def _zero_vector(dim: int) -> list[int]:
    return [0] * dim


def _unit_vector(dim: int, idx: int) -> list[int]:
    out = [0] * dim
    out[idx] = 1
    return out


def _normalize_vector(value: Any, *, dim: int) -> list[int]:
    raw = list(value or [])
    if len(raw) < dim:
        raw = [*raw, *([0] * (dim - len(raw)))]
    if len(raw) > dim:
        raw = raw[:dim]
    return [int(v) for v in raw]


def _normalize_reaction_record(rxn: dict[str, Any], *, dim: int) -> dict[str, Any]:
    reactants = _normalize_vector(rxn.get("reactants"), dim=dim)
    products = _normalize_vector(rxn.get("products"), dim=dim)
    explicit_substrates = "substrates" in rxn
    substrates = [
        _normalize_entry_record(entry)
        for entry in (rxn.get("substrates") or [])
        if isinstance(entry, dict)
    ]
    modifiers = [
        _normalize_entry_record(entry)
        for entry in (rxn.get("modifiers") or [])
        if isinstance(entry, dict)
    ]

    if not explicit_substrates:
        substrates.extend(
            {
                "species_idx": idx,
                "law": "mass_action",
                "order": order,
                "K": None,
                "n": None,
            }
            for idx, order in enumerate(reactants)
            if order > 0
        )

    return {
        "reactants": reactants,
        "products": products,
        "substrates": substrates,
        "modifiers": modifiers,
        "_explicit_substrates": explicit_substrates,
    }


def _dependency_to_modifier(entry: dict[str, Any]) -> dict[str, Any]:
    law = normalize_law(entry.get("law"))
    K = entry.get("K")
    n = _as_int_or_none(entry.get("n"))
    if law == "mass_action":
        law = "linear"
        K = None
        n = None
    elif law == "mm":
        law = "hill_activation"
        n = 1
    elif law == "hill_saturation":
        law = "hill_activation"
    return {
        "species_idx": _as_int_or_none(entry.get("species_idx")),
        "law": law,
        "order": _as_int_or_none(entry.get("order")) or 1,
        "K": K,
        "n": n,
    }


def _negative_term_entries(
    rxn: dict[str, Any],
    *,
    consumed_idx: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    substrates: list[dict[str, Any]] = []
    modifiers: list[dict[str, Any]] = []

    for entry in rxn["substrates"]:
        if entry.get("species_idx") == consumed_idx:
            substrates.append(dict(entry))
        else:
            modifiers.append(_dependency_to_modifier(entry))

    if not substrates and not rxn.get("_explicit_substrates") and rxn["reactants"][consumed_idx] > 0:
        substrates.append(
            {
                "species_idx": consumed_idx,
                "law": "mass_action",
                "order": rxn["reactants"][consumed_idx],
                "K": None,
                "n": None,
            }
        )

    modifiers.extend(_dependency_to_modifier(entry) for entry in rxn["modifiers"])
    return substrates, modifiers


def _positive_term_entries(rxn: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modifiers = [
        *(_dependency_to_modifier(entry) for entry in rxn["substrates"]),
        *(_dependency_to_modifier(entry) for entry in rxn["modifiers"]),
    ]
    return [], modifiers


def project_reaction_to_term_level(
    rxn: dict[str, Any],
    *,
    dim: int,
    stoich_projection_mode: str = "unit",
) -> list[dict[str, Any]]:
    if stoich_projection_mode not in {"duplicate", "unit"}:
        raise ValueError("stoich_projection_mode must be one of: duplicate, unit")

    normalized = _normalize_reaction_record(rxn, dim=dim)
    net = [p - r for p, r in zip(normalized["products"], normalized["reactants"])]
    terms: list[dict[str, Any]] = []

    for sign in (-1, 1):
        for idx, delta in enumerate(net):
            if delta == 0 or (delta < 0) != (sign < 0):
                continue
            multiplicity = abs(delta) if stoich_projection_mode == "duplicate" else 1
            for _ in range(multiplicity):
                if delta < 0:
                    substrates, modifiers = _negative_term_entries(normalized, consumed_idx=idx)
                    terms.append(
                        {
                            "reactants": _unit_vector(dim, idx),
                            "products": _zero_vector(dim),
                            "substrates": substrates,
                            "modifiers": modifiers,
                        }
                    )
                else:
                    substrates, modifiers = _positive_term_entries(normalized)
                    terms.append(
                        {
                            "reactants": _zero_vector(dim),
                            "products": _unit_vector(dim, idx),
                            "substrates": substrates,
                            "modifiers": modifiers,
                        }
                    )
    return terms


def project_reactions_to_term_level(
    reactions: list[dict[str, Any]],
    *,
    dim: int,
    stoich_projection_mode: str = "unit",
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for rxn in reactions:
        projected.extend(
            project_reaction_to_term_level(
                rxn,
                dim=dim,
                stoich_projection_mode=stoich_projection_mode,
            )
        )
    return projected


def _sort_entry_key(entry: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple("<none>" if item is None else str(item) for item in entry)


def _normalize_K(value: Any) -> Any:
    value = _as_float_or_none(value)
    if value is None:
        return None
    return round(value, 12)


def _normalize_entry(entry: dict[str, Any], *, options: ReactionOptions) -> tuple[Any, ...]:
    species_idx = _as_int_or_none(entry.get("species_idx"))
    law = normalize_law(entry.get("law"))
    order = _as_int_or_none(entry.get("order"))
    n = _as_int_or_none(entry.get("n"))
    K = _normalize_K(entry.get("K")) if options.strict_K else None

    if not options.strict_kinetic_law:
        return (species_idx,)
    if law in {"mass_action", "linear"}:
        law_detail: Any = order if order is not None else 1
    elif law in {"hill_activation", "hill_inhibition", "hill_saturation"}:
        law_detail = n if options.strict_hill_n else None
    else:
        law_detail = order if order is not None else None

    if options.strict_K:
        return (species_idx, law, law_detail, K)
    return (species_idx, law, law_detail)


def canonicalize_reaction(
    rxn: dict[str, Any],
    *,
    strict_kinetic_law: bool = True,
    strict_hill_n: bool = False,
    strict_K: bool = False,
) -> tuple[Any, ...]:
    options = ReactionOptions(
        strict_kinetic_law=strict_kinetic_law,
        strict_hill_n=strict_hill_n,
        strict_K=strict_K,
    )
    reactants = tuple(int(v) for v in (rxn.get("reactants") or []))
    products = tuple(int(v) for v in (rxn.get("products") or []))

    has_explicit_entries = "substrates" in rxn or "modifiers" in rxn
    substrates = [dict(entry) for entry in (rxn.get("substrates") or []) if isinstance(entry, dict)]
    modifiers = [dict(entry) for entry in (rxn.get("modifiers") or []) if isinstance(entry, dict)]
    if not has_explicit_entries:
        substrates.extend(
            {
                "species_idx": idx,
                "law": "mass_action",
                "order": order,
                "K": None,
                "n": None,
            }
            for idx, order in enumerate(reactants)
            if order > 0
        )

    if not strict_kinetic_law:
        participation = sorted(
            {_normalize_entry(entry, options=options) for entry in [*substrates, *modifiers]},
            key=_sort_entry_key,
        )
        return (reactants, products, tuple(participation))

    substrate_entries = tuple(
        sorted((_normalize_entry(entry, options=options) for entry in substrates), key=_sort_entry_key)
    )
    modifier_entries = tuple(
        sorted((_normalize_entry(entry, options=options) for entry in modifiers), key=_sort_entry_key)
    )
    return (reactants, products, substrate_entries, modifier_entries)


def _infer_dim(*reaction_groups: list[dict[str, Any]]) -> int:
    dim = 0
    for reactions in reaction_groups:
        for rxn in reactions:
            dim = max(dim, len(rxn.get("reactants") or []), len(rxn.get("products") or []))
    return dim


def _counts_to_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    """Convert TP/FP/FN counts to precision, recall, F1, and Jaccard."""
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if fn == 0 else 0.0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 1.0 if fp == 0 else 0.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    jaccard = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    return precision, recall, f1, jaccard


def compute_reaction_metrics(
    true_rxns: list[dict[str, Any]],
    pred_rxns: list[dict[str, Any]],
    options: ReactionOptions,
    *,
    dim: int | None = None,
) -> dict[str, Any]:
    if dim is None:
        dim = _infer_dim(true_rxns, pred_rxns)

    true_native_count = len(true_rxns)
    pred_native_count = len(pred_rxns)
    if options.projection_mode == "term":
        eval_true_rxns = project_reactions_to_term_level(
            true_rxns,
            dim=dim,
            stoich_projection_mode=options.stoich_projection_mode,
        )
        eval_pred_rxns = project_reactions_to_term_level(
            pred_rxns,
            dim=dim,
            stoich_projection_mode=options.stoich_projection_mode,
        )
    elif options.projection_mode == "native":
        eval_true_rxns = true_rxns
        eval_pred_rxns = pred_rxns
    else:
        raise ValueError("projection_mode must be one of: term, native")

    true_signatures = [
        canonicalize_reaction(
            rxn,
            strict_kinetic_law=options.strict_kinetic_law,
            strict_hill_n=options.strict_hill_n,
            strict_K=options.strict_K,
        )
        for rxn in eval_true_rxns
    ]
    pred_signatures = [
        canonicalize_reaction(
            rxn,
            strict_kinetic_law=options.strict_kinetic_law,
            strict_hill_n=options.strict_hill_n,
            strict_K=options.strict_K,
        )
        for rxn in eval_pred_rxns
    ]
    true_counter = Counter(true_signatures)
    pred_counter = Counter(pred_signatures)
    all_keys = set(true_counter) | set(pred_counter)
    tp = sum(min(true_counter[key], pred_counter[key]) for key in all_keys)
    total_true = sum(true_counter.values())
    total_pred = sum(pred_counter.values())
    fp = total_pred - tp
    fn = total_true - tp

    precision, recall, f1, jaccard = _counts_to_prf(tp, fp, fn)

    return {
        "reaction_TP": tp,
        "reaction_FP": fp,
        "reaction_FN": fn,
        "true_reaction_count": total_true,
        "pred_reaction_count": total_pred,
        "true_native_reaction_count": true_native_count,
        "pred_native_reaction_count": pred_native_count,
        "true_projected_reaction_count": len(eval_true_rxns),
        "pred_projected_reaction_count": len(eval_pred_rxns),
        "reaction_precision": precision,
        "reaction_recall": recall,
        "reaction_f1": f1,
        "reaction_jaccard": jaccard,
        "reaction_iou": jaccard,
        # reaction_acc is kept as a deprecated alias for reaction_jaccard.
        # There is no TN-defined accuracy for reaction detection.
        "reaction_acc": jaccard,
        "reaction_acc_deprecated_alias": jaccard,
        "reaction_metric_note": "reaction_acc is Jaccard/IoU; no TN-defined accuracy",
        "reaction_projection_mode": options.projection_mode,
        "reaction_stoich_projection_mode": options.stoich_projection_mode,
        "true_signatures": true_signatures,
        "pred_signatures": pred_signatures,
        "true_projected_reactions": eval_true_rxns,
        "pred_projected_reactions": eval_pred_rxns,
    }


def _counter_prf(
    true_counter: Counter,
    pred_counter: Counter,
) -> tuple[int, int, int, float, float, float]:
    keys = set(true_counter) | set(pred_counter)
    tp = sum(min(true_counter[key], pred_counter[key]) for key in keys)
    total_true = sum(true_counter.values())
    total_pred = sum(pred_counter.values())
    fp = total_pred - tp
    fn = total_true - tp
    precision, recall, f1, _ = _counts_to_prf(tp, fp, fn)
    return int(tp), int(fp), int(fn), precision, recall, f1


def _modifier_counter(
    reactions: list[dict[str, Any]],
    *,
    dim: int,
    linear_only: bool,
) -> Counter:
    counter: Counter = Counter()
    for rxn in reactions:
        compact_rxn = (
            _normalize_kineselectsr_reaction(rxn)
            if "modifier_law" in rxn and "modifiers" not in rxn
            else rxn
        )
        normalized = _normalize_reaction_record(compact_rxn, dim=dim)
        skeleton = (
            tuple(normalized["reactants"]),
            tuple(normalized["products"]),
        )
        for modifier in normalized["modifiers"]:
            species_idx = _as_int_or_none(modifier.get("species_idx"))
            if species_idx is None:
                continue
            law = normalize_law(modifier.get("law"))
            if law == "none":
                continue
            if linear_only and law != "linear":
                continue
            law_key = "linear" if linear_only else "any"
            counter[(*skeleton, species_idx, law_key)] += 1
    return counter


def compute_modifier_diagnostics(
    true_rxns: list[dict[str, Any]],
    pred_rxns: list[dict[str, Any]],
    *,
    dim: int | None = None,
) -> dict[str, Any]:
    """Compute modifier/regulator recovery diagnostics.

    The generic modifier score ignores modifier law type and matches by native
    reaction skeleton plus modifier species. The linear score restricts both
    true and predicted records to linear modifiers.
    """
    if dim is None:
        dim = _infer_dim(true_rxns, pred_rxns)

    true_mod = _modifier_counter(true_rxns, dim=dim, linear_only=False)
    pred_mod = _modifier_counter(pred_rxns, dim=dim, linear_only=False)
    true_linear = _modifier_counter(true_rxns, dim=dim, linear_only=True)
    pred_linear = _modifier_counter(pred_rxns, dim=dim, linear_only=True)

    modifier_tp, modifier_fp, modifier_fn, modifier_p, modifier_r, modifier_f1 = _counter_prf(
        true_mod, pred_mod
    )
    linear_tp, linear_fp, linear_fn, linear_p, linear_r, linear_f1 = _counter_prf(
        true_linear, pred_linear
    )

    selected_linear = int(sum(pred_linear.values()))
    true_linear_count = int(sum(true_linear.values()))
    return {
        "modifier_TP": modifier_tp,
        "modifier_FP": modifier_fp,
        "modifier_FN": modifier_fn,
        "modifier_precision": modifier_p,
        "modifier_recall": modifier_r,
        "modifier_F1": modifier_f1,
        "linear_modifier_TP": linear_tp,
        "linear_modifier_FP": linear_fp,
        "linear_modifier_FN": linear_fn,
        "linear_modifier_precision": linear_p,
        "linear_modifier_recall": linear_r,
        "linear_modifier_F1": linear_f1,
        "true_modifier_count": int(sum(true_mod.values())),
        "pred_modifier_count": int(sum(pred_mod.values())),
        "true_linear_modifier_count": true_linear_count,
        "pred_linear_modifier_count": selected_linear,
        "linear_modifier_overselection_ratio": selected_linear / max(true_linear_count, 1),
    }
