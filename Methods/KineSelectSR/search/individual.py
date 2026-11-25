from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
import numpy as np
from Methods.KineSelectSR.formulation import KineSelectSRReactionGene, KineSelectSRKineticVariant, KineSelectSRModel, KineSelectSREvalMetrics


def _canonical_float(value: Any) -> str:
    """Encode a numeric value without platform-dependent decimal rendering."""
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number.hex()


def canonical_candidate_payload(candidate: "KineSelectSRIndividual") -> dict[str, Any]:
    """Return the stable identity payload used by every final selector.

    The payload deliberately excludes list position and process completion order.
    Hexadecimal float tokens avoid platform-dependent decimal rendering.
    """
    genes = [
        {
            "modifier_idx": (
                None if gene.modifier_idx is None else int(gene.modifier_idx)
            ),
            "products": [int(value) for value in gene.products],
            "reactants": [int(value) for value in gene.reactants],
        }
        for gene in candidate.genes
    ]
    variants = None
    if candidate.selected_variants is not None:
        variants = [
            {
                "modifier_K": (
                    None
                    if variant.modifier_K is None
                    else _canonical_float(variant.modifier_K)
                ),
                "modifier_idx": (
                    None if variant.modifier_idx is None else int(variant.modifier_idx)
                ),
                "modifier_law": str(variant.modifier_law),
                "modifier_n": (
                    None if variant.modifier_n is None else int(variant.modifier_n)
                ),
                "products": [int(value) for value in variant.products],
                "reactants": [int(value) for value in variant.reactants],
                "substrate_K": (
                    None
                    if variant.substrate_K is None
                    else _canonical_float(variant.substrate_K)
                ),
                "substrate_idx": (
                    None if variant.substrate_idx is None else int(variant.substrate_idx)
                ),
                "substrate_law": str(variant.substrate_law),
            }
            for variant in candidate.selected_variants
        ]

    rates = None
    if candidate.rate_constants is not None:
        rates = [_canonical_float(value) for value in candidate.rate_constants]

    metric_payload = None
    if candidate.metrics is not None:
        metric_payload = {
            "complexity": int(candidate.metrics.complexity),
            "deriv_scaled_nmse": _canonical_float(candidate.metrics.deriv_scaled_nmse),
            "deriv_scaled_nmse_raw": _canonical_float(
                candidate.metrics.deriv_scaled_nmse_raw
            ),
            "n_active_reactions": int(candidate.metrics.n_active_reactions),
            "rate_constants": [
                _canonical_float(value) for value in candidate.metrics.rate_constants
            ],
            "rollout_nmse": _canonical_float(candidate.metrics.rollout_nmse),
        }
    return {
        "genes": genes,
        "selected_variants": variants,
        "rate_constants": rates,
        "metrics": metric_payload,
    }


def _canonical_candidate_json(candidate: "KineSelectSRIndividual") -> str:
    """Serialize the identity payload before computing ``candidate_id``."""
    payload = canonical_candidate_payload(candidate)
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_candidate_id(candidate: "KineSelectSRIndividual") -> str:
    encoded = _canonical_candidate_json(candidate).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

@dataclass
class KineSelectSRIndividual:
    genes: tuple[KineSelectSRReactionGene, ...]
    selected_variants: tuple[KineSelectSRKineticVariant, ...] | None = None
    rate_constants: np.ndarray | None = None
    metrics: KineSelectSREvalMetrics | None = None
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self):
        if len(set(self.genes)) != len(self.genes):
            raise ValueError("Duplicate genes detected in KineSelectSRIndividual")
        
        def sort_key(gene):
            m = -1 if gene.modifier_idx is None else gene.modifier_idx
            return (gene.reactants, gene.products, m)
            
        self.genes = tuple(sorted(self.genes, key=sort_key))

    @property
    def n_reactions(self) -> int:
        return len(self.genes)

    @property
    def is_evaluated(self) -> bool:
        return self.metrics is not None

    @property
    def reaction_key(self) -> frozenset[KineSelectSRReactionGene]:
        return frozenset(self.genes)

    def to_model(self) -> KineSelectSRModel:
        reactions = self.selected_variants if self.selected_variants is not None else ()
        return KineSelectSRModel(
            reactions=reactions,
            rate_constants=self.rate_constants.copy() if self.rate_constants is not None else None,
            deriv_loss=self.metrics.deriv_scaled_nmse if self.metrics is not None else None
        )

    def __hash__(self) -> int:
        return hash(self.genes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KineSelectSRIndividual):
            return False
        return self.genes == other.genes
