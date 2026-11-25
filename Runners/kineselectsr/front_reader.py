"""Strict numerical reader for saved KineSelectSR Pareto fronts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import zstandard

from Methods.KineSelectSR.formulation import (
    KineSelectSREvalMetrics,
    KineSelectSRKineticVariant,
    KineSelectSRReactionGene,
)
from Methods.KineSelectSR.search import KineSelectSRIndividual
from Runners.kineselectsr.artifacts import (
    METHOD_VERSION,
    SEARCH_CONTRACT_VERSION,
    _candidate_record,
    _front_jsonl,
)


_CANDIDATE_RE = re.compile(r"^candidate-[0-9]{6,}$")
_MAX_FRONT_BYTES = 512 * 1024 * 1024


class SavedFrontError(RuntimeError):
    """A fail-closed saved-front integrity or identity error."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"{self.reason}: {self.detail}")


@dataclass(frozen=True)
class SavedFront:
    individuals: tuple[KineSelectSRIndividual, ...]
    records: tuple[dict[str, Any], ...]
    selection_objective: str


@dataclass(frozen=True)
class SavedFrontContent:
    content: bytes


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise SavedFrontError("json_duplicate_key", f"duplicate key {key!r}")
        payload[key] = value
    return payload


def strict_json_loads(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SavedFrontError("json_nonfinite", f"{label}: {value}")
            ),
        )
    except SavedFrontError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SavedFrontError("json_invalid", f"{label}: {exc}") from exc


def strict_json_file(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise SavedFrontError("artifact_missing", str(path))
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SavedFrontError("artifact_read_failed", f"{path}: {exc}") from exc
    return strict_json_loads(text, label=path.name)


def _require_dict(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SavedFrontError("candidate_schema_invalid", f"{field} must be an object")
    return value


def _require_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SavedFrontError("candidate_schema_invalid", f"{field} must be an array")
    return value


def _require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SavedFrontError("candidate_schema_invalid", f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise SavedFrontError(
            "candidate_schema_invalid", f"{field} must be >= {minimum}"
        )
    return value


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, field=field)


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SavedFrontError("candidate_schema_invalid", f"{field} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise SavedFrontError("candidate_schema_invalid", f"{field} must be finite")
    return output


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field=field)


def _stoichiometry(value: Any, *, field: str, dimension: int) -> tuple[int, ...]:
    entries = _require_list(value, field=field)
    if len(entries) != dimension:
        raise SavedFrontError(
            "candidate_dimension_mismatch",
            f"{field} has {len(entries)} entries, expected {dimension}",
        )
    return tuple(
        _require_int(entry, field=f"{field}[{index}]", minimum=0)
        for index, entry in enumerate(entries)
    )


def deserialize_candidate(record: Any, *, dimension: int) -> KineSelectSRIndividual:
    """Reconstruct one candidate and prove lossless round-trip identity."""
    row = _require_dict(record, field="candidate")
    if row.get("model_format") != "sysbiosr-model-v2":
        raise SavedFrontError(
            "unsupported_model_format",
            "Candidate record must have model_format='sysbiosr-model-v2'.",
        )
    candidate_id = row.get("candidate_id")
    if (
        not isinstance(candidate_id, str)
        or _CANDIDATE_RE.fullmatch(candidate_id) is None
    ):
        raise SavedFrontError(
            "candidate_schema_invalid", "candidate_id must be a local candidate number"
        )

    gene_rows = _require_list(row.get("proposed_genes"), field="proposed_genes")
    genes: list[KineSelectSRReactionGene] = []
    for index, raw_gene in enumerate(gene_rows):
        gene = _require_dict(raw_gene, field=f"proposed_genes[{index}]")
        if (
            _require_int(gene.get("index"), field=f"proposed_genes[{index}].index")
            != index
        ):
            raise SavedFrontError(
                "candidate_schema_invalid", "proposed gene indices must be contiguous"
            )
        genes.append(
            KineSelectSRReactionGene(
                reactants=_stoichiometry(
                    gene.get("reactants"),
                    field=f"proposed_genes[{index}].reactants",
                    dimension=dimension,
                ),
                products=_stoichiometry(
                    gene.get("products"),
                    field=f"proposed_genes[{index}].products",
                    dimension=dimension,
                ),
                modifier_idx=_optional_int(
                    gene.get("proposed_modifier_idx"),
                    field=f"proposed_genes[{index}].proposed_modifier_idx",
                ),
            )
        )

    variant_rows = _require_list(
        row.get("selected_variants"), field="selected_variants"
    )
    variants: list[KineSelectSRKineticVariant] = []
    for index, raw_variant in enumerate(variant_rows):
        variant = _require_dict(raw_variant, field=f"selected_variants[{index}]")
        if (
            _require_int(
                variant.get("index"), field=f"selected_variants[{index}].index"
            )
            != index
        ):
            raise SavedFrontError(
                "candidate_schema_invalid",
                "selected variant indices must be contiguous",
            )
        modifier_idx = _optional_int(
            variant.get("selected_modifier_idx"),
            field=f"selected_variants[{index}].selected_modifier_idx",
        )
        variants.append(
            KineSelectSRKineticVariant(
                reactants=_stoichiometry(
                    variant.get("reactants"),
                    field=f"selected_variants[{index}].reactants",
                    dimension=dimension,
                ),
                products=_stoichiometry(
                    variant.get("products"),
                    field=f"selected_variants[{index}].products",
                    dimension=dimension,
                ),
                substrate_law=str(variant.get("substrate_law")),
                substrate_idx=_optional_int(
                    variant.get("substrate_idx"),
                    field=f"selected_variants[{index}].substrate_idx",
                ),
                substrate_K=_optional_float(
                    variant.get("substrate_K"),
                    field=f"selected_variants[{index}].substrate_K",
                ),
                modifier_law=str(variant.get("modifier_law")),
                modifier_idx=modifier_idx,
                modifier_K=_optional_float(
                    variant.get("modifier_K"),
                    field=f"selected_variants[{index}].modifier_K",
                ),
                modifier_n=_optional_int(
                    variant.get("modifier_n"),
                    field=f"selected_variants[{index}].modifier_n",
                ),
            )
        )

    raw_rates = _require_list(row.get("rate_constants"), field="rate_constants")
    rates = np.asarray(
        [
            _finite_float(value, field=f"rate_constants[{index}]")
            for index, value in enumerate(raw_rates)
        ],
        dtype=float,
    )
    if len(rates) != len(variants):
        raise SavedFrontError(
            "candidate_schema_invalid",
            "rate_constants length must equal selected_variants length",
        )

    deriv = _finite_float(row.get("deriv_scaled_nmse"), field="deriv_scaled_nmse")
    deriv_raw = _finite_float(
        row.get("deriv_scaled_nmse_raw"), field="deriv_scaled_nmse_raw"
    )
    rollout_raw = row.get("rollout_nmse")
    rollout_nmse = (
        float("inf")
        if rollout_raw is None
        else _finite_float(rollout_raw, field="rollout_nmse")
    )
    complexity = _require_int(row.get("complexity"), field="complexity", minimum=0)
    active_count = _require_int(
        row.get("n_active_reactions"), field="n_active_reactions", minimum=0
    )
    observed_active = int(np.count_nonzero(np.abs(rates) > 1.0e-14))
    if active_count != observed_active:
        raise SavedFrontError(
            "candidate_active_count_mismatch",
            f"declared={active_count}, observed={observed_active}",
        )

    diagnostics = row.get("sindy_diagnostics") or {}
    if not isinstance(diagnostics, dict):
        raise SavedFrontError(
            "candidate_schema_invalid", "sindy_diagnostics must be an object"
        )
    individual = KineSelectSRIndividual(
        genes=tuple(genes),
        selected_variants=tuple(variants),
        rate_constants=rates,
        metrics=KineSelectSREvalMetrics(
            deriv_scaled_nmse=deriv,
            deriv_scaled_nmse_raw=deriv_raw,
            rollout_nmse=rollout_nmse,
            complexity=complexity,
            rate_constants=rates.copy(),
            n_active_reactions=active_count,
        ),
        diagnostics=dict(diagnostics),
    )
    regenerated = _candidate_record(individual, candidate_id)
    if regenerated != row:
        raise SavedFrontError(
            "candidate_roundtrip_mismatch",
            f"candidate {candidate_id} is not losslessly canonical",
        )
    return individual


def read_saved_front_content(
    *,
    front_path: Path,
    search_summary: dict[str, Any],
) -> SavedFrontContent:
    """Validate storage, decompress zstd, then validate content identity."""
    if (
        search_summary.get("method_version") != METHOD_VERSION
        or search_summary.get("search_contract_version") != SEARCH_CONTRACT_VERSION
    ):
        raise SavedFrontError(
            "unsupported_search_version",
            "use the matching development version for historical fronts",
        )
    if not front_path.is_file() or front_path.is_symlink():
        raise SavedFrontError("front_missing", str(front_path))
    storage = front_path.read_bytes()
    try:
        content = zstandard.ZstdDecompressor().decompress(
            storage,
            max_output_size=_MAX_FRONT_BYTES,
            allow_extra_data=False,
        )
    except zstandard.ZstdError as exc:
        raise SavedFrontError("front_zstd_invalid", repr(exc)) from exc
    if len(content) > _MAX_FRONT_BYTES:
        raise SavedFrontError("front_too_large", str(len(content)))
    return SavedFrontContent(content=content)


def parse_saved_front(
    *,
    front_content: SavedFrontContent,
    search_summary: dict[str, Any],
    dimension: int,
) -> SavedFront:
    """Validate canonical JSONL and reconstruct every saved candidate."""
    content = front_content.content
    records: list[dict[str, Any]] = []
    if content:
        if not content.endswith(b"\n"):
            raise SavedFrontError("front_noncanonical_jsonl", "missing final newline")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SavedFrontError("front_utf8_invalid", repr(exc)) from exc
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line:
                raise SavedFrontError(
                    "front_noncanonical_jsonl", f"blank line {line_number}"
                )
            record = strict_json_loads(line, label=f"front line {line_number}")
            records.append(_require_dict(record, field=f"front line {line_number}"))

    individuals: list[KineSelectSRIndividual] = []
    for record in records:
        try:
            individuals.append(deserialize_candidate(record, dimension=dimension))
        except SavedFrontError:
            raise
        except (IndexError, TypeError, ValueError) as exc:
            raise SavedFrontError(
                "candidate_schema_invalid", type(exc).__name__
            ) from exc
    candidate_ids = [str(record["candidate_id"]) for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SavedFrontError(
            "duplicate_candidate_id", "front candidate IDs are not unique"
        )
    reaction_keys = [individual.reaction_key for individual in individuals]
    if len(reaction_keys) != len(set(reaction_keys)):
        raise SavedFrontError(
            "duplicate_reaction_key", "front reaction keys are not unique"
        )

    objective = str(search_summary.get("selection_objective") or "")
    if objective not in {"deriv", "rollout"}:
        raise SavedFrontError(
            "search_identity_mismatch", f"unknown selection objective {objective!r}"
        )
    regenerated, regenerated_records = _front_jsonl(individuals, objective)
    if regenerated != content or regenerated_records != records:
        raise SavedFrontError(
            "front_noncanonical_jsonl",
            "record rendering or deterministic front ordering differs",
        )
    declared_size = search_summary.get("pareto_front_size")
    if declared_size != len(records):
        raise SavedFrontError(
            "front_size_mismatch",
            f"declared={declared_size}, observed={len(records)}",
        )
    return SavedFront(
        individuals=tuple(individuals),
        records=tuple(records),
        selection_objective=objective,
    )


def read_saved_front(
    *,
    front_path: Path,
    search_summary: dict[str, Any],
    dimension: int,
) -> SavedFront:
    """Convenience wrapper for complete saved-front validation."""
    front_content = read_saved_front_content(
        front_path=front_path,
        search_summary=search_summary,
    )
    return parse_saved_front(
        front_content=front_content,
        search_summary=search_summary,
        dimension=dimension,
    )


__all__ = [
    "SavedFront",
    "SavedFrontContent",
    "SavedFrontError",
    "deserialize_candidate",
    "parse_saved_front",
    "read_saved_front",
    "read_saved_front_content",
    "strict_json_file",
    "strict_json_loads",
]
