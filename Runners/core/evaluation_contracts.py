"""Pure helpers shared by online runs and saved-model evaluation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from Metrics.reaction_annotation_metrics import (
    METRIC_VERSION as REACTION_METRIC_VERSION,
    ReactionOptions,
    compute_modifier_diagnostics,
    compute_reaction_metrics,
    extract_predicted_reactions_with_metadata,
    extract_true_reactions_from_annotation,
)
from .method_capabilities import REACTION_UNSUPPORTED, method_supports_reaction_metrics


def to_jsonable(value: Any) -> Any:
    """Convert supported result values to strict JSON-compatible values."""
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def stable_input_model_payload(
    *,
    method_name: str,
    method_payload: dict[str, Any],
    num_states: int,
) -> dict[str, Any]:
    """Build the timing-free fitted-model identity used by metric artifacts."""
    equation = method_payload.get("generated_equation")
    metadata = (method_payload.get("equations") or {}).get("metadata") or {}
    if method_name in {
        "sindy",
        "sindy_pi",
        "odeformer",
        "ffx",
        "pysr",
        "sisr",
        "reactive_sindy",
    }:
        return {
            "method": method_name,
            "equation": equation,
            "model": metadata.get("selected_model"),
        }
    if method_name == "kineselectsr":
        candidate = metadata.get("selected_model") or metadata.get("kineselectsr") or {}
        return {
            "method": method_name,
            "equation": equation,
            "model": {
                key: candidate.get(key)
                for key in ("rate_constants", "selected_variants", "proposed_genes")
            },
        }

    try:
        predicted, source, extraction_error, _ = (
            extract_predicted_reactions_with_metadata(
                method_payload,
                method_name,
                num_states,
                term_tolerance=1.0e-12,
                kineselectsr_source_policy="auto",
            )
        )
    except Exception as exc:  # identity remains defined for invalid models
        predicted, source = [], "extraction_failed"
        extraction_error = type(exc).__name__
    predicted = sorted(
        to_jsonable(predicted),
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    )
    return {
        "identity_version": "input-model-v1",
        "method": method_name,
        "generated_equation": method_payload.get("generated_equation"),
        "predicted_reactions": predicted,
        "prediction_source": source,
        "prediction_extraction_error": extraction_error,
    }


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", ""}:
        return False
    return bool(value)


def reaction_evaluation(
    *,
    method_name: str,
    method_payload: dict[str, Any],
    annotation: dict[str, Any] | None,
    fit_config: dict[str, Any],
    num_states: int,
) -> dict[str, Any]:
    """Evaluate reaction recovery using the frozen SysBioSR metric contract."""
    enabled_value = fit_config.get("reaction_metrics")
    if enabled_value is None:
        enabled = fit_config.get("benchmark_partition") in {"main", "challenge"}
    else:
        enabled = coerce_bool(enabled_value)

    supported = method_supports_reaction_metrics(method_name)
    enabled = enabled and supported
    record: dict[str, Any] = {
        "metric_version": REACTION_METRIC_VERSION,
        "enabled": enabled,
        "included_in_aggregation": enabled,
        "exclusion_reason": (
            REACTION_UNSUPPORTED
            if not supported
            else (None if enabled else "partition_reaction_metrics_false")
        ),
    }
    if not enabled:
        return record
    if not isinstance(annotation, dict):
        record.update(
            {
                "status": "invalid",
                "included_in_aggregation": False,
                "exclusion_reason": "reaction_annotation_missing",
            }
        )
        return record

    options = ReactionOptions(
        strict_kinetic_law=True,
        strict_hill_n=False,
        strict_K=False,
        term_tolerance=1.0e-12,
        projection_mode="term",
        stoich_projection_mode="unit",
        kineselectsr_source_policy="auto",
    )
    true_reactions = extract_true_reactions_from_annotation(annotation)
    predicted, source, extraction_error, extraction_metadata = (
        extract_predicted_reactions_with_metadata(
            method_payload,
            method_name,
            num_states,
            term_tolerance=options.term_tolerance,
            kineselectsr_source_policy=options.kineselectsr_source_policy,
        )
    )
    scores = compute_reaction_metrics(
        true_reactions,
        predicted,
        options,
        dim=num_states,
    )
    modifier_diagnostics = compute_modifier_diagnostics(
        true_reactions,
        predicted,
        dim=num_states,
    )
    record.update(
        {
            "status": "success" if extraction_error is None else "empty_prediction",
            "prediction_source": source,
            "prediction_extraction_error": extraction_error,
            "prediction_extraction_metadata": extraction_metadata,
            "options": asdict(options),
            "scores": scores,
            "modifier_diagnostics": modifier_diagnostics,
        }
    )
    return record


__all__ = [
    "coerce_bool",
    "reaction_evaluation",
    "stable_input_model_payload",
    "to_jsonable",
]
