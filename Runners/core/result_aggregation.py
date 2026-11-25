"""Shared fixed-denominator reducers for paper result aggregation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from Metrics.reaction_annotation_metrics import (
    ReactionOptions,
    _counts_to_prf,
    compute_reaction_metrics,
    extract_true_reactions_from_annotation,
)


PARTITIONS = ("main", "challenge", "stress")


def safe_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else sum(values) / len(values)


def aggregate_fixed_denominator(
    rows: list[dict[str, Any]],
    *,
    partitions: Sequence[str] = PARTITIONS,
    trajectory_protocol: str = "default_fit_to_perturbed_rollout",
    supports_reaction_metrics: bool = True,
) -> dict[str, Any]:
    """Reduce trajectory/reaction rows without dropping failed systems."""
    trajectory_rows = [row for row in rows if row.get("trajectory_included")]
    reaction_denominator = [row for row in rows if row.get("reaction_metrics_enabled")]
    reaction_rows = [row for row in reaction_denominator if row.get("reaction_included")]
    tp = sum(int(row.get("reaction_tp") or 0) for row in reaction_rows)
    fp = sum(int(row.get("reaction_fp") or 0) for row in reaction_rows)
    fn = sum(int(row.get("reaction_fn") or 0) for row in reaction_rows)
    if reaction_rows:
        precision, recall, micro_f1, _ = _counts_to_prf(tp, fp, fn)
    else:
        precision = recall = micro_f1 = None

    by_partition: dict[str, Any] = {}
    for partition in partitions:
        selected = [row for row in rows if row.get("benchmark_partition") == partition]
        included = [row for row in selected if row.get("trajectory_included")]
        by_partition[str(partition)] = {
            "fixed_denominator": len(selected),
            "num_trajectory_included": len(included),
            "trajectory_success_rate": len(included) / len(selected) if selected else None,
            "rmse_mean": safe_mean(included, "rmse_mean"),
            "nrmse_mean": safe_mean(included, "nrmse_mean"),
            "r2_mean": safe_mean(included, "r2_mean"),
        }

    result = {
        "fixed_denominator": len(rows),
        "status_counts": dict(
            sorted(Counter(str(row.get("status")) for row in rows).items())
        ),
        "artifact_integrity_passed": all(
            row.get("artifact_integrity") for row in rows
        ),
        "trajectory_metrics": {
            "protocol": trajectory_protocol,
            "fixed_denominator": len(rows),
            "num_included": len(trajectory_rows),
            "success_rate": len(trajectory_rows) / len(rows) if rows else None,
            "rmse_mean": safe_mean(trajectory_rows, "rmse_mean"),
            "nrmse_mean": safe_mean(trajectory_rows, "nrmse_mean"),
            "r2_mean": safe_mean(trajectory_rows, "r2_mean"),
            "by_partition": by_partition,
        },
        "reaction_metrics": {
            "partitions": ["main", "challenge"],
            "fixed_denominator": len(reaction_denominator),
            "num_included": len(reaction_rows),
            "fit_success_rate": (
                sum(row.get("status") == "success" for row in reaction_denominator)
                / len(reaction_denominator)
                if reaction_denominator
                else None
            ),
            "macro_f1": safe_mean(reaction_rows, "reaction_f1"),
            "micro_tp": tp,
            "micro_fp": fp,
            "micro_fn": fn,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": micro_f1,
        },
    }

    if not supports_reaction_metrics:
        from .method_capabilities import REACTION_UNSUPPORTED
        result["reaction_metrics"].update(
            supported=False, exclusion_reason=REACTION_UNSUPPORTED,
            micro_precision=None, micro_recall=None, micro_f1=None,
        )
    return result


def empty_prediction_scores(annotation_path: Path) -> dict[str, Any]:
    """Score an empty prediction under the frozen reaction metric options."""
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    true_reactions = extract_true_reactions_from_annotation(annotation)
    return compute_reaction_metrics(
        true_reactions,
        [],
        ReactionOptions(
            strict_kinetic_law=True,
            strict_hill_n=False,
            strict_K=False,
            term_tolerance=1.0e-12,
            projection_mode="term",
            stoich_projection_mode="unit",
            kineselectsr_source_policy="auto",
        ),
        dim=len(annotation.get("state_names") or []),
    )


__all__ = [
    "PARTITIONS",
    "aggregate_fixed_denominator",
    "empty_prediction_scores",
    "safe_mean",
]
