"""Aggregate current SysBioSR results under the fixed benchmark protocol."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from DataAdapters import BenchmarkDefinition, benchmark_definition_path, benchmark_root
from Runners.core.common import build_result_path
from Runners.core.artifact_validation import _safe_load_json, validate_saved_result
from Runners.core.method_capabilities import (
    method_supports_reaction_metrics,
    REACTION_UNSUPPORTED,
)
from Runners.core.result_aggregation import (
    PARTITIONS,
    aggregate_fixed_denominator,
    empty_prediction_scores,
)
from Runners.core.inputs import atomic_json


def _index_row(
    *,
    result_path: Path,
    method: str,
    experiment_id: str,
    dataset_version: str,
    partition: str,
    system_id: str,
    seed: int,
    annotation_path: Path,
    reaction_metrics_enabled: bool,
    result_label=None,
    benchmark_spec=None,
    system_dir: Path | None = None,
) -> dict[str, Any]:
    supported = method_supports_reaction_metrics(method)
    reaction_metrics_enabled = bool(reaction_metrics_enabled and supported)
    row = {
        "method": method,
        "experiment_id": experiment_id,
        "dataset_version": dataset_version,
        "benchmark_partition": partition,
        "system_id": system_id,
        "seed": seed,
        "result_path": result_label or str(result_path),
        "status": "invalid",
        "failure_reason": "missing_summary",
        "artifact_integrity": False,
        "trajectory_included": False,
        "reaction_metrics_enabled": reaction_metrics_enabled,
        "reaction_included": False,
        "trajectory_status": "invalid",
        "rmse_mean": None,
        "nrmse_mean": None,
        "r2_mean": None,
    }
    if not supported:
        row["reaction_exclusion_reason"] = REACTION_UNSUPPORTED
    saved = _safe_load_json(result_path)
    scores = {}
    if saved:
        expected = {
            "method": method,
            "experiment_id": experiment_id,
            "dataset_version": dataset_version,
            "benchmark_partition": partition,
            "system_id": system_id,
            "random_seed": seed,
            "trajectory_name": "default",
            "benchmark_spec": benchmark_spec,
            "system_dir": system_dir,
        }
        rejection = validate_saved_result(
            saved, output_json=result_path, expected=expected
        )
        row.update(
            status=saved.get("status", "invalid"),
            failure_reason=saved.get("failure_reason"),
            artifact_integrity=rejection is None,
            artifact_integrity_error=rejection,
            run_id=saved.get("run_id"),
        )
        if rejection:
            # Preserve the existing nominal fit-status counts. An unusable
            # artifact/evaluation excludes scores; changed experiment inputs
            # or configuration additionally invalidate the fit identity.
            identity_failure = rejection.startswith(
                "configuration_mismatch:"
            ) or rejection in {
                "unsupported_result_format",
                "result_schema_invalid",
                "state_order_changed",
                "trajectory_metadata_changed",
                "trajectory_inputs_changed",
                "evaluation_inputs_changed",
                "benchmark_definition_changed",
                "fit_inputs_changed",
            }
            if identity_failure:
                row.update(status="invalid", failure_reason=rejection)
        elif saved.get("status") == "success":
            records = [
                r
                for r in saved.get("evaluations", [])
                if r.get("eval_trajectory_name") == "perturbed"
            ]
            if len(records) == 1:
                evaluation = _safe_load_json(
                    result_path.parent / records[0]["output_json_relative_to_fit"]
                )
                if evaluation and evaluation.get("status") == "success":
                    row.update(trajectory_status="success", trajectory_included=True)
                    overall = (evaluation.get("metrics") or {}).get("overall") or {}
                    for key in ("rmse_mean", "nrmse_mean", "r2_mean"):
                        row[key] = overall.get(key)
            reaction = saved.get("reaction_evaluation") or {}
            if reaction.get("enabled") and reaction.get("included_in_aggregation"):
                scores = reaction.get("scores") or {}
    if reaction_metrics_enabled:
        reported = bool(scores)
        if not scores:
            scores = empty_prediction_scores(annotation_path)
        row.update(
            reaction_included=True,
            reaction_score_source=(
                "reported_prediction"
                if reported
                else "empty_prediction_for_failed_or_missing_run"
            ),
            reaction_tp=scores.get("reaction_TP"),
            reaction_fp=scores.get("reaction_FP"),
            reaction_fn=scores.get("reaction_FN"),
            reaction_f1=scores.get("reaction_f1"),
        )
    return row


def aggregate_results(
    *,
    results_root,
    dataset_root,
    experiment_id,
    method,
    seed,
    output_dir,
    benchmark_definition=None,
):
    root = benchmark_root(dataset_root)
    definition = BenchmarkDefinition(
        benchmark_definition or benchmark_definition_path(root)
    )
    rows = []
    configurations = []
    for partition in PARTITIONS:
        definition.validate_partition(root / partition, partition=partition)
        for member in definition.partition_rows(partition):
            path = build_result_path(
                results_root,
                experiment_id=experiment_id,
                method_name=method,
                dataset_version=definition.dataset_version,
                benchmark_partition=partition,
                system_id=member.system_id,
                trajectory_name="default",
                seed=seed,
                run_mode="batch",
                config_stem=partition,
            )
            row = _index_row(
                result_path=path,
                method=method,
                experiment_id=experiment_id,
                dataset_version=definition.dataset_version,
                partition=partition,
                system_id=member.system_id,
                seed=seed,
                annotation_path=root
                / partition
                / member.system_id
                / "reaction_annotation.json",
                reaction_metrics_enabled=member.reaction_metrics,
                result_label=path.relative_to(results_root).as_posix(),
                benchmark_spec=definition.spec,
                system_dir=root / partition / member.system_id,
            )
            rows.append(row)
            saved = _safe_load_json(path)
            if saved and row["artifact_integrity"]:
                config = {
                    k: v
                    for k, v in saved["run_config"].items()
                    if k not in {"benchmark_partition", "reaction_metrics"}
                }
                if config not in configurations:
                    configurations.append(config)
    if len(configurations) > 1:
        raise ValueError(
            "Inconsistent scientific configurations in one experiment; use separate experiment_id values"
        )
    summary = aggregate_fixed_denominator(
        rows, supports_reaction_metrics=method_supports_reaction_metrics(method)
    )
    summary.update(
        schema_version=2,
        experiment_id=experiment_id,
        method=method,
        seed=seed,
        dataset_version=definition.dataset_version,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_index.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    )
    atomic_json(output_dir / "summary.json", summary)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset/SysBioSR"))
    parser.add_argument("--benchmark-definition", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--method",
        choices=(
            "kineselectsr",
            "reactive_sindy",
            "sisr",
            "sindy",
            "pysr",
            "sindy_pi",
            "odeformer",
            "ffx",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        _, summary = aggregate_results(**vars(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary["artifact_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


def aggregate_index(rows):
    return aggregate_fixed_denominator(
        rows,
        supports_reaction_metrics=any(
            method_supports_reaction_metrics(r["method"]) for r in rows
        ),
    )
