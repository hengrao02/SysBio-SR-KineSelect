"""Aggregate FPMS and Alpha evaluations without rerunning search."""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from .saved_front import targets, validate_evaluation_sidecar, REQUESTED_SELECTORS
from Runners.core.artifact_validation import _safe_load_json
from Runners.core.inputs import atomic_json
from Runners.core.result_aggregation import (
    PARTITIONS,
    aggregate_fixed_denominator,
    empty_prediction_scores,
)


def aggregate_selector_results(
    config_paths, *, output_dir, dataset_root=None, results_root=None, system_ids=None
):
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]
    rows = []
    seen = set()
    for config_path in config_paths:
        for config, definition, member, system_dir, source in targets(
            config_path,
            dataset_root=dataset_root,
            results_root=results_root,
            system_ids=system_ids,
        ):
            if member.system_id in seen:
                raise ValueError("Duplicate system across selector aggregation configs")
            seen.add(member.system_id)
            for mode in REQUESTED_SELECTORS:
                path = (
                    source.parent
                    / "selection"
                    / mode
                    / "metrics/evaluation_summary.json"
                )
                saved = _safe_load_json(path)
                reason = (
                    "missing_evaluation"
                    if not saved
                    else validate_evaluation_sidecar(
                        saved,
                        path=path,
                        source_dir=source.parent,
                        config=config,
                        system_dir=system_dir,
                        benchmark_spec=definition.spec,
                    )
                )
                row = {
                    "system_id": member.system_id,
                    "benchmark_partition": member.benchmark_partition,
                    "requested_selector": mode,
                    "status": (
                        "invalid" if not saved else saved.get("status", "invalid")
                    ),
                    "failure_reason": reason,
                    "artifact_integrity": reason is None,
                    "trajectory_included": False,
                    "reaction_metrics_enabled": member.reaction_metrics,
                    "reaction_included": member.reaction_metrics,
                    "rmse_mean": None,
                    "nrmse_mean": None,
                    "r2_mean": None,
                }
                scores = {}
                if reason is None:
                    record = saved["trajectories"]["perturbed"]
                    row["trajectory_included"] = record["status"] == "success"
                    if row["trajectory_included"]:
                        for key in ("rmse_mean", "nrmse_mean", "r2_mean"):
                            row[key] = record["metrics"]["overall"][key]
                    reaction = (
                        saved["trajectories"]["default"]["reaction_evaluation"] or {}
                    )
                    if reaction.get("enabled") and reaction.get(
                        "included_in_aggregation"
                    ):
                        scores = reaction.get("scores") or {}
                if member.reaction_metrics:
                    if not scores:
                        scores = empty_prediction_scores(
                            system_dir / "reaction_annotation.json"
                        )
                    row.update(
                        reaction_tp=scores.get("reaction_TP"),
                        reaction_fp=scores.get("reaction_FP"),
                        reaction_fn=scores.get("reaction_FN"),
                        reaction_f1=scores.get("reaction_f1"),
                    )
                rows.append(row)
    rows.sort(
        key=lambda r: (
            PARTITIONS.index(r["benchmark_partition"]),
            r["system_id"],
            r["requested_selector"],
        )
    )
    summary = {
        "schema_version": 2,
        "formal_complete": len(seen) == 198 and not system_ids,
        "selectors": {
            mode: aggregate_fixed_denominator(
                [r for r in rows if r["requested_selector"] == mode]
            )
            for mode in REQUESTED_SELECTORS
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_index.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    )
    atomic_json(output_dir / "summary.json", summary)
    return rows, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path, nargs="+")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--results-root", type=Path)
    p.add_argument("--system-id", action="append", nargs="+", default=[])
    args = p.parse_args(argv)
    try:
        _, summary = aggregate_selector_results(
            args.config,
            output_dir=args.output_dir,
            dataset_root=args.dataset_root,
            results_root=args.results_root,
            system_ids=[s for group in args.system_id for s in group],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        p.error(str(exc))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return (
        0
        if all(v["artifact_integrity_passed"] for v in summary["selectors"].values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
