"""Evaluate FPMS and Alpha from saved numerical fronts, without rerunning search."""

from __future__ import annotations
import argparse
from dataclasses import fields
import json
from pathlib import Path
import shutil
from DataAdapters import BenchmarkDefinition
from Methods.KineSelectSR.formulation import FormulationConfig
from Methods.KineSelectSR.search import KineSelectSRSearchConfig, canonical_candidate_id
from Methods.KineSelectSR.search.serialization import (
    candidate_labels,
    public_references,
    serialize_individual,
)
from Methods.KineSelectSR.sindy.selector_modes import normalize_selector_mode
from Wrappers.kineselectsr_wrapper import KineSelectSRWrapper
from Wrappers.base_ode_sr_wrapper import EquationResult
from Runners.core.base_single_system_runner import BaseSingleSystemRunner
from Runners.core.common import build_result_path
from Runners.core.inputs import atomic_json, scientific_config
from Runners.core.artifact_validation import (
    _safe_load_json,
    validate_declared_artifacts,
)
from .artifacts import (
    _candidate_record,
    _select_from_front,
    build_search_config,
    build_selector_config,
)
from .front_reader import SavedFrontError, read_saved_front, strict_json_file
from .run import load_run_config

REQUESTED_SELECTORS = ("fpms", "alpha")


def _canonicalize_search_selector(config: dict) -> dict:
    """Normalize the one historical selector label before strict comparison."""
    normalized = dict(config)
    search = normalized.get("search")
    if isinstance(search, dict) and "sindy_selector_mode" in search:
        search = dict(search)
        search["sindy_selector_mode"] = normalize_selector_mode(
            search["sindy_selector_mode"], field="sindy_selector_mode"
        )
        normalized["search"] = search
    return normalized


def _resolved_method_configs(config):
    form = {f.name for f in fields(FormulationConfig)}
    search = {f.name for f in fields(KineSelectSRSearchConfig)}
    return FormulationConfig(
        **{k: v for k, v in config.items() if k in form}
    ), KineSelectSRSearchConfig(**{k: v for k, v in config.items() if k in search})


def _apply_path_overrides(config_path, config, *, dataset_root=None, results_root=None):
    from run import _apply_cli_overrides

    return _apply_cli_overrides(
        config, dataset_root=dataset_root, results_root=results_root, resume=False
    )


def targets(config_path, *, dataset_root=None, results_root=None, system_ids=None):
    path, config = load_run_config(config_path)
    config = _apply_path_overrides(
        path, config, dataset_root=dataset_root, results_root=results_root
    )
    definition = BenchmarkDefinition(config["benchmark_definition"])
    part = config["benchmark_partition"]
    root = (
        Path(config["systems_root"])
        if config["run_type"] == "batch"
        else Path(config["system_dir"]).parent
    )
    definition.validate_partition(root, partition=part)
    allowed = None if not system_ids else set(system_ids)
    members = definition.partition_rows(part)
    if config["run_type"] == "single":
        members = [r for r in members if r.system_id == Path(config["system_dir"]).name]
    if allowed is not None and not allowed.issubset({r.system_id for r in members}):
        raise ValueError("system allowlist contains unknown systems for this config")
    for member in members:
        if allowed is not None and member.system_id not in allowed:
            continue
        result = build_result_path(
            config["results_root"],
            method_name="kineselectsr",
            run_mode="batch",
            config_stem=part,
            experiment_id=config["experiment_id"],
            dataset_version=definition.dataset_version,
            benchmark_partition=part,
            system_id=member.system_id,
            trajectory_name="default",
            seed=config["run_seed"],
        )
        yield config, definition, member, root / member.system_id, result


def validate_evaluation_sidecar(
    summary, *, path, source_dir, config, system_dir, benchmark_spec
):
    try:
        if summary.get("schema_version") != 2 or summary.get("status") != "success":
            return "evaluation_not_complete"
        if summary.get("run_config") != scientific_config(config):
            return "evaluation_configuration_changed"
        if summary.get("system_id") != Path(system_dir).name:
            return "evaluation_system_mismatch"
        if (path.parent / "source_front.jsonl.zst").read_bytes() != (
            source_dir / "search/pareto_front.jsonl.zst"
        ).read_bytes():
            return "source_front_changed"
        mode = summary["requested_selector"]
        selected = json.loads(
            (source_dir / "selection" / mode / "selected_model.json").read_text()
        )
        if summary["selected_model"] != selected:
            return "selection_changed"
        for role in ("default", "perturbed"):
            record = summary["trajectories"][role]
            target = (path.parent / record["path"]).resolve()
            target.relative_to(path.parent.resolve())
            if target.is_symlink():
                return "evaluation_path_invalid"
            saved = _safe_load_json(target)
            if not saved or saved.get("status") != "success":
                return "evaluation_not_complete"
            if saved.get("system", {}).get("trajectory_name") != role:
                return "evaluation_role_mismatch"
            reason = validate_declared_artifacts(saved, output_json=target)
            if reason:
                return reason
            from Runners.core.inputs import validate_current_inputs

            reason = validate_current_inputs(
                target.parent / "inputs.npz",
                system_dir=system_dir,
                trajectory_name=role,
                benchmark_spec=benchmark_spec,
            )
            if reason:
                return reason
            if (
                record["metrics"] != saved["metrics"]
                or record["reaction_evaluation"] != saved["reaction_evaluation"]
            ):
                return "evaluation_scores_mismatch"
    except (OSError, KeyError, TypeError, ValueError):
        return "evaluation_invalid"
    return None


def evaluate_one_saved_front(
    *, config, definition, member, system_dir, summary_path, resume=False
):
    source = strict_json_file(summary_path)
    if source.get("schema_version") != 2:
        raise SavedFrontError(
            "unsupported_result_format",
            "summary.json must have schema_version=2.",
        )
    reason = validate_declared_artifacts(
        source,
        output_json=summary_path,
        require_failure=source.get("status") != "success",
    )
    if reason:
        raise SavedFrontError(reason, str(summary_path))
    from Runners.core.inputs import validate_current_inputs

    reason = validate_current_inputs(
        summary_path.parent / "inputs.npz",
        system_dir=system_dir,
        trajectory_name="default",
        benchmark_spec=definition.spec,
    )
    if reason:
        raise SavedFrontError(reason, str(summary_path))
    formulation, ga = _resolved_method_configs(config)
    source_config = strict_json_file(
        summary_path.parent / "search/resolved_search_config.json"
    )
    if _canonicalize_search_selector(source_config) != _canonicalize_search_selector(
        build_search_config(formulation, ga)
    ):
        raise SavedFrontError(
            "search_configuration_changed", "refit with this search configuration"
        )
    search = strict_json_file(summary_path.parent / "search/search_summary.json")
    front = read_saved_front(
        front_path=summary_path.parent / "search/pareto_front.jsonl.zst",
        search_summary=search,
        dimension=len(source["system"]["state_names"]),
    )
    labels = candidate_labels(front.individuals, front.selection_objective)
    results = []
    for mode in REQUESTED_SELECTORS:
        directory = summary_path.parent / "selection" / mode / "metrics"
        target = directory / "evaluation_summary.json"
        if directory.exists():
            if not resume:
                raise FileExistsError(
                    f"Existing selector evaluation: {directory}; use resume or a new results root"
                )
            saved = _safe_load_json(target)
            reason = (
                "evaluation_missing"
                if saved is None
                else validate_evaluation_sidecar(
                    saved,
                    path=target,
                    source_dir=summary_path.parent,
                    config=config,
                    system_dir=system_dir,
                    benchmark_spec=definition.spec,
                )
            )
            if reason:
                raise SavedFrontError(reason, str(target))
            results.append(saved)
            continue
        winner, actual, fallback, trace = _select_from_front(
            mode, list(front.individuals), front.selection_objective, ga
        )
        selected = (
            None
            if winner is None
            else _candidate_record(winner, labels[canonical_candidate_id(winner)])
        )
        existing = strict_json_file(
            summary_path.parent / "selection" / mode / "selected_model.json"
        )
        if selected != existing:
            raise SavedFrontError(
                "selector_configuration_changed",
                "online and offline selections differ; use a new experiment",
            )
        directory.mkdir(parents=True)
        shutil.copyfile(
            summary_path.parent / "search/pareto_front.jsonl.zst",
            directory / "source_front.jsonl.zst",
        )
        runner = BaseSingleSystemRunner(system_dir, trajectory_name="default")
        runner._run_settings = source["run_config"]
        wrapper = KineSelectSRWrapper()
        wrapper._formulation_config = formulation
        wrapper._model = None if winner is None else winner.to_model()
        wrapper._is_fitted = winner is not None
        metadata = {} if winner is None else serialize_individual(winner)
        if selected is not None:
            metadata.update(
                best_substituted=selected["equation_substituted"],
                selected_model=selected,
            )
        equation = EquationResult(
            best=None if selected is None else selected["equation"],
            candidates=[],
            status="failed" if selected is None else "success",
            metadata=metadata,
        )
        fit_config = {
            **source["fit_config"],
            "final_selection_mode": mode,
            "eval_trajectory_names": [],
        }
        records = {}
        for role in ("default", "perturbed"):
            trajectory = runner.system.get_trajectory(role)
            output = directory / role / f"seed-{config['run_seed']}" / "summary.json"
            runner._save_result(
                times=trajectory.times,
                gt_trajectory=trajectory.values,
                y0=trajectory.rollout_initial_condition(),
                output_json=output,
                method_name="kineselectsr",
                equation_result=equation,
                fit_config=fit_config,
                extra_payload={
                    "record_kind": "selector_evaluation",
                    "eval_trajectory_name": role,
                },
                plot_title=f"{mode}: {member.system_id} {role}",
                wrapper=wrapper,
                failure_reason="selector_no_winner" if selected is None else None,
            )
            payload = strict_json_file(output)
            payload["complete"] = True
            atomic_json(output, payload)
            records[role] = {
                "path": output.relative_to(directory).as_posix(),
                "status": payload["status"],
                "metrics": payload.get("metrics"),
                "reaction_evaluation": payload.get("reaction_evaluation"),
            }
        summary = {
            "schema_version": 2,
            "system_id": member.system_id,
            "benchmark_partition": member.benchmark_partition,
            "requested_selector": mode,
            "actual_selector": actual,
            "fallback_reason": fallback,
            "selected_model": selected,
            "run_config": scientific_config(config),
            "trajectories": records,
            "status": (
                "success"
                if all(r["status"] == "success" for r in records.values())
                else "failed"
            ),
            "selector_config": build_selector_config(mode, ga),
            "selection": public_references(trace, labels),
        }
        atomic_json(target, summary)
        results.append(summary)
    return results


def evaluate_saved_fronts(
    config_path, *, dataset_root=None, results_root=None, system_ids=None, resume=False
):
    results = []
    for config, definition, member, system_dir, path in targets(
        config_path,
        dataset_root=dataset_root,
        results_root=results_root,
        system_ids=system_ids,
    ):
        results.extend(
            evaluate_one_saved_front(
                config=config,
                definition=definition,
                member=member,
                system_dir=system_dir,
                summary_path=path,
                resume=resume,
            )
        )
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--results-root", type=Path)
    p.add_argument("--system-id", action="append", nargs="+", default=[])
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    try:
        rows = evaluate_saved_fronts(
            args.config,
            dataset_root=args.dataset_root,
            results_root=args.results_root,
            system_ids=[s for group in args.system_id for s in group],
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        p.error(str(exc))
    print(
        json.dumps(
            {
                "selector_evaluations": len(rows),
                "success": sum(r["status"] == "success" for r in rows),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
