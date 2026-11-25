from __future__ import annotations
from .inputs import scientific_config

import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from DataAdapters import validate_partition_preflight
from Metrics.reaction_annotation_metrics import (
    ReactionOptions,
    _counts_to_prf,
    compute_reaction_metrics,
    extract_true_reactions_from_annotation,
)
from Utils.paths import repository_relative, sanitize_paths

from .exceptions import is_nonfatal_timeout_exception
from . import artifact_validation
from .method_capabilities import REACTION_UNSUPPORTED, method_supports_reaction_metrics


def _populate_row_from_saved(
    row: dict[str, Any], saved: dict[str, Any] | None
) -> dict[str, Any]:
    """Attach summary.json fields used by batch-level summaries."""
    if saved is None:
        row["status"] = "invalid"
        row["failure_reason"] = "missing_summary"
        return row

    row["status"] = saved.get("status", "failed")
    row["saved_status"] = saved.get("status")
    row["failure_reason"] = saved.get("failure_reason")
    row["error"] = saved.get("error")
    row["best_equation"] = saved.get("generated_equation", {}).get("best")
    row["best_equation_substituted"] = saved.get("generated_equation", {}).get(
        "best_substituted"
    )
    row["original_equation_substituted"] = saved.get("original_equation", {}).get(
        "substituted"
    )
    fit_metrics = saved.get("metrics") or {}
    fit_overall = fit_metrics.get("overall") or {}
    row["fit_r2_mean"] = fit_overall.get("r2_mean")
    row["fit_rmse_mean"] = fit_overall.get("rmse_mean")
    row["fit_nrmse_mean"] = fit_overall.get("nrmse_mean")

    evaluations = [
        item for item in (saved.get("evaluations") or []) if isinstance(item, dict)
    ]
    expected_evaluations = [
        str(name)
        for name in ((saved.get("fit_config") or {}).get("eval_trajectory_names") or [])
    ]
    official = next(
        (
            item
            for item in evaluations
            if item.get("eval_trajectory_name")
            == (
                "perturbed"
                if "perturbed" in expected_evaluations
                else (expected_evaluations[0] if expected_evaluations else None)
            )
        ),
        None,
    )
    metrics = (official or {}).get("metrics") if expected_evaluations else fit_metrics
    metrics = metrics or {}
    overall = metrics.get("overall") or {}
    row["trajectory_metric_protocol"] = (official or {}).get(
        "eval_trajectory_name"
    ) or (expected_evaluations[0] if expected_evaluations else "fit_reconstruction")
    row["trajectory_evaluation_status"] = (
        (official or {}).get("status") if expected_evaluations else saved.get("status")
    )
    trajectory_record = (
        (official or {}).get("trajectory_evaluation")
        if expected_evaluations
        else saved.get("trajectory_evaluation")
    ) or {}
    row["trajectory_metrics_included"] = bool(
        trajectory_record.get("included_in_aggregation", False)
    )
    row["r2_mean"] = overall.get("r2_mean")
    row["rmse_mean"] = overall.get("rmse_mean")
    row["nrmse_mean"] = overall.get("nrmse_mean")
    reaction = saved.get("reaction_evaluation") or {}
    reaction_scores = reaction.get("scores") or {}
    row["reaction_metrics_enabled"] = bool(
        reaction.get("enabled", row.get("reaction_metrics_enabled", False))
    )
    row["reaction_metrics_included"] = bool(
        reaction.get(
            "included_in_aggregation", row.get("reaction_metrics_included", False)
        )
    )
    row["reaction_status"] = reaction.get("status")
    for source_key, target_key in (
        ("reaction_TP", "reaction_tp"),
        ("reaction_FP", "reaction_fp"),
        ("reaction_FN", "reaction_fn"),
        ("reaction_f1", "reaction_f1"),
        ("reaction_precision", "reaction_precision"),
        ("reaction_recall", "reaction_recall"),
    ):
        row[target_key] = reaction_scores.get(source_key)
    if not method_supports_reaction_metrics(
        str(row.get("method") or saved.get("method") or "")
    ):
        row.update(
            reaction_metrics_enabled=False,
            reaction_metrics_included=False,
            reaction_exclusion_reason=REACTION_UNSUPPORTED,
        )
        for key in (
            "reaction_f1",
            "reaction_precision",
            "reaction_recall",
            "reaction_tp",
            "reaction_fp",
            "reaction_fn",
        ):
            row[key] = None
    return row


def _write_worker_failure(
    output_json: Path,
    *,
    status: str,
    reason: str,
    exc: BaseException,
    elapsed_seconds: float,
    traceback_text: str | None,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_paths(
        {
            "schema_version": 1,
            "stage": "worker",
            "status": status,
            "failure_reason": reason,
            "exception_type": type(exc).__name__,
            "message": repr(exc),
            "traceback": traceback_text,
            "elapsed_seconds": elapsed_seconds,
            "retry_count": 0,
        }
    )
    (output_json.parent / "failure.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_single_system_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one system in a standalone worker process.

    Redirects stderr to ``run.log`` inside the output directory so that
    verbose GA logs are captured even when running under ``ProcessPoolExecutor``.
    """
    row = dict(task["row"])
    output_json = Path(task["output_json"])
    output_dir = output_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    t0 = time.time()

    # --- Redirect stderr to run.log for the duration of this task ---
    # Use Python file objects (not fd-level dup2) so that Python's IO
    # buffering layer sees the redirect.  buffering=1 = line-buffered,
    # so each log line is flushed to disk immediately.
    old_stderr = sys.stderr
    _redirected = False
    try:
        log_fh = open(log_path, "a", buffering=1, encoding="utf-8")
        # Write a header so the log file is self-contained
        log_fh.write(
            f"{'='*80}\n"
            f"config: {repository_relative(task.get('config_path'))}\n"
            f"system: {Path(task['system_dir']).name}\n"
            f"started_at: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"{'='*80}\n"
        )
        sys.stderr = log_fh
        _redirected = True

        # Clear verbose logging handlers inherited from the parent process
        # (fork semantics).  _configure_verbose_logging() inside wrapper.fit()
        # skips creating new handlers if one already exists; the inherited
        # handler still references the parent's stderr.  Clearing forces fresh
        # StreamHandler creation that captures the redirected sys.stderr.
        import logging

        for _logger_name in (
            "Wrappers.kinetic_aware_sr_wrapper",
            "Methods.KineticAwareSR",
            "Wrappers.kineselectsr_wrapper",
            "Methods.KineSelectSR",
        ):
            _lg = logging.getLogger(_logger_name)
            _lg.handlers.clear()
            _lg.propagate = True
    except (OSError, AttributeError):
        # If redirection fails, keep writing to the original stderr
        pass

    try:
        runner = task["single_runner_cls"](
            task["system_dir"], trajectory_name=task["trajectory_name"]
        )
        saved_json = runner.run(
            output_json=output_json,
            config_path=task["config_path"],
            results_root=task["results_root"],
            run_mode="batch",
            config_stem=task["config_stem"],
            **task["runner_kwargs"],
        )
        row["runtime_sec"] = time.time() - t0

        try:
            saved = json.loads(Path(saved_json).read_text(encoding="utf-8"))
        except Exception:
            saved = None
        return _populate_row_from_saved(row, saved)
    except Exception as exc:
        row["runtime_sec"] = time.time() - t0
        try:
            saved = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception:
            saved = None
        if isinstance(saved, dict) and saved.get("status") in {
            "failed",
            "timeout",
            "invalid",
        }:
            return _populate_row_from_saved(row, saved)
        row["status"] = "failed"
        row["failure_reason"] = "exception"
        row["error"] = sanitize_paths(repr(exc))
        _write_worker_failure(
            output_json,
            status="failed",
            reason="exception",
            exc=exc,
            elapsed_seconds=row["runtime_sec"],
            traceback_text=traceback.format_exc(),
        )
        return row
    except BaseException as exc:
        if not is_nonfatal_timeout_exception(exc):
            raise
        row["runtime_sec"] = time.time() - t0
        try:
            saved = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception:
            saved = None
        if isinstance(saved, dict) and saved.get("status") in {
            "failed",
            "timeout",
            "invalid",
        }:
            return _populate_row_from_saved(row, saved)
        row["status"] = "timeout"
        row["failure_reason"] = "timeout"
        row["error"] = sanitize_paths(repr(exc))
        _write_worker_failure(
            output_json,
            status="timeout",
            reason="timeout",
            exc=exc,
            elapsed_seconds=row["runtime_sec"],
            traceback_text=traceback.format_exc(),
        )
        return row
    finally:
        if _redirected:
            sys.stderr = old_stderr
            log_fh.write(
                f"{'='*80}\n"
                f"finished_at: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                f"runtime_sec: {time.time() - t0:.1f}\n"
                f"{'='*80}\n"
            )
            log_fh.close()


class BaseBatchRunner:
    """Generic batch runner for SysBioSR single-system runners.

    It scans a root directory for model folders containing:
        traj/trajectory_metadata.json

    and repeatedly calls a provided single-system runner class.
    """

    def __init__(
        self,
        *,
        systems_root: str | Path,
        output_root: str | Path,
        single_runner_cls,
        method_name: str,
        trajectory_name: str = "default",
        recursive: bool = True,
        results_root: str | Path | None = None,
        config_stem: str | None = None,
        benchmark_definition: str | Path | None = None,
        benchmark_partition: str | None = None,
        verify_data_integrity: bool = False,
        run_seed: int = 0,
    ) -> None:
        self.systems_root = Path(systems_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.results_root = (
            None if results_root is None else Path(results_root).resolve()
        )
        self.config_stem = config_stem
        self.benchmark_definition = (
            None
            if benchmark_definition is None
            else Path(benchmark_definition).resolve()
        )
        self.benchmark_partition = benchmark_partition
        from DataAdapters import reject_runtime_integrity_request

        reject_runtime_integrity_request(verify_data_integrity)
        self.verify_data_integrity = False
        self.run_seed = int(run_seed)
        self._manifest_rows: dict[str, Any] = {}
        self._release_context: dict[str, str] = {}
        self.single_runner_cls = single_runner_cls
        self.method_name = method_name
        self.supports_reaction_metrics = method_supports_reaction_metrics(method_name)
        self.trajectory_name = trajectory_name
        self.recursive = recursive

    def discover_system_dirs(self) -> list[Path]:
        if self.benchmark_definition is not None:
            if not self.benchmark_partition:
                raise ValueError(
                    "benchmark_partition is required with benchmark_definition"
                )
            with ProcessPoolExecutor(max_workers=1) as executor:
                paths, rows, context = executor.submit(
                    validate_partition_preflight,
                    self.benchmark_definition,
                    self.systems_root,
                    partition=self.benchmark_partition,
                ).result()
            self._manifest_rows = {row.system_id: row for row in rows}
            self._release_context = dict(context)
            self._release_context["benchmark_integrity_preflight"] = "isolated_process"
            return paths
        pattern = (
            "**/traj/trajectory_metadata.json"
            if self.recursive
            else "*/traj/trajectory_metadata.json"
        )
        metadata_files = sorted(self.systems_root.glob(pattern))
        system_dirs = [meta.parent.parent.resolve() for meta in metadata_files]
        # de-duplicate while preserving order
        seen = set()
        unique_dirs = []
        for d in system_dirs:
            if d not in seen:
                seen.add(d)
                unique_dirs.append(d)
        return unique_dirs

    def build_output_json_path(self, system_dir: str | Path) -> Path:
        system_id = Path(system_dir).name
        return (
            self.output_root
            / system_id
            / self.trajectory_name
            / f"seed-{self.run_seed}"
            / "summary.json"
        )

    @staticmethod
    def _safe_load_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _safe_mean(rows: list[dict[str, Any]], key: str) -> float | None:
        values: list[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except Exception:
                continue
            if math.isnan(value):
                continue
            values.append(value)
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _row_is_metrics_aggregatable(row: dict[str, Any]) -> bool:
        status = row.get("status")
        saved_status = row.get("saved_status")

        if status == "success":
            pass
        elif status == "skipped" and saved_status == "success":
            pass
        else:
            return False

        if row.get("trajectory_evaluation_status") != "success":
            return False
        if not row.get("trajectory_metrics_included"):
            return False
        return any(
            row.get(key) is not None for key in ("rmse_mean", "nrmse_mean", "r2_mean")
        )

    @staticmethod
    def _validate_declared_artifacts(
        saved: dict[str, Any],
        *,
        output_json: Path,
        require_failure: bool = False,
    ) -> str | None:
        """Compatibility delegate; new callers use artifact_validation directly."""
        return artifact_validation.validate_declared_artifacts(
            saved, output_json=output_json, require_failure=require_failure
        )

    @staticmethod
    def _validate_resumable_result(
        saved: dict[str, Any] | None,
        *,
        output_json: Path,
        expected: dict[str, Any],
    ) -> str | None:
        """Compatibility delegate; new callers use artifact_validation directly."""
        return artifact_validation.validate_resumable_result(
            saved, output_json=output_json, expected=expected
        )

    @staticmethod
    def _score_missing_reaction_prediction(
        row: dict[str, Any], system_dir: Path
    ) -> None:
        """Keep failed main/challenge runs in the frozen reaction denominator."""
        if (
            not row.get("reaction_metrics_enabled")
            or row.get("reaction_f1") is not None
        ):
            return
        annotation_path = system_dir / "reaction_annotation.json"
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            true_reactions = extract_true_reactions_from_annotation(annotation)
            scores = compute_reaction_metrics(
                true_reactions,
                [],
                ReactionOptions(
                    strict_kinetic_law=True,
                    strict_hill_n=False,
                    strict_K=False,
                    term_tolerance=1e-12,
                    projection_mode="term",
                    stoich_projection_mode="unit",
                    kineselectsr_source_policy="auto",
                ),
                dim=len(annotation.get("state_names") or []),
            )
        except Exception as exc:
            row["reaction_metrics_included"] = False
            row["reaction_status"] = "invalid_ground_truth"
            row["reaction_error"] = type(exc).__name__
            return
        row.update(
            {
                "reaction_metrics_included": True,
                "reaction_status": "empty_prediction_due_run_failure",
                "reaction_tp": scores["reaction_TP"],
                "reaction_fp": scores["reaction_FP"],
                "reaction_fn": scores["reaction_FN"],
                "reaction_f1": scores["reaction_f1"],
                "reaction_precision": scores["reaction_precision"],
                "reaction_recall": scores["reaction_recall"],
            }
        )

    def run(
        self,
        *,
        limit: int | None = None,
        skip_existing: bool = True,
        continue_on_error: bool = True,
        num_workers: int = 1,
        config_path: str | Path | None = None,
        **runner_kwargs: Any,
    ) -> dict[str, Any]:
        system_dirs = self.discover_system_dirs()
        if limit is not None:
            system_dirs = system_dirs[:limit]

        self.output_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []
        t_batch_start = time.time()
        num_workers = max(1, int(num_workers))
        progress = tqdm(
            system_dirs,
            desc=f"{self.method_name} batch",
            unit="system",
            dynamic_ncols=True,
        )

        for idx, system_dir in enumerate(progress, start=1):
            output_json = self.build_output_json_path(system_dir)

            row: dict[str, Any] = {
                "index": idx,
                "system_id": Path(system_dir).name,
                "method": self.method_name,
                "trajectory_name": self.trajectory_name,
                "output_json": str(output_json.relative_to(self.output_root)),
                "status": None,
                "runtime_sec": None,
                "error": None,
            }

            if not self.supports_reaction_metrics:
                row.update(
                    reaction_metrics_enabled=False,
                    reaction_exclusion_reason=REACTION_UNSUPPORTED,
                )
            task_runner_kwargs = dict(runner_kwargs)
            manifest_row = self._manifest_rows.get(row["system_id"])
            if manifest_row is not None:
                row["dataset_version"] = manifest_row.dataset_version
                row["benchmark_partition"] = manifest_row.benchmark_partition
                row["reaction_metrics_enabled"] = (
                    manifest_row.reaction_metrics and self.supports_reaction_metrics
                )
                configured_version = task_runner_kwargs.get("dataset_version")
                if configured_version not in {None, "", manifest_row.dataset_version}:
                    raise ValueError(
                        f"dataset_version mismatch for {manifest_row.system_id}: "
                        f"config={configured_version!r}, manifest={manifest_row.dataset_version!r}"
                    )
                task_runner_kwargs.update(
                    {
                        "dataset_version": manifest_row.dataset_version,
                        "benchmark_partition": manifest_row.benchmark_partition,
                        "reaction_metrics": manifest_row.reaction_metrics
                        and self.supports_reaction_metrics,
                        "benchmark_spec": self._release_context["benchmark_spec"],
                        "benchmark_integrity_preflight": self._release_context[
                            "benchmark_integrity_preflight"
                        ],
                    }
                )

            if output_json.exists():
                if not skip_existing:
                    raise FileExistsError(
                        f"Refusing to overwrite existing run identity: {output_json}. "
                        "Choose a new --results-root, or enable resume for an identical input."
                    )
                existing = self._safe_load_json(output_json)
                expected_identity = {
                    "method": self.method_name,
                    "run_config": scientific_config(task_runner_kwargs),
                    "system_dir": Path(system_dir),
                    "benchmark_spec": task_runner_kwargs.get("benchmark_spec"),
                    "system_id": row["system_id"],
                    "trajectory_name": self.trajectory_name,
                    "dataset_version": task_runner_kwargs.get("dataset_version"),
                    "benchmark_partition": task_runner_kwargs.get(
                        "benchmark_partition"
                    ),
                    "experiment_id": task_runner_kwargs.get("experiment_id"),
                    "random_seed": task_runner_kwargs.get(
                        "random_seed", task_runner_kwargs.get("seed", 0)
                    ),
                }
                rejection = artifact_validation.validate_resumable_result(
                    existing,
                    output_json=output_json,
                    expected=expected_identity,
                )
                if rejection is not None:
                    raise RuntimeError(
                        f"Existing result cannot be resumed ({rejection}): {output_json}. Choose a new --results-root."
                    )
                row = _populate_row_from_saved(row, existing)
                row["status"] = "skipped"
                rows.append(row)
                tqdm.write(f"[resume] {row['system_id']} -> {output_json}")
                continue

            if output_json.parent.exists() and any(output_json.parent.iterdir()):
                raise FileExistsError(
                    f"Incomplete existing result directory: {output_json.parent}; use a fresh results root"
                )
            output_json.parent.mkdir(parents=True, exist_ok=True)

            tasks.append(
                {
                    "row": row,
                    "system_dir": system_dir,
                    "output_json": output_json,
                    "single_runner_cls": self.single_runner_cls,
                    "method_name": self.method_name,
                    "trajectory_name": self.trajectory_name,
                    "results_root": self.results_root,
                    "config_stem": self.config_stem,
                    "config_path": config_path,
                    "runner_kwargs": task_runner_kwargs,
                }
            )

        progress.close()

        def _report_row(row: dict[str, Any]) -> None:
            tag = "[ok]  " if row.get("status") == "success" else "[fail]"
            if row.get("status") == "success":
                tqdm.write(f"{tag} {row['system_id']} -> {row['output_json']}")
            else:
                tqdm.write(
                    f"{tag} {row['system_id']} -> {row['output_json']}"
                    f" ({row.get('failure_reason') or 'failed'})"
                )

        if num_workers == 1 or len(tasks) <= 1:
            progress = tqdm(
                tasks, desc=f"{self.method_name} run", unit="system", dynamic_ncols=True
            )
            for task in progress:
                row = _run_single_system_task(task)
                rows.append(row)
                _report_row(row)
                if row.get("status") == "failed" and not continue_on_error:
                    progress.close()
                    raise RuntimeError(
                        f"Batch run failed for {row['system_id']}: {row.get('error')}"
                    )
            progress.close()
        else:
            progress = tqdm(
                total=len(tasks),
                desc=f"{self.method_name} run x{num_workers}",
                unit="system",
                dynamic_ncols=True,
            )
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_run_single_system_task, task): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    try:
                        row = future.result()
                    except Exception as exc:
                        failed_task = futures[future]
                        row = dict(failed_task["row"])
                        row["status"] = "failed"
                        row["failure_reason"] = "worker_exception"
                        row["error"] = sanitize_paths(repr(exc))
                        _write_worker_failure(
                            Path(failed_task["output_json"]),
                            status="failed",
                            reason="worker_exception",
                            exc=exc,
                            elapsed_seconds=time.time() - t_batch_start,
                            traceback_text=traceback.format_exc(),
                        )
                    except BaseException as exc:
                        if not is_nonfatal_timeout_exception(exc):
                            raise
                        failed_task = futures[future]
                        row = dict(failed_task["row"])
                        row["status"] = "timeout"
                        row["failure_reason"] = "timeout"
                        row["error"] = sanitize_paths(repr(exc))
                        _write_worker_failure(
                            Path(failed_task["output_json"]),
                            status="timeout",
                            reason="timeout",
                            exc=exc,
                            elapsed_seconds=time.time() - t_batch_start,
                            traceback_text=traceback.format_exc(),
                        )
                    rows.append(row)
                    progress.update(1)
                    _report_row(row)
                    if row.get("status") == "failed" and not continue_on_error:
                        progress.close()
                        raise RuntimeError(
                            f"Batch run failed for {row['system_id']}: {row.get('error')}"
                        )
            progress.close()

        rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
        system_dirs_by_id = {path.name: path for path in system_dirs}
        for row in rows:
            system_dir = system_dirs_by_id.get(str(row.get("system_id")))
            if system_dir is not None:
                self._score_missing_reaction_prediction(row, system_dir)

        metrics_rows = [row for row in rows if self._row_is_metrics_aggregatable(row)]
        reaction_denominator_rows = [
            row for row in rows if row.get("reaction_metrics_enabled")
        ]
        reaction_rows = [
            row
            for row in reaction_denominator_rows
            if row.get("reaction_metrics_included")
        ]
        reaction_tp = sum(int(row.get("reaction_tp") or 0) for row in reaction_rows)
        reaction_fp = sum(int(row.get("reaction_fp") or 0) for row in reaction_rows)
        reaction_fn = sum(int(row.get("reaction_fn") or 0) for row in reaction_rows)
        if reaction_rows:
            reaction_precision, reaction_recall, reaction_micro_f1, _ = (
                _counts_to_prf(reaction_tp, reaction_fp, reaction_fn)
            )
        else:
            reaction_precision = reaction_recall = reaction_micro_f1 = None

        summary = {
            "schema_version": 1,
            "method": self.method_name,
            "config_path": repository_relative(config_path),
            "systems_root": repository_relative(self.systems_root),
            "output_root": repository_relative(self.output_root),
            "trajectory_name": self.trajectory_name,
            "recursive": self.recursive,
            "benchmark_definition": (
                None
                if self.benchmark_definition is None
                else self.benchmark_definition.name
            ),
            "benchmark_partition": self.benchmark_partition,
            "manifest_content_verified": self.verify_data_integrity,
            "num_workers": num_workers,
            "num_systems": len(system_dirs),
            "num_success": sum(1 for r in rows if r.get("status") == "success"),
            "num_failed": sum(1 for r in rows if r.get("status") == "failed"),
            "num_timeout": sum(1 for r in rows if r.get("status") == "timeout"),
            "num_invalid": sum(1 for r in rows if r.get("status") == "invalid"),
            "num_skipped": sum(1 for r in rows if r.get("status") == "skipped"),
            "num_metrics_aggregated": len(metrics_rows),
            "runtime_sec": time.time() - t_batch_start,
            "aggregate_metrics": {
                "protocol": "perturbed_rollout_when_configured_otherwise_fit_reconstruction",
                "fixed_denominator": len(rows),
                "success_rate": (len(metrics_rows) / len(rows)) if rows else None,
                "rmse_mean": self._safe_mean(metrics_rows, "rmse_mean"),
                "nrmse_mean": self._safe_mean(metrics_rows, "nrmse_mean"),
                "r2_mean": self._safe_mean(metrics_rows, "r2_mean"),
            },
            "fit_reconstruction_metrics": {
                "rmse_mean": self._safe_mean(rows, "fit_rmse_mean"),
                "nrmse_mean": self._safe_mean(rows, "fit_nrmse_mean"),
                "r2_mean": self._safe_mean(rows, "fit_r2_mean"),
            },
            "reaction_aggregate_metrics": {
                "fixed_denominator": len(reaction_denominator_rows),
                "num_included": len(reaction_rows),
                "run_success_rate": (
                    sum(
                        1
                        for row in reaction_denominator_rows
                        if row.get("saved_status") == "success"
                    )
                    / len(reaction_denominator_rows)
                    if reaction_denominator_rows
                    else None
                ),
                "macro_f1": self._safe_mean(reaction_rows, "reaction_f1"),
                "micro_tp": reaction_tp,
                "micro_fp": reaction_fp,
                "micro_fn": reaction_fn,
                "micro_precision": reaction_precision,
                "micro_recall": reaction_recall,
                "micro_f1": reaction_micro_f1,
            },
            "rows": rows,
        }

        if not self.supports_reaction_metrics:
            summary["reaction_aggregate_metrics"].update(
                supported=False,
                exclusion_reason=REACTION_UNSUPPORTED,
                micro_precision=None,
                micro_recall=None,
                micro_f1=None,
            )
        summary_json = self.output_root / "summary.json"
        summary_csv = self.output_root / "summary.csv"
        sanitized_summary = sanitize_paths(summary)
        sanitized_rows = sanitize_paths(rows)
        summary_json.write_text(
            json.dumps(sanitized_summary, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(sanitized_rows).to_csv(summary_csv, index=False)
        progress.close()

        print("=" * 80)
        print(f"Batch run finished: method={self.method_name}")
        print(f"config_path   : {summary['config_path']}")
        print(f"systems_root  : {self.systems_root}")
        print(f"output_root   : {self.output_root}")
        print(f"success       : {summary['num_success']}")
        print(f"failed        : {summary['num_failed']}")
        print(f"skipped       : {summary['num_skipped']}")
        print(f"summary json  : {summary_json}")
        print(f"summary csv   : {summary_csv}")
        print("=" * 80)

        return summary
