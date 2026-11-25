"""Numerical search fronts and post-hoc selectors for KineSelectSR."""

from __future__ import annotations
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import zstandard
from jsonschema import ValidationError
from .schemas import validate_selection_record
from Methods.KineSelectSR.formulation.formatting import (
    format_model,
    format_model_substituted,
)
from Methods.KineSelectSR.search import (
    alpha_select,
    fpms_select,
    fpms_scoring_contract,
    serialize_individual,
    canonical_candidate_id,
)
from Methods.KineSelectSR.search.serialization import (
    ordered_candidates,
    candidate_labels,
    public_references,
)
from Runners.core.inputs import atomic_json
from Utils.paths import repository_relative, sanitize_paths

MODEL_FORMAT = "sysbiosr-model-v2"
METHOD_VERSION = "kineselectsr-2"
FPMS_SELECTOR_VERSION = "fpms-dr-kx-v1"
ALPHA_SELECTOR_VERSION = "alpha-v2"
SEARCH_CONTRACT_VERSION = "kineselectsr-search-v3"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return repository_relative(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        _json_safe(sanitize_paths(payload)),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _select_from_front(
    mode: str,
    front: list[Any],
    objective: str,
    ga_config: Any,
) -> tuple[Any | None, str, str | None, dict[str, Any]]:
    if mode not in {"alpha", "fpms"}:
        raise ValueError(f"Unsupported final selector: {mode!r}")
    if not front:
        return None, mode, "no_candidates", {"candidate_count": 0}
    if mode == "fpms" and objective != "rollout":
        winner = alpha_select(front, ga_config.alpha, ga_config.eps, objective="deriv")
        return (
            winner,
            "alpha",
            "global_rollout_front_unavailable",
            {
                "candidate_count": len(front),
                "alpha": ga_config.alpha,
                "eps": ga_config.eps,
            },
        )
    if mode == "fpms":
        selected = fpms_select(
            front,
            eps=ga_config.eps,
            active_threshold=ga_config.fpms_active_threshold,
        )
        ranked = sorted(selected.scores, key=lambda item: item.rank)
        return (
            selected.winner,
            "fpms",
            selected.fallback_reason,
            {
                **fpms_scoring_contract(),
                "candidate_count": len(front),
                "degenerate_dimensions": list(selected.degenerate_dimensions),
                "ranked_scores": [asdict(item) for item in ranked],
            },
        )
    winner = alpha_select(front, ga_config.alpha, ga_config.eps, objective=objective)
    finite_losses = [
        float(
            ind.metrics.rollout_nmse
            if objective == "rollout"
            else ind.metrics.deriv_scaled_nmse
        )
        for ind in front
        if ind.metrics is not None
        and math.isfinite(
            float(
                ind.metrics.rollout_nmse
                if objective == "rollout"
                else ind.metrics.deriv_scaled_nmse
            )
        )
    ]
    best_loss = min(finite_losses) if finite_losses else None
    threshold = (
        None if best_loss is None else best_loss * ga_config.alpha + ga_config.eps
    )
    return (
        winner,
        "alpha",
        (
            "global_rollout_front_unavailable"
            if objective != ga_config.selection_objective
            else None
        ),
        {
            "candidate_count": len(front),
            "best_loss": best_loss,
            "eligibility_threshold": threshold,
            "alpha": ga_config.alpha,
            "eps": ga_config.eps,
        },
    )


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, _json_safe(sanitize_paths(payload)))


def _candidate_record(individual, candidate_id):
    record = serialize_individual(individual)
    record.update(candidate_id=candidate_id, model_format=MODEL_FORMAT)
    model = individual.to_model()
    record["equation"] = format_model(model)
    record["equation_substituted"] = format_model_substituted(model)
    return record


def _front_jsonl(front, objective):
    ordered = ordered_candidates(front, objective)
    records = [
        _candidate_record(ind, f"candidate-{i:06d}") for i, ind in enumerate(ordered, 1)
    ]
    return b"".join(_canonical_bytes(row) + b"\n" for row in records), records


def build_search_config(formulation_config, ga_config):
    search = asdict(ga_config)
    for key in ("final_selection_mode", "fpms_active_threshold", "fpms_criteria"):
        search.pop(key, None)
    return {"formulation": asdict(formulation_config), "search": search}


def build_selector_config(mode, ga_config):
    if mode == "alpha":
        return {
            "requested_selector": mode,
            "alpha": ga_config.alpha,
            "eps": ga_config.eps,
            "selection_objective": ga_config.selection_objective,
        }
    if mode != "fpms":
        raise ValueError(f"Unknown selector: {mode}")
    return {
        **fpms_scoring_contract(),
        "requested_selector": mode,
        "eps": ga_config.eps,
        "active_threshold": ga_config.fpms_active_threshold,
        "criteria": "DR_KX",
    }


def write_search_artifacts(*, output_dir, summary, wrapper, config_path=None):
    result = getattr(wrapper, "_ga_result", None)
    if result is None:
        return None
    formulation, config = wrapper._formulation_config, wrapper._ga_config
    objective = str(
        (result.timing or {}).get("global_selection_objective")
        or config.selection_objective
    )
    front = list(result.top_candidates or [])
    labels = candidate_labels(front, objective)
    content, records = _front_jsonl(front, objective)
    search_dir = output_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "pareto_front.jsonl.zst").write_bytes(
        zstandard.ZstdCompressor(level=9).compress(content)
    )
    _write_json(
        search_dir / "resolved_search_config.json",
        build_search_config(formulation, config),
    )
    _write_json(
        search_dir / "search_summary.json",
        {
            "schema_version": 2,
            "method": "kineselectsr",
            "method_version": METHOD_VERSION,
            "search_contract_version": SEARCH_CONTRACT_VERSION,
            "system_id": summary["system"]["system_id"],
            "state_names": summary["system"]["state_names"],
            "selection_objective": objective,
            "pareto_front_size": len(records),
            "random_seed": config.random_seed,
            "duration_seconds": (result.timing or {}).get("ga_total_sec"),
        },
    )
    artifacts = {
        "kineselectsr_pareto_front": "search/pareto_front.jsonl.zst",
        "kineselectsr_resolved_search_config": "search/resolved_search_config.json",
        "kineselectsr_search_summary": "search/search_summary.json",
    }
    for mode in ("fpms", "alpha"):
        winner, actual, reason, trace = _select_from_front(
            mode, front, objective, config
        )
        ident = None if winner is None else labels[canonical_candidate_id(winner)]
        selected = None if winner is None else _candidate_record(winner, ident)
        selection_dir = output_dir / "selection" / mode
        selection = {
            "schema_version": 2,
            "requested_selector": mode,
            "actual_selector": actual,
            "status": "success" if winner is not None else "invalid",
            "winner_candidate_id": ident,
            "fallback_reason": reason,
            "selection_objective": objective,
            "tie_break_trace": public_references(trace, labels),
        }
        for name, filename, value in (
            ("selected_model", "selected_model.json", selected),
            (
                "selector_config",
                "resolved_selector_config.json",
                build_selector_config(mode, config),
            ),
            ("selector_summary", "selector_summary.json", selection),
        ):
            if name == "selector_summary":
                validate_selection_record(_json_safe(value))
            _write_json(selection_dir / filename, value)
            artifacts[f"kineselectsr_{mode}_{name}"] = str(
                (selection_dir / filename).relative_to(output_dir)
            )
    return {"artifact_files": artifacts}


def validate_saved_model(saved, output_json):
    from .front_reader import read_saved_front

    if (saved.get("fit_config") or {}).get("method_version") != 2:
        return "unsupported_method_version"
    artifacts = saved.get("artifacts") or {}
    relative = artifacts.get("kineselectsr_pareto_front")
    if not relative:
        return None  # evaluation records contain the frozen model, not a front
    try:
        search = json.loads(
            (output_json.parent / artifacts["kineselectsr_search_summary"]).read_text()
        )
        if search.get("schema_version") != 2:
            return "unsupported_search_format"
        front = read_saved_front(
            front_path=output_json.parent / relative,
            search_summary=search,
            dimension=len(saved["system"]["state_names"]),
        )
        records = {r["candidate_id"]: r for r in front.records}
        for mode in ("fpms", "alpha"):
            model = json.loads(
                (
                    output_json.parent
                    / artifacts[f"kineselectsr_{mode}_selected_model"]
                ).read_text()
            )
            info = json.loads(
                (
                    output_json.parent
                    / artifacts[f"kineselectsr_{mode}_selector_summary"]
                ).read_text()
            )
            validate_selection_record(info)
            if model is not None and model != records.get(
                info.get("winner_candidate_id")
            ):
                return "selected_front_mismatch"
        mode = saved["fit_config"].get("final_selection_mode", "fpms")
        chosen = json.loads(
            (
                output_json.parent / artifacts[f"kineselectsr_{mode}_selected_model"]
            ).read_text()
        )
        if saved.get("status") == "success":
            if (
                chosen is None
                or chosen["equation"] != saved["generated_equation"]["best"]
            ):
                return "selected_equation_mismatch"
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        AttributeError,
        ValidationError,
    ):
        return "saved_front_invalid"
    return None
