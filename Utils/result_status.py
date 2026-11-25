from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np


_VAR_PATTERN = re.compile(r"\bx_(\d+)\b")


@dataclass(frozen=True)
class ResultStatus:
    """Final run status used by SysBioSR summaries and batch aggregation."""

    status: str
    failure_reason: str | None = None
    error: str | None = None


def normalize_failure_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    normalized = str(reason).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def _failure_status(reason: str | None) -> str:
    if reason == "timeout":
        return "timeout"
    if reason is not None and reason.startswith("invalid_"):
        return "invalid"
    return "failed"


def split_equation_system(best_equation: str | None) -> list[str]:
    if not isinstance(best_equation, str):
        return []
    return [part.strip() for part in best_equation.split("|") if part.strip()]


def detect_invalid_equation_dimension(best_equation: str | None, num_states: int | None) -> str | None:
    if not isinstance(best_equation, str) or num_states is None:
        return None

    parts = split_equation_system(best_equation)
    if parts and len(parts) != num_states:
        return "invalid_equation_dimension"

    for match in _VAR_PATTERN.finditer(best_equation):
        if int(match.group(1)) >= num_states:
            return "invalid_equation_dimension"
    return None


def decide_result_status(
    *,
    best_equation: str | None,
    num_states: int | None = None,
    pred_trajectory: Any = None,
    target_trajectory: Any = None,
    metrics: dict[str, Any] | None = None,
    failure_reason: str | None = None,
    error: str | None = None,
) -> ResultStatus:
    reason = normalize_failure_reason(failure_reason)

    if not isinstance(best_equation, str) or not best_equation.strip():
        final_reason = reason or "no_equation"
        return ResultStatus(status=_failure_status(final_reason), failure_reason=final_reason, error=error)

    reason = reason or detect_invalid_equation_dimension(best_equation, num_states)
    if reason is not None:
        return ResultStatus(status=_failure_status(reason), failure_reason=reason, error=error)

    if pred_trajectory is None:
        final_reason = reason or "rollout_failed"
        return ResultStatus(status=_failure_status(final_reason), failure_reason=final_reason, error=error)

    pred = np.asarray(pred_trajectory)
    if pred.ndim != 2:
        return ResultStatus(status="invalid", failure_reason="invalid_prediction_shape", error=error)

    if target_trajectory is not None:
        target = np.asarray(target_trajectory)
        if pred.shape != target.shape:
            return ResultStatus(status="invalid", failure_reason="invalid_prediction_shape", error=error)

    if metrics is None:
        final_reason = reason or "metrics_failed"
        return ResultStatus(status=_failure_status(final_reason), failure_reason=final_reason, error=error)

    return ResultStatus(status="success", failure_reason=None, error=error)
