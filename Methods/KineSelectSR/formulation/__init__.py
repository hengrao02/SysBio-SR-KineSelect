"""KineSelectSR local formulation module."""

from .reaction import (
    KineSelectSRReactionGene,
    KineSelectSRKineticVariant,
    KineSelectSRModel,
    KineSelectSREvalMetrics,
    FormulationConfig,
    validate_model,
)
from .rate import compute_rate, compute_rate_batch
from .ode import build_ode_rhs, rollout, get_last_rollout_failure_reason
from .fitter import (
    build_design_matrix,
    build_target,
    fit_rate_constants,
    safe_x_dot_max,
    compute_contributions,
    threshold_by_contribution,
    threshold_rates,
)
from .complexity import compute_complexity
from .evaluate import evaluate
from .constraints import (
    is_valid,
    is_valid_gene,
    enumerate_valid_reactions,
    has_reactant_product_overlap,
    has_noncanonical_source_products,
)
from .formatting import format_model, format_model_substituted

__all__ = [
    "KineSelectSRReactionGene",
    "KineSelectSRKineticVariant",
    "KineSelectSRModel",
    "KineSelectSREvalMetrics",
    "FormulationConfig",
    "validate_model",
    "compute_rate",
    "compute_rate_batch",
    "build_ode_rhs",
    "rollout",
    "get_last_rollout_failure_reason",
    "build_design_matrix",
    "build_target",
    "fit_rate_constants",
    "safe_x_dot_max",
    "compute_contributions",
    "threshold_by_contribution",
    "threshold_rates",
    "compute_complexity",
    "evaluate",
    "is_valid",
    "is_valid_gene",
    "enumerate_valid_reactions",
    "has_reactant_product_overlap",
    "has_noncanonical_source_products",
    "format_model",
    "format_model_substituted",
]
