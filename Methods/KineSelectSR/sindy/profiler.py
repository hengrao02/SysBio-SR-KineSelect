"""GA-facing profiler interface for KineSelectSR-SINDy."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from Methods.KineSelectSR.formulation import (
    KineSelectSRReactionGene,
    KineSelectSRKineticVariant,
)
from Methods.KineSelectSR.diagnostics import (
    kinetic_law_candidate_counts,
    kinetic_law_selected_counts,
)
from .context import TargetContext, prepare_target_context
from .dictionary import (
    _build_gene_column_block_resolved,
    canonical_reaction_gene_key,
    ColumnMeta,
    GroupedDictionary,
)
from .options import SINDyKineticOptions, resolve_sindy_options_for_X
from .selector_modes import KFI_SELECTOR
from .selection import (
    solve_grouped_nnls,
    rank_variants_by_contribution,
    select_kfi_variants,
    refit_selected_model,
)

@dataclass
class FullLawSpaceProfileResult:
    """Result of evaluating the full kinetic law space for reaction genes."""
    raw_nmse_full_law_space: float
    refit_nmse_full_law_space: float
    selected_variants: tuple[KineSelectSRKineticVariant, ...]
    selected_families: tuple[str, ...]
    rate_constants: np.ndarray
    diagnostics: dict[str, object]

class FullLawSpaceProfiler:
    """Stateful profiler that caches gene-level SINDy column blocks."""

    def __init__(self, t: np.ndarray, X: np.ndarray, options: SINDyKineticOptions):
        self.t = np.asarray(t, dtype=float).reshape(-1)
        self.X = np.asarray(X, dtype=float)
        
        self.options = resolve_sindy_options_for_X(options, self.X)
            
        self.target_ctx = prepare_target_context(self.t, self.X)
        # KFI caches full-law-space columns by canonical reaction gene.
        self.gene_column_cache = {}

    def evaluate(self, genes: tuple[KineSelectSRReactionGene, ...]) -> FullLawSpaceProfileResult:
        """Evaluate the full configured law space using cached column blocks."""
        T, D = self.X.shape
        n_reactions = len(genes)
        
        if n_reactions == 0:
            dictionary = GroupedDictionary(
                columns=np.zeros((T * D, 0)),
                column_meta=[],
                group_offsets=[0],
                n_reactions=0,
            )
        else:
            all_columns = []
            all_meta = []
            group_offsets = [0]
            
            for j, gene in enumerate(genes):
                cache_key = canonical_reaction_gene_key(gene)

                if cache_key not in self.gene_column_cache:
                    self.gene_column_cache[cache_key] = _build_gene_column_block_resolved(
                        cache_key, self.X, self.target_ctx, self.options
                    )

                # Retrieve from cache and update reaction_idx
                block = self.gene_column_cache[cache_key]
                for variant, col_normed, norm in block:
                    all_columns.append(col_normed)
                    all_meta.append(
                        ColumnMeta(
                            reaction_idx=j,
                            variant=variant,
                            original_norm=norm,
                        )
                    )
                group_offsets.append(len(all_columns))
                
            if all_columns:
                columns = np.column_stack(all_columns)
            else:
                columns = np.zeros((T * D, 0))
                
            dictionary = GroupedDictionary(
                columns=columns,
                column_meta=all_meta,
                group_offsets=group_offsets,
                n_reactions=n_reactions,
            )

        # Pre-NNLS collinearity pruning
        prune_diag: dict[str, object] = {}
        if self.options.collinearity_pruning:
            from .selection import prune_collinear_columns_by_occam as _prune
            dictionary, prune_diag = _prune(
                dictionary,
                cos_threshold=self.options.collinearity_cos_threshold,
                hill_cos_threshold=self.options.hill_collinearity_cos_threshold,
            )

        # Joint NNLS with L2 regularization
        w_raw, rnorm_raw = solve_grouped_nnls(
            dictionary, self.target_ctx, l2_lambda=self.options.l2_lambda
        )
        
        # Compute raw NNLS physical residual NMSE
        if dictionary.columns.shape[1] > 0:
            raw_residual = self.target_ctx.target_norm - dictionary.columns @ w_raw
            raw_nmse = float(np.dot(raw_residual, raw_residual)) / self.target_ctx.denom
        else:
            raw_nmse = 1.0
            
        # Rank by contribution
        contributions = rank_variants_by_contribution(dictionary, w_raw, self.target_ctx)
        
        # KFI selects the highest-contribution
        # variant inside each reaction group after one full-dictionary NNLS.
        selected_reactions, selected_families = select_kfi_variants(
            dictionary, contributions, self.options.active_threshold
        )
        selector_diagnostics: dict[str, object] = {
            "selector_mode": KFI_SELECTOR
        }
        
        # Refit selected model
        model = refit_selected_model(selected_reactions, self.X, self.target_ctx, self.options)
        
        candidate_counts = kinetic_law_candidate_counts(
            meta.variant for meta in dictionary.column_meta
        )
        selected_counts = kinetic_law_selected_counts(
            model.reactions,
            model.rate_constants,
            active_threshold=getattr(self.options, "numerical_zero_threshold", 1e-14),
        )

        return FullLawSpaceProfileResult(
            raw_nmse_full_law_space=raw_nmse,
            refit_nmse_full_law_space=model.deriv_loss if model.deriv_loss is not None else 1.0,
            selected_variants=tuple(model.reactions),
            selected_families=tuple(selected_families),
            rate_constants=model.rate_constants if model.rate_constants is not None else np.array([], dtype=float),
            diagnostics={
                **candidate_counts,
                **selected_counts,
                **prune_diag,
                "collinearity_pruning_enabled": self.options.collinearity_pruning,
                "w_raw": w_raw,
                "contributions": contributions,
                "species_K_grid": self.options.species_K_grid,
                **selector_diagnostics,
            }
        )

def evaluate_full_law_space(
    genes: tuple[KineSelectSRReactionGene, ...],
    t: np.ndarray,
    X: np.ndarray,
    options: SINDyKineticOptions,
    target_ctx: TargetContext | None = None,
) -> FullLawSpaceProfileResult:
    """Evaluate the full configured law space for the given reaction genes (stateless wrapper)."""
    profiler = FullLawSpaceProfiler(t, X, options)
    if target_ctx is not None:
        profiler.target_ctx = target_ctx
    return profiler.evaluate(genes)
