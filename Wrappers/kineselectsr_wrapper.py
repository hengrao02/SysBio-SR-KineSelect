from __future__ import annotations

import logging
import time
from typing import Any
import numpy as np
from dataclasses import fields

from Wrappers.base_ode_sr_wrapper import BaseODESRWrapper, EquationResult
from Methods.KineSelectSR.formulation import FormulationConfig, KineSelectSRModel
from Methods.KineSelectSR.formulation.formatting import format_model, format_model_substituted
from Methods.KineSelectSR.formulation.ode import rollout
from Methods.KineSelectSR.presets import apply_kinetic_preset
from Methods.KineSelectSR.search import KineSelectSRGAResult, KineSelectSRSearchConfig, run_ga, serialize_ga_result


logger = logging.getLogger(__name__)


def _dedupe_candidate_strings(candidates: list[str | None]) -> list[str | None]:
    """Remove duplicate formatted candidate equations while preserving order."""
    seen = set()
    unique: list[str | None] = []
    for candidate in candidates:
        key = candidate.strip() if isinstance(candidate, str) else candidate
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _configure_verbose_logging() -> None:
    """Enable INFO logs for KineSelectSR without touching the root logger."""
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for logger_name in (__name__, "Methods.KineSelectSR"):
        target_logger = logging.getLogger(logger_name)
        target_logger.setLevel(logging.INFO)
        target_logger.propagate = False
        has_verbose_handler = any(
            getattr(handler, "_kineselectsr_verbose", False)
            for handler in target_logger.handlers
        )
        if not has_verbose_handler:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            setattr(handler, "_kineselectsr_verbose", True)
            target_logger.addHandler(handler)


def _reject_legacy_selector_options(values: dict[str, Any]) -> None:
    search_fields = {f.name for f in fields(KineSelectSRSearchConfig)}
    unknown = sorted(
        key for key in values
        if key.startswith("sindy_") and key not in search_fields
    )
    if unknown:
        raise ValueError(f"Unsupported kinetic-form options: {unknown}")
    legacy = [key for key in values if key.startswith("pfns_")]
    if legacy or values.get("final_selection_mode") == "pfns":
        raise ValueError(
            "Legacy PFNS options are unsupported; use final_selection_mode='fpms' "
            "with fpms_criteria='DR_KX'. FPMS uses four criteria, not the old five."
        )


class KineSelectSRWrapper(BaseODESRWrapper):
    """Unified wrapper for the KineSelectSR genetic algorithm search."""

    method_name = "kineselectsr"

    def __init__(self, **kwargs: Any) -> None:
        _reject_legacy_selector_options(kwargs)
        super().__init__()
        self._verbose = bool(kwargs.pop("verbose", False))
        self.init_kwargs = kwargs
        self._model: KineSelectSRModel | None = None
        self._formulation_config: FormulationConfig | None = None
        self._ga_config: KineSelectSRSearchConfig | None = None
        self._ga_result: KineSelectSRGAResult | None = None

    def fit(
        self,
        times: np.ndarray | list[np.ndarray],
        trajectories: np.ndarray | list[np.ndarray],
        **kwargs: Any,
    ) -> EquationResult:
        fit_t0 = time.perf_counter()
        # Validate trajectories
        times, trajectories = self.validate_trajectories(times, trajectories)

        # Unpack lists of length 1
        if isinstance(times, list) or isinstance(trajectories, list):
            if not isinstance(times, list) or not isinstance(trajectories, list):
                raise TypeError("times and trajectories must either both be lists or both be arrays")
            if len(times) != 1 or len(trajectories) != 1:
                raise NotImplementedError(
                    "KineSelectSRWrapper currently supports one trajectory per fit call."
                )
            times = times[0]
            trajectories = trajectories[0]

        # Construct FormulationConfig and KineSelectSRSearchConfig using kwargs passed to __init__ or fit
        all_kwargs = {**self.init_kwargs, **kwargs}
        _reject_legacy_selector_options(all_kwargs)
        all_kwargs = apply_kinetic_preset(all_kwargs)
        verbose = bool(all_kwargs.pop("verbose", self._verbose))
        preset_metadata = {
            key: all_kwargs.get(key)
            for key in ("kinetic_preset", "kinetic_preset_phase")
            if all_kwargs.get(key) is not None
        }

        # Resolve field names dynamically from the dataclasses
        formulation_fields = {f.name for f in fields(FormulationConfig)}
        search_fields = {f.name for f in fields(KineSelectSRSearchConfig)}

        # Reconcile explicit deprecated aliases without mirroring the active GA
        # budget back into inactive FormulationConfig fields. Their frozen
        # defaults remain 1..8 while min_reactions/max_reactions alone control
        # the GA islands (2..12 in the paper profile).
        # Canonical (read by the GA loop): min_reactions / max_reactions.
        # Deprecated alias (no code consumes them): *_per_model. They are
        # accepted for backward compatibility but, if BOTH are given and they
        # disagree, we fail fast rather than silently letting the canonical key
        # win — a divergence almost always means a config mistake.
        for canonical, alias in (
            ("max_reactions", "max_reactions_per_model"),
            ("min_reactions", "min_reactions_per_model"),
        ):
            has_canon = canonical in all_kwargs
            has_alias = alias in all_kwargs
            if has_canon and has_alias:
                if all_kwargs[canonical] != all_kwargs[alias]:
                    raise ValueError(
                        f"Conflicting reaction-count config: {canonical}="
                        f"{all_kwargs[canonical]} but {alias}={all_kwargs[alias]}. "
                        f"'{alias}' is a deprecated alias of '{canonical}'; set only "
                        f"'{canonical}' (preferred) or keep both equal."
                    )
            elif has_alias and not has_canon:
                all_kwargs[canonical] = all_kwargs[alias]

        formulation_kwargs = {k: v for k, v in all_kwargs.items() if k in formulation_fields}
        search_kwargs = {k: v for k, v in all_kwargs.items() if k in search_fields}

        formulation_config = FormulationConfig(**formulation_kwargs)
        ga_config = KineSelectSRSearchConfig(**search_kwargs)

        self._formulation_config = formulation_config
        self._ga_config = ga_config

        # Run genetic algorithm
        if verbose:
            _configure_verbose_logging()
        logger.info(
            "KineSelectSR GA: dim=%d T=%d reactions=[%d..%d] pop=%d gen=%d seed=%d "
            "selection=%s rollout_per_island=%d substrate_laws=%s "
            "modifier_laws=%s",
            trajectories.shape[1],
            trajectories.shape[0],
            ga_config.min_reactions,
            ga_config.max_reactions,
            ga_config.population_size,
            ga_config.num_generations,
            ga_config.random_seed,
            ga_config.selection_objective,
            ga_config.rollout_per_island,
            formulation_config.substrate_laws,
            formulation_config.modifier_laws,
        )
        result = run_ga(
            t=times,
            X=trajectories,
            formulation_config=formulation_config,
            ga_config=ga_config,
            seed=ga_config.random_seed
        )
        self._ga_result = result

        if result.best_individual is None:
            kineselectsr_metadata = serialize_ga_result(result)
            kineselectsr_metadata.update(preset_metadata)
            timing = dict(kineselectsr_metadata.get("timing") or {})
            timing["fit_total_sec"] = float(time.perf_counter() - fit_t0)
            kineselectsr_metadata["timing"] = timing
            kineselectsr_metadata["reason"] = "GA returned no valid individual"
            logger.warning("KineSelectSR GA returned no valid individual")
            return self._mark_fitted(
                best=None,
                candidates=[],
                status="failed",
                metadata={"kineselectsr": kineselectsr_metadata},
            )

        best_model = result.best_individual.to_model()
        self._model = best_model

        best_eq = format_model(best_model)
        best_substituted = format_model_substituted(best_model)

        # Format candidates
        raw_candidate_strings = []
        for cand in result.top_candidates:
            cand_model = cand.to_model()
            cand_str = format_model(cand_model)
            raw_candidate_strings.append(cand_str)
        candidates = _dedupe_candidate_strings(raw_candidate_strings)

        kineselectsr_metadata = serialize_ga_result(result)
        kineselectsr_metadata.update(preset_metadata)
        kineselectsr_metadata["best_substituted"] = best_substituted
        kineselectsr_metadata["num_candidate_strings_raw"] = len(raw_candidate_strings)
        kineselectsr_metadata["num_candidate_strings_unique"] = len(candidates)
        kineselectsr_metadata["num_candidate_strings_deduped"] = (
            len(raw_candidate_strings) - len(candidates)
        )
        kineselectsr_metadata["top_candidate_equations"] = list(candidates)
        timing = dict(kineselectsr_metadata.get("timing") or {})
        timing["fit_total_sec"] = float(time.perf_counter() - fit_t0)
        kineselectsr_metadata["timing"] = timing

        status = "success" if best_eq is not None else "failed"
        metrics = result.best_individual.metrics
        if metrics is not None:
            logger.info(
                "KineSelectSR selected: status=%s deriv_nmse=%.3e rollout_nmse=%.3e "
                "complexity=%d active=%d top_candidates=%d candidate_strings=%d",
                status,
                metrics.deriv_scaled_nmse,
                metrics.rollout_nmse,
                metrics.complexity,
                metrics.n_active_reactions,
                len(result.top_candidates),
                len(candidates),
            )

        return self._mark_fitted(
            best=best_eq,
            candidates=candidates,
            status=status,
            metadata={
                "best_substituted": best_substituted,
                "kineselectsr": kineselectsr_metadata,
            },
        )

    def predict(
        self,
        times: np.ndarray | list[np.ndarray],
        y0: np.ndarray | list[np.ndarray] | None = None,
        **kwargs: Any,
    ) -> np.ndarray | list[np.ndarray] | None:
        if not self.is_fitted():
            raise RuntimeError("KineSelectSRWrapper must be fitted before calling predict().")
        if self._model is None or self._formulation_config is None:
            return None

        rollout_kwargs = dict(kwargs)
        rollout_kwargs.setdefault(
            "timeout", self._formulation_config.rollout_timeout
        )

        # Check if list of trajectories
        if isinstance(times, list):
            if y0 is None or not isinstance(y0, list):
                raise TypeError("For list input times, y0 must be a list of initial conditions.")
            outputs = []
            for t_i, y0_i in zip(times, y0):
                pred_i = rollout(
                    model=self._model,
                    t=np.asarray(t_i, dtype=float),
                    y0=np.asarray(y0_i, dtype=float),
                    config=self._formulation_config,
                    **rollout_kwargs,
                )
                outputs.append(pred_i)
            return outputs

        if y0 is None:
            raise ValueError("y0 must be provided for prediction.")

        return rollout(
            model=self._model,
            t=np.asarray(times, dtype=float),
            y0=np.asarray(y0, dtype=float),
            config=self._formulation_config,
            **rollout_kwargs,
        )
