from __future__ import annotations
from dataclasses import dataclass
import logging
import math
import time
from typing import Any

import numpy as np
from Methods.KineSelectSR.formulation import FormulationConfig
from Methods.KineSelectSR.sindy.options import SINDyKineticOptions
from .config import KineSelectSRSearchConfig
from .individual import (
    KineSelectSRIndividual,
    canonical_candidate_id,
)
from .operators import (
    enumerate_valid_genes,
    initialize_individual,
    mutate_replace_gene,
    crossover_exchange,
)
from .evaluator import KineSelectSREvaluator
from .pareto import build_pareto_front, alpha_select
from .fpms import fpms_select, FPMS_CRITERIA, FPMS_COST_DIMENSIONS


logger = logging.getLogger(__name__)


_INVALID_EVOLUTION_KEY = (float("inf"), 10**18, 10**18)
_INIT_UNIQUE_RETRY_MULTIPLIER = 16
_OFFSPRING_UNIQUE_RETRY_ATTEMPTS = 16
_OFFSPRING_HISTORICAL_RETRY_ATTEMPTS = 16


@dataclass
class KineSelectSRGAResult:
    """Result container for the KineSelectSR genetic algorithm search.

    Attributes:
        best_individual: The overall best candidate individual found.
        top_candidates: The global Pareto front of all evaluated individuals.
        final_population: The final population of the last run island.
        history: A list of length num_generations + 1 tracking the element-wise
            minimum best-so-far fitness across all islands.
        eval_cache: Evaluator cache containing all evaluated reaction keys.
    """
    best_individual: KineSelectSRIndividual | None
    top_candidates: list[KineSelectSRIndividual]
    final_population: list[KineSelectSRIndividual]
    history: list[float]
    eval_cache: dict
    timing: dict[str, Any] | None = None
    selection_diagnostics: dict[str, Any] | None = None

def _build_sindy_options(
    formulation_config: FormulationConfig,
    ga_config: KineSelectSRSearchConfig,
) -> SINDyKineticOptions:
    """Build SINDy options for GA evaluation from formulation and GA config."""
    return SINDyKineticOptions(
        max_reactant_order=formulation_config.max_reactant_order,
        max_product_order=formulation_config.max_product_order,
        max_stoich_per_species=formulation_config.max_stoich_per_species,
        substrate_laws=list(formulation_config.substrate_laws),
        modifier_laws=list(formulation_config.modifier_laws),
        K_grid=list(formulation_config.K_grid),
        hill_n_values=list(formulation_config.hill_n_values),
        K_grid_mode=ga_config.sindy_K_grid_mode,
        species_K_quantiles=list(ga_config.sindy_species_K_quantiles),
        species_K_positive_eps=ga_config.sindy_species_K_positive_eps,
        species_K_positive_rel_eps=ga_config.sindy_species_K_positive_rel_eps,
        species_K_floor=ga_config.sindy_species_K_floor,
        species_K_dedup_rtol=ga_config.sindy_species_K_dedup_rtol,
        active_threshold=ga_config.sindy_active_threshold,
        numerical_zero_threshold=ga_config.sindy_numerical_zero_threshold,
        top_k_per_reaction=ga_config.sindy_top_k_per_reaction,
        delta_mod=ga_config.sindy_delta_mod,
        delta_sat=ga_config.sindy_delta_sat,
        l2_lambda=formulation_config.l2_lambda,
        negative_state_policy=formulation_config.negative_state_policy,
        negative_state_tol=formulation_config.negative_state_tol,
        rollout_timeout=formulation_config.rollout_timeout,
        rate_threshold_ratio=formulation_config.rate_threshold_ratio,
        species_K_grid=formulation_config.species_K_grid,
        selector_mode=ga_config.sindy_selector_mode,
        collinearity_pruning=ga_config.sindy_collinearity_pruning,
        collinearity_cos_threshold=ga_config.sindy_collinearity_cos_threshold,
        hill_collinearity_cos_threshold=ga_config.sindy_hill_collinearity_cos_threshold,
    )


def _evolution_key(individual: KineSelectSRIndividual) -> tuple[float, int, int]:
    """Return the GA evolution ordering key.

    Derivative loss remains the primary objective, but ties are broken by model
    simplicity so the GA does not preferentially propagate false-complex models
    until the final Pareto selection step.
    """
    metrics = individual.metrics
    if metrics is None:
        return _INVALID_EVOLUTION_KEY
    return (
        float(metrics.deriv_scaled_nmse),
        int(metrics.complexity),
        int(metrics.n_active_reactions),
    )


def _best_evolution_individual(
    individuals: list[KineSelectSRIndividual],
) -> KineSelectSRIndividual | None:
    """Return the best individual under the GA evolution pressure."""
    evaluated = [ind for ind in individuals if ind.metrics is not None]
    if not evaluated:
        return None
    return min(evaluated, key=_evolution_key)


def _objective_sort_loss(individual: KineSelectSRIndividual, objective: str) -> float:
    metrics = individual.metrics
    if metrics is None:
        return float("inf")
    if objective == "rollout":
        return float(metrics.rollout_nmse)
    return float(metrics.deriv_scaled_nmse)


def _sort_key_for_objective(
    individual: KineSelectSRIndividual,
    objective: str,
) -> tuple[float, int, int]:
    metrics = individual.metrics
    if metrics is None:
        return _INVALID_EVOLUTION_KEY
    return (
        _objective_sort_loss(individual, objective),
        int(metrics.complexity),
        int(metrics.n_active_reactions),
    )


def _reaction_keys(individuals: list[KineSelectSRIndividual]) -> set[frozenset]:
    """Return the reaction-key set represented by a population slice."""
    return {ind.reaction_key for ind in individuals}


def _unique_by_reaction_key(
    individuals: list[KineSelectSRIndividual],
) -> list[KineSelectSRIndividual]:
    """Keep the first individual for each reaction key, preserving input order."""
    seen = set()
    unique = []
    for ind in individuals:
        key = ind.reaction_key
        if key in seen:
            continue
        seen.add(key)
        unique.append(ind)
    return unique


def _unique_ratio_target(
    individuals: list[KineSelectSRIndividual],
    target_size: int,
) -> float:
    """Return unique reaction keys divided by the target population size."""
    if target_size <= 0:
        return 0.0
    return len(_reaction_keys(individuals)) / float(target_size)


def _unique_ratio_actual(individuals: list[KineSelectSRIndividual]) -> float:
    """Return unique reaction keys divided by the current population length."""
    if not individuals:
        return 0.0
    return len(_reaction_keys(individuals)) / float(len(individuals))


def _estimate_unique_capacity(num_genes: int, n_reactions: int) -> int:
    """Estimate how many unique reaction-key skeletons are possible."""
    if num_genes < 0 or n_reactions < 0 or n_reactions > num_genes:
        return 0
    return int(math.comb(num_genes, n_reactions))


def _build_island_selection_front(
    island_candidates: list[KineSelectSRIndividual],
    evaluator: KineSelectSREvaluator,
    formulation_config: FormulationConfig,
    ga_config: KineSelectSRSearchConfig,
) -> tuple[list[KineSelectSRIndividual], dict[str, Any]]:
    """Build the island front used for final GA selection."""
    selection_objective = ga_config.selection_objective
    diagnostics: dict[str, Any] = {
        "selection_objective": selection_objective,
        "rollout_per_island": int(ga_config.rollout_per_island),
        "rollout_sec": 0.0,
        "rollout_candidates": 0,
        "rollout_finite": 0,
        "island_front_size": 0,
        "rollout_fallback_used": False,
    }

    if selection_objective == "deriv":
        # Final deriv selection uses the incrementally compressed global archive.
        diagnostics["island_front_size"] = 0
        return [], diagnostics

    deriv_ranked = [
        ind for ind in island_candidates
        if ind.metrics is not None and np.isfinite(ind.metrics.deriv_scaled_nmse)
    ]
    deriv_ranked.sort(key=_evolution_key)
    rollout_candidates = deriv_ranked[:ga_config.rollout_per_island]
    diagnostics["rollout_candidates"] = int(len(rollout_candidates))

    rollout_t0 = time.perf_counter()
    for ind in rollout_candidates:
        evaluator.evaluate_rollout(ind, formulation_config)
    diagnostics["rollout_sec"] = float(time.perf_counter() - rollout_t0)
    diagnostics["rollout_finite"] = int(
        sum(
            1
            for ind in rollout_candidates
            if ind.metrics is not None and np.isfinite(ind.metrics.rollout_nmse)
        )
    )

    front = build_pareto_front(rollout_candidates, objective="rollout")
    if not front:
        front = build_pareto_front(rollout_candidates, objective="deriv")
        diagnostics["rollout_fallback_used"] = True

    diagnostics["island_front_size"] = int(len(front))
    return front, diagnostics


def tournament_select(
    population: list[KineSelectSRIndividual],
    tournament_size: int,
    rng: np.random.RandomState,
    alpha: float | None = None,
    eps: float = 1e-12,
) -> KineSelectSRIndividual:
    """Select one individual via loss-first, simplicity-aware tournament selection.

    When *alpha* is ``None`` or ``<= 1.0`` the classic loss-first tournament
    (``_evolution_key``) is used.  When *alpha* > 1.0 the tournament applies
    *alpha-equivalent simplicity selection*: within a loss envelope of
    ``best_loss * alpha + eps`` the simplest participant (lowest complexity,
    then fewest active reactions, then lowest loss) wins.  This matches the
    post-hoc ``alpha_select`` semantics so that complexity pressure is active
    during the entire GA evolution, not only at the final Pareto selection.
    """
    if not population:
        raise ValueError("Cannot select from empty population")
    idx = rng.randint(0, len(population), size=tournament_size)
    participants = [population[i] for i in idx]

    if alpha is None or alpha <= 1.0:
        participants.sort(key=_evolution_key)
        return participants[0]

    evaluated = [
        ind for ind in participants
        if ind.metrics is not None
        and np.isfinite(ind.metrics.deriv_scaled_nmse)
    ]
    if not evaluated:
        participants.sort(key=_evolution_key)
        return participants[0]

    best_loss = min(float(ind.metrics.deriv_scaled_nmse) for ind in evaluated)
    threshold = best_loss * float(alpha) + float(eps)

    eligible = [
        ind for ind in evaluated
        if float(ind.metrics.deriv_scaled_nmse) <= threshold
    ]
    eligible.sort(
        key=lambda ind: (
            int(ind.metrics.complexity),
            int(ind.metrics.n_active_reactions),
            float(ind.metrics.deriv_scaled_nmse),
        )
    )
    return eligible[0]

def _run_island(
    t: np.ndarray,
    X: np.ndarray,
    n_reactions: int,
    gene_library: list,
    evaluator: KineSelectSREvaluator,
    ga_config: KineSelectSRSearchConfig,
    rng: np.random.RandomState,
    archive: dict,
    *,
    island_seed: int | None = None,
) -> tuple[
    list[KineSelectSRIndividual],
    list[float],
    dict[str, Any],
    list[KineSelectSRIndividual],
]:
    """Run a single GA island with a fixed number of reactions."""
    island_t0 = time.perf_counter()
    population_unique_capacity = _estimate_unique_capacity(
        len(gene_library),
        n_reactions,
    )
    population_target = min(ga_config.population_size, population_unique_capacity)
    population_unique_capacity_exhausted = (
        population_unique_capacity < ga_config.population_size
    )
    initial_n_elite = (
        max(1, int(population_target * ga_config.elitism))
        if population_target > 0
        else 0
    )
    logger.info(
        "KineSelectSR island n=%d start: pop=%d target=%d gen=%d elite=%d seed=%s",
        n_reactions,
        ga_config.population_size,
        population_target,
        ga_config.num_generations,
        initial_n_elite,
        island_seed if island_seed is not None else "?",
    )

    init_t0 = time.perf_counter()
    population = []
    rollout_selection = ga_config.selection_objective == "rollout"
    island_archive = {} if not rollout_selection else None
    island_candidate_keys = set()
    island_candidate_count = 0
    island_top_deriv_entries: list[
        tuple[tuple[float, int, int], int, KineSelectSRIndividual]
    ] = []
    seen_population_keys = set()
    num_duplicate_init_rejected = 0
    num_duplicate_init_fallback_accepted = 0
    num_duplicate_offspring_rejected = 0
    num_duplicate_offspring_fallback_accepted = 0
    num_historical_offspring_rejected = 0
    num_historical_offspring_fallback_accepted = 0
    num_survivor_carryover = 0

    def remember_island_candidate(ind: KineSelectSRIndividual) -> None:
        nonlocal island_candidate_count
        key = ind.reaction_key
        if key in island_candidate_keys:
            if island_archive is not None:
                island_archive[key] = ind
            return

        candidate_order = island_candidate_count
        island_candidate_keys.add(key)
        island_candidate_count += 1

        if island_archive is not None:
            island_archive[key] = ind
            return

        metrics = ind.metrics
        if metrics is None or not np.isfinite(metrics.deriv_scaled_nmse):
            return

        island_top_deriv_entries.append((_evolution_key(ind), candidate_order, ind))
        island_top_deriv_entries.sort(key=lambda item: (item[0], item[1]))
        del island_top_deriv_entries[ga_config.rollout_per_island:]

    max_init_attempts = ga_config.population_size * _INIT_UNIQUE_RETRY_MULTIPLIER
    init_attempts = 0
    while len(population) < population_target and init_attempts < max_init_attempts:
        init_attempts += 1
        ind = initialize_individual(gene_library, n_reactions, rng)
        key = ind.reaction_key
        if key in seen_population_keys:
            num_duplicate_init_rejected += 1
            continue
        evaluator.evaluate(ind)
        population.append(ind)
        archive[key] = ind
        remember_island_candidate(ind)
        seen_population_keys.add(key)

    init_sec = time.perf_counter() - init_t0
    if not population:
        raise RuntimeError(
            "KineSelectSR island failed to initialize a non-empty unique population"
        )

    unique_ratio_target_history = [
        _unique_ratio_target(population, population_target)
    ]
    unique_ratio_actual_history = [_unique_ratio_actual(population)]
    effective_population_size_history = [len(_reaction_keys(population))]
    offspring_historical_reuse_count_history: list[int] = []
    offspring_novel_count_history: list[int] = []
    offspring_accepted_count_history: list[int] = []
    offspring_slot_count_history: list[int] = []
    offspring_generation_exhausted_history: list[bool] = []
    survivor_carryover_count_history: list[int] = []
    population_size_actual_history = [len(population)]
    cache_size_growth_history: list[int] = []
    final_population_historical_reuse_ratio = 0.0

    def make_child_pair() -> tuple[KineSelectSRIndividual, KineSelectSRIndividual]:
        p1 = tournament_select(
            population, ga_config.tournament_size, rng,
            alpha=ga_config.alpha, eps=ga_config.eps,
        )
        p2 = tournament_select(
            population, ga_config.tournament_size, rng,
            alpha=ga_config.alpha, eps=ga_config.eps,
        )

        if rng.rand() < ga_config.crossover_rate:
            c1, c2 = crossover_exchange(p1, p2, gene_library, rng)
        else:
            c1, c2 = KineSelectSRIndividual(p1.genes), KineSelectSRIndividual(p2.genes)

        if rng.rand() < ga_config.mutation_rate:
            c1 = mutate_replace_gene(c1, gene_library, rng)
        if rng.rand() < ga_config.mutation_rate:
            c2 = mutate_replace_gene(c2, gene_library, rng)

        return c1, c2
        
    history = []
    # Generation 0 best fitness
    best_fit = min(ind.metrics.deriv_scaled_nmse for ind in population if ind.metrics is not None)
    history.append(best_fit)
    early_stop_enabled = ga_config.island_early_stop_patience > 0
    early_stop_reference_best = float(best_fit)
    early_stop_stale_generations = 0
    early_stop_triggered = False
    early_stop_generation: int | None = None
    best_init = _best_evolution_individual(population)
    if best_init is not None and best_init.metrics is not None:
        logger.info(
            "KineSelectSR island n=%d initialized: archive=%d cache=%d "
            "best_deriv=%.3e complexity=%d active=%d",
            n_reactions,
            len(archive),
            len(evaluator.cache),
            best_init.metrics.deriv_scaled_nmse,
            best_init.metrics.complexity,
            best_init.metrics.n_active_reactions,
        )

    evolution_t0 = time.perf_counter()
    for gen in range(ga_config.num_generations):
        historical_keys_snapshot = {
            key for key in evaluator.cache.keys() if len(key) == n_reactions
        }
        cache_size_before_generation = len(evaluator.cache)

        # Elite preservation uses derivative loss first, then simplicity.
        sorted_pop = sorted(
            population,
            key=_evolution_key,
        )
        unique_sorted_pop = _unique_by_reaction_key(sorted_pop)
        n_elite = max(1, int(len(population) * ga_config.elitism))
        n_elite = min(n_elite, len(unique_sorted_pop), population_target)
        elites = list(unique_sorted_pop[:n_elite])
        seen_next_gen_keys = _reaction_keys(elites)
        
        n_offspring = max(0, population_target - len(elites))
        offspring = []
        pending_children: list[KineSelectSRIndividual] = []
        num_historical_offspring_reused_gen = 0
        offspring_attempt_limit = (
            _OFFSPRING_UNIQUE_RETRY_ATTEMPTS
            + _OFFSPRING_HISTORICAL_RETRY_ATTEMPTS
        )
        offspring_generation_exhausted = False
        
        while len(offspring) < n_offspring:
            accepted_child = None
            attempts = 0

            while accepted_child is None and attempts < offspring_attempt_limit:
                if not pending_children:
                    pending_children.extend(make_child_pair())
                child = pending_children.pop(0)
                attempts += 1
                key = child.reaction_key
                if key in seen_next_gen_keys:
                    num_duplicate_offspring_rejected += 1
                    continue

                if key in historical_keys_snapshot:
                    num_historical_offspring_rejected += 1
                    continue

                accepted_child = child

            if accepted_child is None:
                offspring_generation_exhausted = True
                break

            key = accepted_child.reaction_key
            evaluator.evaluate(accepted_child)
            offspring.append(accepted_child)
            archive[key] = accepted_child
            remember_island_candidate(accepted_child)
            seen_next_gen_keys.add(key)

        survivor_carryover = []
        for survivor in unique_sorted_pop[n_elite:]:
            if len(elites) + len(offspring) + len(survivor_carryover) >= population_target:
                break
            key = survivor.reaction_key
            if key in seen_next_gen_keys:
                continue
            survivor_carryover.append(survivor)
            seen_next_gen_keys.add(key)

        num_survivor_carryover += len(survivor_carryover)
                    
        population = elites + offspring + survivor_carryover
        unique_ratio_target_history.append(
            _unique_ratio_target(population, population_target)
        )
        unique_ratio_actual_history.append(_unique_ratio_actual(population))
        effective_population_size_history.append(len(_reaction_keys(population)))
        offspring_historical_reuse_count_history.append(
            num_historical_offspring_reused_gen
        )
        offspring_novel_count_history.append(len(offspring))
        offspring_accepted_count_history.append(len(offspring))
        offspring_slot_count_history.append(n_offspring)
        offspring_generation_exhausted_history.append(offspring_generation_exhausted)
        survivor_carryover_count_history.append(len(survivor_carryover))
        population_size_actual_history.append(len(population))
        cache_size_growth_history.append(
            len(evaluator.cache) - cache_size_before_generation
        )
        if population:
            final_population_historical_reuse_ratio = (
                sum(
                    1
                    for ind in population
                    if ind.reaction_key in historical_keys_snapshot
                )
                / float(len(population))
            )
        
        # Track best fitness so far
        current_best = min(ind.metrics.deriv_scaled_nmse for ind in population if ind.metrics is not None)
        best_fit = min(best_fit, current_best)
        history.append(best_fit)
        best_gen = _best_evolution_individual(population)
        if best_gen is not None and best_gen.metrics is not None:
            logger.info(
                "KineSelectSR island n=%d gen=%d/%d: archive=%d cache=%d "
                "best_deriv=%.3e best_so_far=%.3e complexity=%d active=%d",
                n_reactions,
                gen + 1,
                ga_config.num_generations,
                len(archive),
                len(evaluator.cache),
                best_gen.metrics.deriv_scaled_nmse,
                best_fit,
                best_gen.metrics.complexity,
                best_gen.metrics.n_active_reactions,
            )
        if early_stop_enabled:
            min_improvement = max(
                float(ga_config.eps),
                abs(early_stop_reference_best)
                * float(ga_config.island_early_stop_min_rel_improvement),
            )
            if early_stop_reference_best - best_fit > min_improvement:
                early_stop_reference_best = float(best_fit)
                early_stop_stale_generations = 0
            else:
                early_stop_stale_generations += 1
                if early_stop_stale_generations >= ga_config.island_early_stop_patience:
                    early_stop_triggered = True
                    early_stop_generation = gen + 1
                    logger.info(
                        "KineSelectSR island n=%d early-stopped at gen=%d/%d: "
                        "patience=%d min_rel_improvement=%.3e best_so_far=%.3e",
                        n_reactions,
                        early_stop_generation,
                        ga_config.num_generations,
                        ga_config.island_early_stop_patience,
                        ga_config.island_early_stop_min_rel_improvement,
                        best_fit,
                    )
                    break
    generations_run = max(0, len(history) - 1)
    if len(history) < ga_config.num_generations + 1:
        history.extend([best_fit] * (ga_config.num_generations + 1 - len(history)))
    evolution_sec = time.perf_counter() - evolution_t0
    best_final = _best_evolution_individual(population)
    initial_unique_population_ratio_target = unique_ratio_target_history[0]
    initial_unique_population_ratio_actual = unique_ratio_actual_history[0]
    final_unique_population_ratio_target = unique_ratio_target_history[-1]
    final_unique_population_ratio_actual = unique_ratio_actual_history[-1]
    total_offspring_slots = sum(offspring_slot_count_history)
    total_historical_offspring_reused = sum(
        offspring_historical_reuse_count_history
    )
    total_offspring_accepted = sum(offspring_accepted_count_history)
    total_novel_offspring_accepted = sum(offspring_novel_count_history)
    offspring_historical_reuse_ratio = (
        total_historical_offspring_reused / float(total_offspring_accepted)
        if total_offspring_accepted > 0
        else 0.0
    )
    offspring_novel_key_ratio = (
        total_novel_offspring_accepted / float(total_offspring_slots)
        if total_offspring_slots > 0
        else 0.0
    )
    offspring_historical_reuse_ratio_history = [
        reused / float(accepted) if accepted > 0 else 0.0
        for reused, accepted in zip(
            offspring_historical_reuse_count_history,
            offspring_accepted_count_history,
        )
    ]
    offspring_novel_key_ratio_history = [
        novel / float(slots) if slots > 0 else 0.0
        for novel, slots in zip(
            offspring_novel_count_history,
            offspring_slot_count_history,
        )
    ]
    timing: dict[str, Any] = {
        "n_reactions": int(n_reactions),
        "seed": None if island_seed is None else int(island_seed),
        "total_sec": float(time.perf_counter() - island_t0),
        "init_sec": float(init_sec),
        "evolution_sec": float(evolution_sec),
        "pareto_sec": 0.0,
        "archive_size": int(len(archive)),
        "unique_archive_size": int(len(archive)),
        "cache_size": int(len(evaluator.cache)),
        "front_size": 0,
        "island_candidate_count": int(island_candidate_count),
        "island_selection_candidate_count": 0,
        "best_deriv_nmse": None,
        "best_complexity": None,
        "best_active_reactions": None,
        "population_size_configured": int(ga_config.population_size),
        "population_size_target": int(population_target),
        "initial_population_size_actual": int(population_size_actual_history[0]),
        "final_population_size_actual": int(population_size_actual_history[-1]),
        "population_size_actual_per_generation": [
            int(value) for value in population_size_actual_history
        ],
        "num_duplicate_init_rejected": int(num_duplicate_init_rejected),
        "num_duplicate_init_fallback_accepted": int(
            num_duplicate_init_fallback_accepted
        ),
        "num_duplicate_offspring_rejected": int(num_duplicate_offspring_rejected),
        "num_duplicate_offspring_fallback_accepted": int(
            num_duplicate_offspring_fallback_accepted
        ),
        "num_historical_offspring_rejected": int(
            num_historical_offspring_rejected
        ),
        "num_historical_offspring_fallback_accepted": int(
            num_historical_offspring_fallback_accepted
        ),
        "num_survivor_carryover": int(num_survivor_carryover),
        "survivor_carryover_per_generation": [
            int(value) for value in survivor_carryover_count_history
        ],
        "num_offspring_generation_exhausted": int(
            sum(1 for value in offspring_generation_exhausted_history if value)
        ),
        "offspring_generation_exhausted_per_generation": [
            bool(value) for value in offspring_generation_exhausted_history
        ],
        "initial_unique_population_ratio_target": float(
            initial_unique_population_ratio_target
        ),
        "initial_unique_population_ratio_actual": float(
            initial_unique_population_ratio_actual
        ),
        "final_unique_population_ratio_target": float(
            final_unique_population_ratio_target
        ),
        "final_unique_population_ratio_actual": float(
            final_unique_population_ratio_actual
        ),
        "min_unique_population_ratio_target": float(
            min(unique_ratio_target_history)
        ),
        "mean_unique_population_ratio_target": float(
            np.mean(unique_ratio_target_history)
        ),
        "effective_population_size_mean": float(
            np.mean(effective_population_size_history)
        ),
        "num_historical_offspring_reused": int(
            total_historical_offspring_reused
        ),
        "num_historical_offspring_novel": int(
            total_novel_offspring_accepted
        ),
        "num_offspring_generated_accepted": int(total_offspring_accepted),
        "num_offspring_slots_target": int(total_offspring_slots),
        "offspring_historical_reuse_ratio": float(
            offspring_historical_reuse_ratio
        ),
        "offspring_novel_key_ratio": float(offspring_novel_key_ratio),
        "offspring_historical_reuse_ratio_per_generation": [
            float(value) for value in offspring_historical_reuse_ratio_history
        ],
        "offspring_novel_key_ratio_per_generation": [
            float(value) for value in offspring_novel_key_ratio_history
        ],
        "cache_size_growth_per_generation": [
            int(value) for value in cache_size_growth_history
        ],
        "cache_size_growth_mean": float(
            np.mean(cache_size_growth_history) if cache_size_growth_history else 0.0
        ),
        "generations_run": int(generations_run),
        "island_early_stop_enabled": bool(early_stop_enabled),
        "island_early_stop_triggered": bool(early_stop_triggered),
        "island_early_stop_generation": (
            None if early_stop_generation is None else int(early_stop_generation)
        ),
        "island_early_stop_patience": int(ga_config.island_early_stop_patience),
        "island_early_stop_min_rel_improvement": float(
            ga_config.island_early_stop_min_rel_improvement
        ),
        "final_population_historical_reuse_ratio": float(
            final_population_historical_reuse_ratio
        ),
        "population_unique_capacity": int(population_unique_capacity),
        "population_unique_capacity_exhausted": bool(
            population_unique_capacity_exhausted
        ),
    }
    if best_final is not None and best_final.metrics is not None:
        timing.update({
            "best_deriv_nmse": float(best_final.metrics.deriv_scaled_nmse),
            "best_complexity": int(best_final.metrics.complexity),
            "best_active_reactions": int(best_final.metrics.n_active_reactions),
        })

    if rollout_selection:
        island_candidates = [entry[2] for entry in island_top_deriv_entries]
    else:
        island_candidates = list(island_archive.values())
    timing["island_selection_candidate_count"] = int(len(island_candidates))

    return population, history, timing, island_candidates

def run_ga(
    t: np.ndarray,
    X: np.ndarray,
    formulation_config: FormulationConfig,
    ga_config: KineSelectSRSearchConfig,
    seed: int | None = None,
) -> KineSelectSRGAResult:
    """Run the island-based GA reaction skeleton search framework."""
    t0 = time.perf_counter()
    t = np.asarray(t, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    D = X.shape[1]
    
    library_t0 = time.perf_counter()
    gene_library = enumerate_valid_genes(D, formulation_config)
    gene_library_sec = time.perf_counter() - library_t0
    logger.info("KineSelectSR gene library: %d genes (dim=%d)", len(gene_library), D)
    logger.info(
        "KineSelectSR GA budget: reactions=[%d..%d] pop=%d gen=%d seed=%s",
        ga_config.min_reactions,
        ga_config.max_reactions,
        ga_config.population_size,
        ga_config.num_generations,
        seed if seed is not None else ga_config.random_seed,
    )
    if not gene_library:
        logger.warning("Empty KineSelectSR gene library for dim=%d", D)
        ga_total_sec = time.perf_counter() - t0
        return KineSelectSRGAResult(
            None,
            [],
            [],
            [],
            {},
            timing={
                "ga_total_sec": float(ga_total_sec),
                "gene_library_sec": float(gene_library_sec),
                "islands_total_sec": 0.0,
                "global_pareto_sec": 0.0,
                "global_selection_sec": 0.0,
                "n_islands": 0,
                "gene_library_size": 0,
                "selection_objective": ga_config.selection_objective,
                "rollout_per_island": int(ga_config.rollout_per_island),
                "all_island_candidates": 0,
                "all_island_front_candidates": 0,
                "top_candidates": 0,
                "global_rollout_fallback_used": False,
                "cache_size": 0,
                "islands": [],
            },
            selection_diagnostics=(
                {
                    "mode": "fpms",
                    "criteria": FPMS_CRITERIA,
                    "cost_dimensions": list(FPMS_COST_DIMENSIONS),
                    "candidate_count": 0,
                    "winner_index": None,
                    "winner_score": None,
                    "winner_fidelity_cost": None,
                    "winner_parsimony_cost": None,
                    "score_margin": None,
                    "near_tie": False,
                    "degenerate_dimensions": [],
                    "fallback_reason": "no_candidates",
                    "requested_selector": "fpms",
                    "actual_selector": "fpms",
                    "objective": "rollout",
                }
                if ga_config.final_selection_mode == "fpms"
                else None
            ),
        )
        
    options = _build_sindy_options(formulation_config, ga_config)
    
    evaluator = KineSelectSREvaluator(t, X, options)
    
    archive = {}
    best_history = [float("inf")] * (ga_config.num_generations + 1)
    final_pop = []
    island_timings: list[dict[str, Any]] = []
    island_selection_candidates: list[KineSelectSRIndividual] = []
    all_island_candidates = 0

    if seed is None:
        seed = ga_config.random_seed if ga_config.random_seed is not None else 42

    for n_reactions in range(ga_config.min_reactions, ga_config.max_reactions + 1):
        if n_reactions > len(gene_library):
            logger.info(
                "KineSelectSR island n=%d skipped: n_reactions exceeds gene library size %d",
                n_reactions,
                len(gene_library),
            )
            continue

        island_seed = seed + n_reactions * 1000
        island_rng = np.random.RandomState(island_seed)

        island_total_t0 = time.perf_counter()
        population, history, island_timing, island_candidates = _run_island(
            t,
            X,
            n_reactions,
            gene_library,
            evaluator,
            ga_config,
            island_rng,
            archive,
            island_seed=island_seed,
        )
        final_pop = population
        all_island_candidates += int(
            island_timing.get("island_candidate_count", len(island_candidates))
        )
        for gen in range(min(len(best_history), len(history))):
            best_history[gen] = min(best_history[gen], history[gen])

        selection_front, selection_timing = _build_island_selection_front(
            island_candidates,
            evaluator,
            formulation_config,
            ga_config,
        )
        island_selection_candidates.extend(selection_front)

        # Incremental Pareto archive compression: after each island, discard dominated
        # individuals from the archive and keep only the current Pareto front.
        # By Pareto transitivity this is equivalent to building the global front from
        # all evaluated individuals, but limits archive size to O(front_size) rather
        # than O(n_islands × pop_size × n_generations).
        pareto_t0 = time.perf_counter()
        current_front = build_pareto_front(list(archive.values()))
        pareto_sec = time.perf_counter() - pareto_t0
        archive = {ind.reaction_key: ind for ind in current_front}
        island_timing.update(selection_timing)
        island_timing.update({
            "pareto_sec": float(pareto_sec),
            "total_sec": float(time.perf_counter() - island_total_t0),
            "front_size": int(len(current_front)),
            "archive_size": int(len(archive)),
            "unique_archive_size": int(len(archive)),
            "cache_size": int(len(evaluator.cache)),
        })
        if ga_config.selection_objective == "rollout":
            logger.info(
                "KineSelectSR island n=%d rollout selection: candidates=%d finite=%d "
                "rollout=%.1fs island_front=%d fallback=%s",
                n_reactions,
                island_timing["rollout_candidates"],
                island_timing["rollout_finite"],
                island_timing["rollout_sec"],
                island_timing["island_front_size"],
                island_timing["rollout_fallback_used"],
            )
        island_timings.append(island_timing)
        best_island = _best_evolution_individual(population)
        if best_island is not None and best_island.metrics is not None:
            logger.info(
                "KineSelectSR island n=%d finished: total=%.1fs init=%.1fs "
                "evolution=%.1fs pareto=%.1fs front=%d cache=%d "
                "best_deriv=%.3e complexity=%d active=%d",
                n_reactions,
                island_timing["total_sec"],
                island_timing["init_sec"],
                island_timing["evolution_sec"],
                island_timing["pareto_sec"],
                len(current_front),
                len(evaluator.cache),
                best_island.metrics.deriv_scaled_nmse,
                best_island.metrics.complexity,
                best_island.metrics.n_active_reactions,
            )

    # Build top candidates from the per-island selection fronts.
    # best_individual is derived from this front by the requested selector so
    # it is consistent with top_candidates (no per-island greedy mismatch).
    global_pareto_t0 = time.perf_counter()
    selection_objective = ga_config.selection_objective
    global_rollout_fallback_used = False
    global_selection_objective = selection_objective
    if selection_objective == "deriv":
        top_candidates = list(archive.values())
    else:
        top_candidates = build_pareto_front(
            island_selection_candidates,
            objective=selection_objective,
        )
    if selection_objective == "rollout" and not top_candidates:
        top_candidates = build_pareto_front(list(archive.values()), objective="deriv")
        global_rollout_fallback_used = True
        global_selection_objective = "deriv"
    top_candidates.sort(
        key=lambda ind: _sort_key_for_objective(ind, global_selection_objective)
    )
    global_pareto_sec = time.perf_counter() - global_pareto_t0
    global_selection_t0 = time.perf_counter()
    selection_diagnostics: dict[str, Any] | None = None
    if not top_candidates:
        best_ind = None
        if ga_config.final_selection_mode == "fpms":
            selection_diagnostics = {
                "mode": "fpms",
                "criteria": FPMS_CRITERIA,
                "cost_dimensions": list(FPMS_COST_DIMENSIONS),
                "candidate_count": 0,
                "winner_index": None,
                "winner_score": None,
                "winner_fidelity_cost": None,
                "winner_parsimony_cost": None,
                "score_margin": None,
                "near_tie": False,
                "degenerate_dimensions": [],
                "fallback_reason": "no_candidates",
                "requested_selector": "fpms",
                "actual_selector": "fpms",
                "objective": global_selection_objective,
            }
    elif ga_config.final_selection_mode == "fpms":
        if global_rollout_fallback_used:
            best_ind = alpha_select(
                top_candidates,
                ga_config.alpha,
                ga_config.eps,
                objective="deriv",
            )
            selection_diagnostics = {
                "mode": "fpms",
                "criteria": FPMS_CRITERIA,
                "cost_dimensions": list(FPMS_COST_DIMENSIONS),
                "candidate_count": int(len(top_candidates)),
                "winner_index": (
                    None if best_ind is None else top_candidates.index(best_ind)
                ),
                "winner_score": None,
                "winner_fidelity_cost": None,
                "winner_parsimony_cost": None,
                "score_margin": None,
                "near_tie": False,
                "degenerate_dimensions": [],
                "fallback_reason": "global_rollout_front_unavailable",
                "requested_selector": "fpms",
                "actual_selector": "alpha",
                "objective": "deriv",
            }
        else:
            fpms_result = fpms_select(
                top_candidates,
                eps=ga_config.eps,
                active_threshold=ga_config.fpms_active_threshold,
            )
            best_ind = fpms_result.winner
            ranked_scores = sorted(fpms_result.scores, key=lambda item: item.rank)
            winner_score = ranked_scores[0] if ranked_scores else None
            score_margin = (
                None
                if len(ranked_scores) < 2
                else ranked_scores[0].score - ranked_scores[1].score
            )
            selection_diagnostics = {
                "mode": "fpms",
                "criteria": FPMS_CRITERIA,
                "cost_dimensions": list(FPMS_COST_DIMENSIONS),
                "candidate_count": int(len(top_candidates)),
                "winner_index": (
                    None if winner_score is None else winner_score.candidate_index
                ),
                "winner_score": (
                    None if winner_score is None else winner_score.score
                ),
                "winner_fidelity_cost": (
                    None if winner_score is None else winner_score.fidelity_cost
                ),
                "winner_parsimony_cost": (
                    None if winner_score is None else winner_score.parsimony_cost
                ),
                "score_margin": score_margin,
                "near_tie": (
                    False if score_margin is None else score_margin <= 1e-12
                ),
                "degenerate_dimensions": list(
                    fpms_result.degenerate_dimensions
                ),
                "fallback_reason": fpms_result.fallback_reason,
                "requested_selector": "fpms",
                "actual_selector": "fpms",
                "objective": "rollout",
            }
    else:
        best_ind = alpha_select(
            top_candidates,
            ga_config.alpha,
            ga_config.eps,
            objective=global_selection_objective,
        )
        selection_diagnostics = {
            "mode": "alpha",
            "candidate_count": int(len(top_candidates)),
            "winner_index": (
                None if best_ind is None else top_candidates.index(best_ind)
            ),
            "winner_candidate_id": (
                None if best_ind is None else canonical_candidate_id(best_ind)
            ),
            "fallback_reason": None,
            "requested_selector": "alpha",
            "actual_selector": "alpha",
            "objective": global_selection_objective,
            "alpha": float(ga_config.alpha),
            "eps": float(ga_config.eps),
        }
    global_selection_sec = time.perf_counter() - global_selection_t0
    ga_total_sec = time.perf_counter() - t0
    islands_total_sec = float(sum(item["total_sec"] for item in island_timings))
    timing: dict[str, Any] = {
        "ga_total_sec": float(ga_total_sec),
        "gene_library_sec": float(gene_library_sec),
        "islands_total_sec": islands_total_sec,
        "global_pareto_sec": float(global_pareto_sec),
        "global_selection_sec": float(global_selection_sec),
        "n_islands": int(len(island_timings)),
        "gene_library_size": int(len(gene_library)),
        "selection_objective": selection_objective,
        "rollout_per_island": int(ga_config.rollout_per_island),
        "all_island_candidates": int(all_island_candidates),
        "all_island_front_candidates": int(len(island_selection_candidates)),
        "top_candidates": int(len(top_candidates)),
        "global_rollout_fallback_used": bool(global_rollout_fallback_used),
        "global_selection_objective": global_selection_objective,
        "final_selection_mode": ga_config.final_selection_mode,
        "fpms_active_threshold": float(ga_config.fpms_active_threshold),
        "fpms_criteria": ga_config.fpms_criteria,
        "cache_size": int(len(evaluator.cache)),
        "islands": island_timings,
    }
    logger.info(
        "KineSelectSR GA complete: total=%.1fs library=%.1fs islands=%.1fs "
        "global_pareto=%.1fs top_candidates=%d cache=%d",
        ga_total_sec,
        gene_library_sec,
        islands_total_sec,
        global_pareto_sec,
        len(top_candidates),
        len(evaluator.cache),
    )
    if best_ind is None:
        logger.warning("KineSelectSR global selection produced no best individual")
    elif best_ind.metrics is not None:
        logger.info(
            "KineSelectSR global selected: objective=%s deriv_nmse=%.3e "
            "rollout_nmse=%.3e complexity=%d active=%d",
            global_selection_objective,
            best_ind.metrics.deriv_scaled_nmse,
            best_ind.metrics.rollout_nmse,
            best_ind.metrics.complexity,
            best_ind.metrics.n_active_reactions,
        )

    return KineSelectSRGAResult(
        best_individual=best_ind,
        top_candidates=top_candidates,
        final_population=final_pop,
        history=best_history,
        eval_cache=evaluator.cache,
        timing=timing,
        selection_diagnostics=selection_diagnostics,
    )
