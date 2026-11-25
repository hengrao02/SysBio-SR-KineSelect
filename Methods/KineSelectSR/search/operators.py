from __future__ import annotations
import numpy as np
from Methods.KineSelectSR.formulation import KineSelectSRReactionGene, FormulationConfig
from Methods.KineSelectSR.formulation.constraints import is_valid_gene
from .individual import KineSelectSRIndividual

# Hard ceiling on species dimension for gene enumeration. The naive enumeration
# space is (max_stoich+1)**dim per side; even with sum-budget pruning the count
# of valid skeletons grows fast with dim. We cap at 7 (current benchmark scope)
# to fail loudly rather than hang on pathological inputs.
MAX_SUPPORTED_DIM = 7


def _enumerate_bounded_vectors(dim: int, max_stoich: int, max_order: int):
    """Yield length-`dim` stoichiometric vectors with each entry in
    ``[0, max_stoich]`` and total sum ``<= max_order``.

    Produces the SAME ordering as
    ``(v for v in itertools.product(range(max_stoich+1), repeat=dim)
       if sum(v) <= max_order)`` — lexicographic, last position varying
    fastest — but prunes branches that exceed the sum budget instead of
    materializing the full ``(max_stoich+1)**dim`` space. Order preservation is
    required so fixed-seed GA runs draw identical genes from the library.
    """
    vec = [0] * dim

    def rec(pos: int, remaining: int):
        if pos == dim:
            yield tuple(vec)
            return
        hi = max_stoich if max_stoich < remaining else remaining
        for v in range(hi + 1):
            vec[pos] = v
            yield from rec(pos + 1, remaining - v)
        vec[pos] = 0

    yield from rec(0, max_order)


def enumerate_valid_genes(dim: int, config: FormulationConfig) -> list[KineSelectSRReactionGene]:
    """Enumerate valid reaction genes for a given dimension and configuration."""
    if dim <= 0:
        return []
    if dim > MAX_SUPPORTED_DIM:
        raise ValueError(
            f"enumerate_valid_genes: dim={dim} exceeds MAX_SUPPORTED_DIM="
            f"{MAX_SUPPORTED_DIM}. The enumeration space grows steeply with the "
            f"species dimension; raise MAX_SUPPORTED_DIM deliberately only after "
            f"confirming the gene-library size and GA cost are acceptable."
        )

    max_stoich = config.max_stoich_per_species

    reactant_vectors = list(
        _enumerate_bounded_vectors(dim, max_stoich, config.max_reactant_order)
    )
    product_vectors = list(
        _enumerate_bounded_vectors(dim, max_stoich, config.max_product_order)
    )

    valid_genes = []
    if any(law != "none" for law in config.modifier_laws):
        modifiers = [None] + list(range(dim))
    else:
        modifiers = [None]

    for r_vec in reactant_vectors:
        for p_vec in product_vectors:
            for m_idx in modifiers:
                gene = KineSelectSRReactionGene(reactants=r_vec, products=p_vec, modifier_idx=m_idx)
                if is_valid_gene(gene, config):
                    valid_genes.append(gene)

    return valid_genes

def initialize_individual(
    gene_library: list[KineSelectSRReactionGene],
    n_reactions: int,
    random_state: np.random.RandomState,
) -> KineSelectSRIndividual:
    """Randomly initialize an individual with unique genes from the library."""
    if len(gene_library) < n_reactions:
        raise ValueError("gene_library size is smaller than n_reactions")
    
    chosen_indices = random_state.choice(len(gene_library), size=n_reactions, replace=False)
    genes = tuple(gene_library[idx] for idx in chosen_indices)
    return KineSelectSRIndividual(genes)

def mutate_replace_gene(
    individual: KineSelectSRIndividual,
    gene_library: list[KineSelectSRReactionGene],
    random_state: np.random.RandomState,
    max_attempts: int = 32,
) -> KineSelectSRIndividual:
    """Mutate an individual by replacing exactly one of its genes with a new unique one.

    Uses rejection sampling (O(n_reactions) average) rather than scanning the
    full library (O(|gene_library|)).  Falls back to a full scan only when the
    library is nearly exhausted (i.e. n_reactions ≈ |gene_library|).
    """
    n = len(individual.genes)
    if n == 0 or len(gene_library) <= n:
        return KineSelectSRIndividual(individual.genes)

    idx_to_replace = random_state.randint(0, n)
    current_set = set(individual.genes)
    lib_size = len(gene_library)

    # Fast path: rejection sampling — O(n_reactions) expected when library >> n
    for _ in range(max_attempts):
        new_gene = gene_library[random_state.randint(0, lib_size)]
        if new_gene not in current_set:
            new_genes = list(individual.genes)
            new_genes[idx_to_replace] = new_gene
            return KineSelectSRIndividual(tuple(new_genes))

    # Fallback: full scan for dense libraries (O(|gene_library|))
    candidates = [g for g in gene_library if g not in current_set]
    if not candidates:
        return KineSelectSRIndividual(individual.genes)

    new_genes = list(individual.genes)
    new_genes[idx_to_replace] = candidates[random_state.randint(0, len(candidates))]
    return KineSelectSRIndividual(tuple(new_genes))

def refill_genes(
    child_genes: list[KineSelectSRReactionGene],
    n: int,
    gene_library: list[KineSelectSRReactionGene],
    random_state: np.random.RandomState,
    max_attempts: int = 32,
) -> list[KineSelectSRReactionGene]:
    """Refill a gene list to length n using unique genes drawn from the library.

    Uses rejection sampling per slot (O(n_reactions) average) rather than
    scanning the full library each time.  Falls back to a full scan when the
    library is nearly exhausted.
    """
    current_set = set(child_genes)
    n_needed = n - len(current_set)
    if n_needed <= 0:
        return list(current_set)[:n]

    lib_size = len(gene_library)
    if lib_size - len(current_set) < n_needed:
        raise ValueError("Not enough unique genes in library to refill child to size n.")

    for _ in range(n_needed):
        # Rejection sampling for this slot
        placed = False
        for _ in range(max_attempts):
            g = gene_library[random_state.randint(0, lib_size)]
            if g not in current_set:
                child_genes.append(g)
                current_set.add(g)
                placed = True
                break
        if not placed:
            # Fallback: full scan (O(|gene_library|)) for dense case
            remaining = [g for g in gene_library if g not in current_set]
            if not remaining:
                raise ValueError("Not enough unique genes in library to refill child to size n.")
            chosen = remaining[random_state.randint(0, len(remaining))]
            child_genes.append(chosen)
            current_set.add(chosen)

    return child_genes

def crossover_exchange(
    parent1: KineSelectSRIndividual,
    parent2: KineSelectSRIndividual,
    gene_library: list[KineSelectSRReactionGene],
    random_state: np.random.RandomState,
) -> tuple[KineSelectSRIndividual, KineSelectSRIndividual]:
    """Crossover two individuals by exchanging random subsets of their genes.

    Uses independent random permutations so the exchanged subsets are free of
    the sort-order bias inherent in simple one-point crossover on sorted tuples.
    """
    n = len(parent1.genes)
    if n <= 1:
        return KineSelectSRIndividual(parent1.genes), KineSelectSRIndividual(parent2.genes)
        
    k = random_state.randint(1, n)
    
    g1 = parent1.genes
    g2 = parent2.genes

    idx1 = random_state.permutation(n)
    idx2 = random_state.permutation(n)

    c1_genes = [g1[i] for i in idx1[:k]] + [g2[i] for i in idx2[k:]]
    c2_genes = [g2[i] for i in idx2[:k]] + [g1[i] for i in idx1[k:]]
    
    # Deduplicate child 1
    seen1 = set()
    c1_dedup = []
    for g in c1_genes:
        if g not in seen1:
            seen1.add(g)
            c1_dedup.append(g)
            
    # Deduplicate child 2
    seen2 = set()
    c2_dedup = []
    for g in c2_genes:
        if g not in seen2:
            seen2.add(g)
            c2_dedup.append(g)
            
    # Refill if needed
    if len(c1_dedup) < n:
        c1_dedup = refill_genes(c1_dedup, n, gene_library, random_state)
    if len(c2_dedup) < n:
        c2_dedup = refill_genes(c2_dedup, n, gene_library, random_state)
        
    return KineSelectSRIndividual(tuple(c1_dedup)), KineSelectSRIndividual(tuple(c2_dedup))
