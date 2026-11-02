"""
ga.py: Memetic Single-Objective Genetic Algorithm for Job Shop Scheduling (minimizing Makespan).
Combines:
  - Precedence Operation / Job-based Order Crossover (POX/JOX, Bierwirth 1995)
  - Hybrid Swap + Insertion Shift Mutation
  - Nowicki & Smutnicki (1996) N5 Critical-Block Local Search (Memetic Refinement)
  - True Active Schedule Decoding (Giffler & Thompson / Bierwirth left-shift)
Supports both pre-disruption predictive scheduling and post-disruption reactive rescheduling.
"""

from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from src.problem import JSPInstance
from src.schedule import Schedule, ScheduledOperation, decode_chromosome
from src.disruption import DisruptedState


def job_based_crossover(
    parent1: List[int],
    parent2: List[int],
    instance: JSPInstance,
) -> Tuple[List[int], List[int]]:
    """
    Job-based Order Crossover (JOX / POX, Bierwirth 1995; Zhang et al. 2008):
    Selects a non-empty proper subset of jobs present in the chromosome. In child 1,
    genes belonging to the subset retain their exact positions from parent 1, while
    remaining slots are filled in the relative order of parent 2. Guarantees feasibility.
    """
    n = len(parent1)
    if n < 2:
        return list(parent1), list(parent2)

    present_jobs = list(set(parent1))
    if len(present_jobs) < 2:
        return list(parent1), list(parent2)

    subset_size = random.randint(1, len(present_jobs) - 1)
    job_subset = set(random.sample(present_jobs, subset_size))

    child1 = [gene if gene in job_subset else -1 for gene in parent1]
    p2_rem = [gene for gene in parent2 if gene not in job_subset]
    ptr = 0
    for i in range(n):
        if child1[i] == -1:
            child1[i] = p2_rem[ptr]
            ptr += 1

    child2 = [gene if gene in job_subset else -1 for gene in parent2]
    p1_rem = [gene for gene in parent1 if gene not in job_subset]
    ptr = 0
    for i in range(n):
        if child2[i] == -1:
            child2[i] = p1_rem[ptr]
            ptr += 1

    return child1, child2


def swap_mutation(chromosome: List[int], mutation_rate: float = 0.15) -> List[int]:
    """
    Hybrid Operation Mutation (Swap + Insertion Shift):
    With probability `mutation_rate`, applies either a 2-point gene swap (50%)
    or a single-gene insertion shift (50%) to diversify machine precedence neighborhoods.
    """
    chrom = list(chromosome)
    n = len(chrom)
    if n < 2 or random.random() >= mutation_rate:
        return chrom

    idx1, idx2 = random.sample(range(n), 2)
    if random.random() < 0.5:
        chrom[idx1], chrom[idx2] = chrom[idx2], chrom[idx1]
    else:
        gene = chrom.pop(idx1)
        chrom.insert(idx2, gene)
    return chrom


def light_perturb_chromosome(chromosome: List[int], num_swaps: int = 1) -> List[int]:
    """Creates a close neighbor of a seed chromosome via `num_swaps` adjacent/local swaps."""
    chrom = list(chromosome)
    n = len(chrom)
    if n < 2:
        return chrom
    for _ in range(num_swaps):
        i = random.randrange(n - 1)
        j = min(n - 1, i + random.randint(1, min(4, n - 1)))
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def generate_n5_critical_neighbors(schedule: Schedule) -> List[List[int]]:
    """
    Nowicki & Smutnicki (1996) N5 Critical-Block Neighborhood Generator:
    1. Identifies the critical path terminating at C_max.
    2. Extracts maximal contiguous same-machine Critical Blocks B_k of length >= 2.
    3. Reverses only block-boundary unfrozen pairs (first pair and last pair of each block)
       via canonical topological chromosome projection, guaranteeing 100% cycle-free validity.
    """
    if not schedule.operations:
        return []

    # Build machine predecessor map
    m_pred: Dict[Tuple[int, int], ScheduledOperation] = {}
    for _, ops in schedule.machine_schedules.items():
        for i in range(1, len(ops)):
            m_pred[ops[i].op_key] = ops[i - 1]

    # Trace critical path backwards from makespan operation
    curr = max(schedule.operations, key=lambda o: (o.end_time, -o.job_id))
    crit_path: List[ScheduledOperation] = []
    while curr is not None:
        crit_path.append(curr)
        mp = m_pred.get(curr.op_key)
        if mp is not None and mp.end_time == curr.start_time:
            curr = mp
            continue
        if curr.op_idx > 0:
            jp = schedule.op_lookup.get((curr.job_id, curr.op_idx - 1))
            if jp is not None and jp.end_time == curr.start_time:
                curr = jp
                continue
        break

    crit_path.reverse()

    # Partition critical path into contiguous same-machine critical blocks
    blocks: List[List[ScheduledOperation]] = []
    for op in crit_path:
        if not blocks or blocks[-1][-1].machine_id != op.machine_id:
            blocks.append([op])
        else:
            blocks[-1].append(op)

    boundary_pairs: List[Tuple[ScheduledOperation, ScheduledOperation]] = []
    for b_idx, block in enumerate(blocks):
        if len(block) < 2:
            continue
        # First pair of block (valid for all blocks except possibly the very first if no pred, still safe to test)
        if b_idx > 0 or len(blocks) == 1:
            u, v = block[0], block[1]
            if not u.is_frozen and not v.is_frozen and u.job_id != v.job_id:
                boundary_pairs.append((u, v))
        # Last pair of block
        if len(block) >= 3 or b_idx < len(blocks) - 1:
            u2, v2 = block[-2], block[-1]
            if (u2.op_key, v2.op_key) != (block[0].op_key, block[1].op_key):
                if not u2.is_frozen and not v2.is_frozen and u2.job_id != v2.job_id:
                    boundary_pairs.append((u2, v2))

    if not boundary_pairs:
        return []

    unfrozen_ops = [op for op in schedule.operations if not op.is_frozen]
    neighbors: List[List[int]] = []

    for u, v in boundary_pairs:
        # Canonical topological sort placing v before any co-starting successor of u
        ordered = sorted(
            unfrozen_ops,
            key=lambda o: (
                o.start_time,
                1 if (o.job_id == u.job_id and o.op_idx > u.op_idx) else 0,
                o.op_idx,
                o.job_id,
            ),
        )
        pos_u = -1
        pos_v = -1
        for idx, op in enumerate(ordered):
            if op.op_key == u.op_key:
                pos_u = idx
            elif op.op_key == v.op_key:
                pos_v = idx

        if 0 <= pos_u < pos_v:
            chrom = [op.job_id for op in ordered]
            gene_u = chrom.pop(pos_u)
            # After popping pos_u < pos_v, v is now at pos_v - 1; insert gene_u right after v (at pos_v)
            chrom.insert(pos_v, gene_u)
            neighbors.append(chrom)

    return neighbors


def tournament_selection(population: List[List[int]], fitnesses: List[float], k: int = 3) -> List[int]:
    """Selects the best individual among k randomly drawn candidates."""
    k_eff = min(k, len(population))
    selected_indices = random.sample(range(len(population)), k_eff)
    best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]


def run_genetic_algorithm(
    instance: JSPInstance,
    pop_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.20,
    tournament_k: int = 3,
    elite_count: int = 2,
    machine_ready_times: Optional[Dict[int, int]] = None,
    job_ready_times: Optional[Dict[int, int]] = None,
    job_op_offsets: Optional[Dict[int, int]] = None,
    disrupted_state: Optional[DisruptedState] = None,
    warm_start_chromosomes: Optional[List[List[int]]] = None,
    local_search: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Schedule, List[int], List[float]]:
    """
    Executes Memetic Single-Objective Genetic Algorithm minimizing Makespan (C_max).
    Applies Nowicki-Smutnicki N5 critical-block refinement to the elite schedule per generation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    frozen_ops = disrupted_state.frozen_ops if disrupted_state else None
    unavail = disrupted_state.unavailable_windows if disrupted_state else None
    min_start = disrupted_state.min_start_time if disrupted_state else 0
    eff_offsets = disrupted_state.job_op_offsets if disrupted_state else job_op_offsets

    base_pool: List[int] = []
    for job in instance.jobs:
        offset = eff_offsets[job.job_id] if eff_offsets else 0
        rem_ops = len(job.operations) - offset
        base_pool.extend([job.job_id] * rem_ops)

    population: List[List[int]] = []
    if warm_start_chromosomes:
        for ws in warm_start_chromosomes:
            if len(ws) == len(base_pool) and len(population) < pop_size:
                population.append(list(ws))

    while len(population) < pop_size:
        individual = list(base_pool)
        np.random.shuffle(individual)
        population.append(individual)

    def decode_ind(ind: List[int]) -> Schedule:
        return decode_chromosome(
            ind,
            instance,
            machine_ready_times=machine_ready_times,
            job_ready_times=job_ready_times,
            job_op_offsets=eff_offsets,
            frozen_ops=frozen_ops,
            unavailable_windows=unavail,
            min_start_time=min_start,
            active=True,
        )

    schedules = [decode_ind(ind) for ind in population]
    fitnesses = [s.makespan for s in schedules]
    best_idx = int(np.argmin(fitnesses))
    best_chromosome = list(population[best_idx])
    best_schedule = schedules[best_idx]
    best_makespan = fitnesses[best_idx]
    history: List[float] = [best_makespan]

    for _ in range(generations):
        sorted_indices = np.argsort(fitnesses)
        new_population = [list(population[idx]) for idx in sorted_indices[:elite_count]]

        # Memetic N5 critical-block refinement on current top elite
        if local_search and len(base_pool) >= 2:
            elite_sched = schedules[sorted_indices[0]]
            for nb_chrom in generate_n5_critical_neighbors(elite_sched):
                nb_sched = decode_ind(nb_chrom)
                if nb_sched.makespan < fitnesses[sorted_indices[0]]:
                    new_population[0] = nb_chrom
                    if nb_sched.makespan < best_makespan:
                        best_makespan = nb_sched.makespan
                        best_chromosome = list(nb_chrom)
                        best_schedule = nb_sched
                    break

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses, k=tournament_k)
            p2 = tournament_selection(population, fitnesses, k=tournament_k)

            if random.random() < crossover_rate:
                c1, c2 = job_based_crossover(p1, p2, instance)
            else:
                c1, c2 = list(p1), list(p2)

            c1 = swap_mutation(c1, mutation_rate)
            c2 = swap_mutation(c2, mutation_rate)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population
        schedules = [decode_ind(ind) for ind in population]
        fitnesses = [s.makespan for s in schedules]

        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_makespan:
            best_makespan = fitnesses[gen_best_idx]
            best_chromosome = list(population[gen_best_idx])
            best_schedule = schedules[gen_best_idx]

        history.append(best_makespan)

    # Final steepest-descent N5 polish on best discovered schedule
    if local_search and len(base_pool) >= 2:
        improved = True
        steps = 0
        while improved and steps < 15:
            improved = False
            steps += 1
            for nb_chrom in generate_n5_critical_neighbors(best_schedule):
                nb_sched = decode_ind(nb_chrom)
                if nb_sched.makespan < best_makespan:
                    best_makespan = nb_sched.makespan
                    best_chromosome = list(nb_chrom)
                    best_schedule = nb_sched
                    improved = True
                    break

    return best_schedule, best_chromosome, history
