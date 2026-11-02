"""
nsga2.py: Elitist Non-Dominated Sorting Genetic Algorithm (NSGA-II) for Dynamic Job Shop Scheduling.
Based on Deb, Pratap, Agarwal, & Meyarivan (2002), IEEE TEVC.

Optimizes 3 or 4 conflicting objectives:
  f1: Makespan (C_max)
  f2: Total Tardiness (sum max(0, C_j - d_j))
  f3: Schedule Instability / Nervousness (Delta S relative to baseline schedule)
  f4: Total Energy Consumption (TEC in kWh, optional when include_energy=True)

Includes:
  - True Active Schedule Decoding + Warm-Start Seeding from Predictive/Right-Shift Schedules
  - Vectorized O(MN^2) Fast Non-Dominated Sorting
  - Objective-Clone Control to prevent discrete plateau duplicates from collapsing front diversity
"""

from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from src.problem import JSPInstance
from src.schedule import Schedule, decode_chromosome
from src.ga import (
    job_based_crossover,
    swap_mutation,
    light_perturb_chromosome,
    generate_n5_critical_neighbors,
)
from src.disruption import (
    BreakdownDisruption,
    DisruptedState,
    build_disrupted_state,
    extract_warm_start_chromosomes,
)


class Individual:
    def __init__(self, chromosome: List[int], active_decode: bool = True):
        self.chromosome: List[int] = list(chromosome)
        self.active_decode: bool = active_decode
        self.objectives: np.ndarray = np.zeros(3)
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.schedule: Optional[Schedule] = None

    def dominates(self, other: "Individual") -> bool:
        """Pareto dominance: True iff self <= other in all objectives and < in at least one."""
        return bool(
            np.all(self.objectives <= other.objectives)
            and np.any(self.objectives < other.objectives)
        )


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Vectorized O(MN^2) Fast Non-Dominated Sorting (Deb et al., 2002).
    Returns fronts [F_0, F_1, ...], where F_0 is the Pareto-optimal non-dominated front.
    """
    n = len(population)
    if n == 0:
        return []

    objs = np.array([ind.objectives for ind in population], dtype=float)
    # dom_matrix[i, j] == True iff individual i dominates individual j
    dom_matrix = np.all(objs[:, None, :] <= objs[None, :, :], axis=2) & np.any(
        objs[:, None, :] < objs[None, :, :], axis=2
    )
    dom_counts = np.sum(dom_matrix, axis=0).astype(int)
    dominated_lists = [np.flatnonzero(dom_matrix[i]).tolist() for i in range(n)]

    fronts_idx: List[List[int]] = []
    current_front = np.flatnonzero(dom_counts == 0).tolist()
    rank = 0

    while current_front:
        for idx in current_front:
            population[idx].rank = rank
        fronts_idx.append(current_front)
        next_front: List[int] = []
        for p_idx in current_front:
            for q_idx in dominated_lists[p_idx]:
                dom_counts[q_idx] -= 1
                if dom_counts[q_idx] == 0:
                    next_front.append(q_idx)
        rank += 1
        current_front = next_front

    return [[population[idx] for idx in front] for front in fronts_idx]


def deduplicate_population(population: List[Individual]) -> Tuple[List[Individual], List[Individual]]:
    """
    Separates individuals with distinct objective vectors from objective-space clones
    so discrete schedule duplicates do not crowd out genuine Pareto trade-offs.
    """
    unique_inds: List[Individual] = []
    clones: List[Individual] = []
    seen = set()
    for ind in population:
        key = tuple(np.round(ind.objectives, 4))
        if key not in seen:
            seen.add(key)
            unique_inds.append(ind)
        else:
            clones.append(ind)
    return unique_inds, clones


def calculate_crowding_distance(front: List[Individual]):
    """Assigns normalized Manhattan crowding distances to individuals in a front."""
    l = len(front)
    if l == 0:
        return
    for ind in front:
        ind.crowding_distance = 0.0

    if l <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    num_objectives = len(front[0].objectives)

    for m in range(num_objectives):
        front.sort(key=lambda ind: ind.objectives[m])
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]
        obj_range = obj_max - obj_min

        if obj_range <= 1e-12:
            continue

        for i in range(1, l - 1):
            if front[i].crowding_distance != float("inf"):
                dist = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
                front[i].crowding_distance += dist


def crowded_tournament_selection(population: List[Individual], k: int = 2) -> Individual:
    """Crowded comparison operator (<_n): prefers lower rank, then larger crowding distance."""
    k_eff = min(k, len(population))
    candidates = random.sample(population, k_eff)
    best = candidates[0]
    for other in candidates[1:]:
        if other.rank < best.rank:
            best = other
        elif other.rank == best.rank and other.crowding_distance > best.crowding_distance:
            best = other
    return best


def run_nsga2(
    instance: JSPInstance,
    baseline_schedule: Optional[Schedule] = None,
    pop_size: int = 60,
    generations: int = 80,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.20,
    machine_ready_times: Optional[Dict[int, int]] = None,
    job_ready_times: Optional[Dict[int, int]] = None,
    job_op_offsets: Optional[Dict[int, int]] = None,
    disruption_time: int = 0,
    disruption: Optional[BreakdownDisruption] = None,
    disrupted_state: Optional[DisruptedState] = None,
    include_energy: bool = False,
    warm_start: bool = True,
    local_search: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[Schedule], np.ndarray]:
    """
    Executes NSGA-II to find the non-dominated Pareto front across 3 (or 4) objectives:
      f1: Makespan (C_max)
      f2: Total Tardiness (sum T_j)
      f3: Schedule Instability (Delta S relative to baseline_schedule)
      f4: Total Energy Consumption (TEC in kWh, if include_energy=True)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if disrupted_state is None and baseline_schedule is not None and disruption is not None:
        disrupted_state = build_disrupted_state(baseline_schedule, disruption, instance)
    elif (
        disrupted_state is None
        and baseline_schedule is not None
        and machine_ready_times is not None
        and disruption_time > 0
    ):
        # Infer disruption from machine_ready_times if called via legacy signature
        inferred_m = max(machine_ready_times, key=lambda m: machine_ready_times[m])
        inferred_dur = max(0, machine_ready_times[inferred_m] - disruption_time)
        if inferred_dur > 0:
            disruption = BreakdownDisruption(
                machine_id=inferred_m,
                breakdown_start=disruption_time,
                duration=inferred_dur,
            )
            disrupted_state = build_disrupted_state(baseline_schedule, disruption, instance)

    frozen_ops = disrupted_state.frozen_ops if disrupted_state else None
    unavail = disrupted_state.unavailable_windows if disrupted_state else None
    min_start = disrupted_state.min_start_time if disrupted_state else disruption_time
    eff_offsets = disrupted_state.job_op_offsets if disrupted_state else job_op_offsets

    base_pool: List[int] = []
    for job in instance.jobs:
        offset = eff_offsets[job.job_id] if eff_offsets else 0
        rem_ops = len(job.operations) - offset
        base_pool.extend([job.job_id] * rem_ops)

    def evaluate_individual(ind: Individual):
        sched = decode_chromosome(
            ind.chromosome,
            instance,
            machine_ready_times=machine_ready_times,
            job_ready_times=job_ready_times,
            job_op_offsets=eff_offsets,
            frozen_ops=frozen_ops,
            unavailable_windows=unavail,
            min_start_time=min_start,
            active=ind.active_decode,
        )
        ind.schedule = sched
        f1 = sched.makespan
        f2 = sched.total_tardiness
        f3 = sched.compute_instability(baseline_schedule, impacted_only_since=min_start)
        if include_energy:
            f4 = sched.total_energy_consumption
            ind.objectives = np.array([f1, f2, f3, f4], dtype=float)
        else:
            ind.objectives = np.array([f1, f2, f3], dtype=float)

    # Initialize Population (light 5% predictive-reactive warm-starting to anchor ΔS without losing diversity)
    population: List[Individual] = []
    if warm_start and baseline_schedule is not None and disrupted_state is not None:
        ws_chroms = extract_warm_start_chromosomes(
            baseline_schedule, disrupted_state.disruption, instance
        )
        if ws_chroms and len(ws_chroms[0]) == len(base_pool):
            ws0 = ws_chroms[0]
            for act_flag in (True, False):
                if len(population) < pop_size:
                    ind = Individual(ws0, active_decode=act_flag)
                    evaluate_individual(ind)
                    population.append(ind)
            if len(population) < pop_size:
                pert = light_perturb_chromosome(ws0, num_swaps=2)
                ind = Individual(pert, active_decode=True)
                evaluate_individual(ind)
                population.append(ind)

    while len(population) < pop_size:
        chrom = list(base_pool)
        np.random.shuffle(chrom)
        ind = Individual(chrom, active_decode=True)
        evaluate_individual(ind)
        population.append(ind)

    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        calculate_crowding_distance(front)

    # Generational Loop
    for _ in range(generations):
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = crowded_tournament_selection(population)
            p2 = crowded_tournament_selection(population)

            if random.random() < crossover_rate:
                c1_chrom, c2_chrom = job_based_crossover(p1.chromosome, p2.chromosome, instance)
            else:
                c1_chrom, c2_chrom = list(p1.chromosome), list(p2.chromosome)

            c1_chrom = swap_mutation(c1_chrom, mutation_rate)
            c2_chrom = swap_mutation(c2_chrom, mutation_rate)

            act1 = p1.active_decode if random.random() < 0.25 else True
            act2 = p2.active_decode if random.random() < 0.25 else True

            ind1 = Individual(c1_chrom, active_decode=act1)
            evaluate_individual(ind1)
            offspring.append(ind1)

            if len(offspring) < pop_size:
                ind2 = Individual(c2_chrom, active_decode=act2)
                evaluate_individual(ind2)
                offspring.append(ind2)

        # Memetic N5 critical-block local search on extreme Makespan & Tardiness individuals
        if local_search and len(base_pool) >= 2 and population:
            best_ms_ind = min(population, key=lambda x: x.objectives[0])
            best_td_ind = min(population, key=lambda x: x.objectives[1])
            for anchor in (best_ms_ind, best_td_ind):
                if anchor.schedule is not None:
                    for nb_chrom in generate_n5_critical_neighbors(anchor.schedule)[:4]:
                        nb_ind = Individual(nb_chrom, active_decode=True)
                        evaluate_individual(nb_ind)
                        if not anchor.dominates(nb_ind):
                            offspring.append(nb_ind)

        # Combine and prioritize unique objective vectors before clones
        combined_unique, combined_clones = deduplicate_population(population + offspring)
        fronts = fast_non_dominated_sort(combined_unique)

        new_population: List[Individual] = []
        for front in fronts:
            calculate_crowding_distance(front)
            if len(new_population) + len(front) <= pop_size:
                new_population.extend(front)
            else:
                remaining_slots = pop_size - len(new_population)
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                new_population.extend(front[:remaining_slots])
                break

        # If fewer unique solutions than pop_size, backfill from clones sorted by rank
        if len(new_population) < pop_size and combined_clones:
            clone_fronts = fast_non_dominated_sort(combined_clones)
            for c_front in clone_fronts:
                for c_ind in c_front:
                    if len(new_population) < pop_size:
                        new_population.append(c_ind)

        population = new_population

    # Extract final Rank-0 non-dominated front
    final_fronts = fast_non_dominated_sort(population)
    pareto_front = final_fronts[0] if final_fronts else []
    unique_pareto, _ = deduplicate_population(pareto_front)

    unique_pareto.sort(key=lambda ind: tuple(ind.objectives))
    pareto_schedules = [ind.schedule for ind in unique_pareto]
    pareto_objs = np.array([ind.objectives for ind in unique_pareto], dtype=float)

    return pareto_schedules, pareto_objs
