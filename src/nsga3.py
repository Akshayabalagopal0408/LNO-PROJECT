"""
nsga3.py: Reference-Point-Based Many-Objective Evolutionary Algorithm (NSGA-III)
for Dynamic Green Job Shop Scheduling.
Based on Deb & Jain (2014), IEEE Transactions on Evolutionary Computation, 18(4), 577-601.

Specifically designed for 4+ conflicting objectives where NSGA-II's Manhattan crowding
distance loses selection pressure and boundary/interior uniformity:
  f1: Makespan (C_max)
  f2: Total Tardiness (sum max(0, C_j - d_j))
  f3: Schedule Instability (Delta S)
  f4: Total Energy Consumption (TEC in kWh)

Implements:
  - Das & Dennis (1998) systematic simplex reference point generation
  - Persistent ideal point z_min and Achievement Scalarizing Function (ASF) extreme-point
    hyperplane intercept normalization (Deb & Jain, 2014, Section IV-C)
  - Vectorized perpendicular ray-distance association & niche-preservation selection
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
from src.nsga2 import Individual, fast_non_dominated_sort, deduplicate_population
from src.disruption import (
    BreakdownDisruption,
    DisruptedState,
    build_disrupted_state,
    extract_warm_start_chromosomes,
)


def generate_reference_points(num_objectives: int, num_divisions: int) -> np.ndarray:
    """
    Generates uniformly spaced reference points on a normalized unit simplex
    using Das and Dennis's systematic approach (sum(x_i) = 1).
    Number of points H = binom(M + p - 1, p).
    """
    def _generate_recursive(dim: int, depth: int, left: int, total: int):
        if depth == dim - 1:
            return [[left / total]]
        res = []
        for i in range(left + 1):
            for suffix in _generate_recursive(dim, depth + 1, left - i, total):
                res.append([i / total] + suffix)
        return res

    points = _generate_recursive(num_objectives, 0, num_divisions, num_divisions)
    return np.array(points, dtype=float)


def compute_asf_intercepts(
    translated_st: np.ndarray,
    translated_f0: np.ndarray,
) -> np.ndarray:
    """
    Computes objective-axis intercepts using Deb & Jain's (2014) Achievement Scalarizing
    Function (ASF) extreme-point hyperplane construction, with degenerate-matrix safeguards.
    """
    num_obj = translated_st.shape[1]
    extreme_pts = np.zeros((num_obj, num_obj), dtype=float)

    for j in range(num_obj):
        w = np.full(num_obj, 1e-6, dtype=float)
        w[j] = 1.0
        asf = np.max(translated_st / w[None, :], axis=1)
        # Tie-break by L1 norm
        best_idx = int(np.lexsort((np.sum(translated_st, axis=1), asf))[0])
        extreme_pts[j] = translated_st[best_idx]

    intercepts = None
    try:
        if np.linalg.matrix_rank(extreme_pts) == num_obj:
            u = np.linalg.solve(extreme_pts, np.ones(num_obj, dtype=float))
            if np.all(np.isfinite(u)) and np.all(u > 1e-6):
                cand_a = 1.0 / u
                f0_max = np.max(translated_f0, axis=0)
                # Guard against negative or wildly collapsed hyperplane intercepts
                if np.all(cand_a >= 0.1 * f0_max) and np.all(cand_a > 1e-6):
                    intercepts = cand_a
    except np.linalg.LinAlgError:
        intercepts = None

    if intercepts is None:
        # Standard Deb & Jain fallback: nadir estimate from non-dominated front F_0, then S_t
        intercepts = np.max(translated_f0, axis=0).copy()
        st_max = np.max(translated_st, axis=0)
        mask = intercepts <= 1e-6
        intercepts[mask] = st_max[mask]
        intercepts[intercepts <= 1e-6] = 1.0

    return intercepts


def associate_to_reference_points(
    population: List[Individual],
    reference_points: np.ndarray,
    ideal_point: np.ndarray,
    intercepts: np.ndarray,
) -> Tuple[List[int], List[float]]:
    """
    Normalizes objectives via (f - z_min) / intercepts and associates each individual
    with its nearest reference ray via perpendicular Euclidean distance.
    """
    safe_denom = np.where(intercepts > 1e-6, intercepts, 1.0)
    raw_objs = np.array([ind.objectives for ind in population], dtype=float)
    norm_objs = (raw_objs - ideal_point[None, :]) / safe_denom[None, :]

    ray_norms = np.linalg.norm(reference_points, axis=1, keepdims=True)
    ray_norms[ray_norms == 0] = 1.0
    unit_rays = reference_points / ray_norms  # (H, M)

    # Vectorized perpendicular distances: (K, H)
    proj_scales = norm_objs @ unit_rays.T
    proj_vecs = proj_scales[:, :, None] * unit_rays[None, :, :]
    perp_dists = np.linalg.norm(norm_objs[:, None, :] - proj_vecs, axis=2)

    best_rps = np.argmin(perp_dists, axis=1)
    min_dists = perp_dists[np.arange(len(population)), best_rps]

    return best_rps.tolist(), min_dists.tolist()


def rank_tournament_selection(population: List[Individual], k: int = 2) -> Individual:
    """Selects parent preferring lower non-dominated rank (random tie-breaking per NSGA-III)."""
    k_eff = min(k, len(population))
    candidates = random.sample(population, k_eff)
    best = candidates[0]
    for other in candidates[1:]:
        if other.rank < best.rank or (other.rank == best.rank and random.random() < 0.5):
            best = other
    return best


def run_nsga3(
    instance: JSPInstance,
    baseline_schedule: Optional[Schedule] = None,
    pop_size: int = 56,
    generations: int = 70,
    num_divisions: int = 5,  # For M=4, p=5 => H = binom(8, 5) = 56 reference points
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.20,
    machine_ready_times: Optional[Dict[int, int]] = None,
    job_ready_times: Optional[Dict[int, int]] = None,
    job_op_offsets: Optional[Dict[int, int]] = None,
    disruption_time: int = 0,
    disruption: Optional[BreakdownDisruption] = None,
    disrupted_state: Optional[DisruptedState] = None,
    include_energy: bool = True,
    warm_start: bool = True,
    local_search: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[Schedule], np.ndarray]:
    """
    Runs NSGA-III for Many-Objective Dynamic Scheduling (3 or 4 objectives).
    If include_energy is True:
      f1: Makespan (C_max)
      f2: Total Tardiness (sum T_j)
      f3: Schedule Instability (Delta S)
      f4: Total Energy Consumption (TEC in kWh)
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

    num_objectives = 4 if include_energy else 3
    ref_points = generate_reference_points(num_objectives, num_divisions)

    base_pool: List[int] = []
    for job in instance.jobs:
        offset = eff_offsets[job.job_id] if eff_offsets else 0
        rem_ops = len(job.operations) - offset
        base_pool.extend([job.job_id] * rem_ops)

    def evaluate_ind(ind: Individual):
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

    # Initial Population (light 5% predictive-reactive warm-starting to anchor ΔS without losing diversity)
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
                    evaluate_ind(ind)
                    population.append(ind)
            if len(population) < pop_size:
                pert = light_perturb_chromosome(ws0, num_swaps=2)
                ind = Individual(pert, active_decode=True)
                evaluate_ind(ind)
                population.append(ind)

    while len(population) < pop_size:
        chrom = list(base_pool)
        np.random.shuffle(chrom)
        ind = Individual(chrom, active_decode=True)
        evaluate_ind(ind)
        population.append(ind)

    fast_non_dominated_sort(population)
    ideal_pt = np.min(np.array([ind.objectives for ind in population]), axis=0)

    # Generational Loop
    for _ in range(generations):
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = rank_tournament_selection(population)
            p2 = rank_tournament_selection(population)

            if random.random() < crossover_rate:
                c1_chrom, c2_chrom = job_based_crossover(p1.chromosome, p2.chromosome, instance)
            else:
                c1_chrom, c2_chrom = list(p1.chromosome), list(p2.chromosome)

            c1_chrom = swap_mutation(c1_chrom, mutation_rate)
            c2_chrom = swap_mutation(c2_chrom, mutation_rate)

            act1 = p1.active_decode if random.random() < 0.25 else True
            act2 = p2.active_decode if random.random() < 0.25 else True

            ind1 = Individual(c1_chrom, active_decode=act1)
            evaluate_ind(ind1)
            offspring.append(ind1)
            if len(offspring) < pop_size:
                ind2 = Individual(c2_chrom, active_decode=act2)
                evaluate_ind(ind2)
                offspring.append(ind2)

        # Memetic N5 critical-block local search on extreme Makespan & Tardiness individuals
        if local_search and len(base_pool) >= 2 and population:
            best_ms_ind = min(population, key=lambda x: x.objectives[0])
            best_td_ind = min(population, key=lambda x: x.objectives[1])
            for anchor in (best_ms_ind, best_td_ind):
                if anchor.schedule is not None:
                    for nb_chrom in generate_n5_critical_neighbors(anchor.schedule)[:4]:
                        nb_ind = Individual(nb_chrom, active_decode=True)
                        evaluate_ind(nb_ind)
                        if not anchor.dominates(nb_ind):
                            offspring.append(nb_ind)

        # Update persistent ideal point z_min
        combined_all = population + offspring
        ideal_pt = np.minimum(ideal_pt, np.min([ind.objectives for ind in combined_all], axis=0))

        # Prioritize distinct objective vectors over clones
        combined_unique, combined_clones = deduplicate_population(combined_all)
        fronts = fast_non_dominated_sort(combined_unique)

        selected_pool: List[Individual] = []
        last_front_idx = -1
        for f_idx, front in enumerate(fronts):
            if len(selected_pool) + len(front) <= pop_size:
                selected_pool.extend(front)
            else:
                last_front_idx = f_idx
                break

        if last_front_idx == -1:
            # All unique individuals fit; backfill remaining slots from sorted clones if needed
            if len(selected_pool) < pop_size and combined_clones:
                clone_fronts = fast_non_dominated_sort(combined_clones)
                for c_front in clone_fronts:
                    for c_ind in c_front:
                        if len(selected_pool) < pop_size:
                            selected_pool.append(c_ind)
            population = selected_pool
            continue

        # Niche-preservation selection on splitting front F_l
        front_l = fronts[last_front_idx]
        needed_slots = pop_size - len(selected_pool)

        eval_subset = selected_pool + front_l
        translated_st = np.array([ind.objectives for ind in eval_subset], dtype=float) - ideal_pt[None, :]
        translated_f0 = np.array([ind.objectives for ind in fronts[0]], dtype=float) - ideal_pt[None, :]
        intercepts = compute_asf_intercepts(translated_st, translated_f0)

        assoc_rp, dists = associate_to_reference_points(eval_subset, ref_points, ideal_pt, intercepts)

        niche_counts = np.zeros(len(ref_points), dtype=int)
        for i in range(len(selected_pool)):
            niche_counts[assoc_rp[i]] += 1

        front_l_candidates = []
        for i in range(len(front_l)):
            idx_in_eval = len(selected_pool) + i
            front_l_candidates.append({
                "ind": front_l[i],
                "rp": assoc_rp[idx_in_eval],
                "dist": dists[idx_in_eval],
            })

        while needed_slots > 0 and front_l_candidates:
            active_rps = list({c["rp"] for c in front_l_candidates})
            min_niche = min(niche_counts[rp] for rp in active_rps)
            candidate_rps = [rp for rp in active_rps if niche_counts[rp] == min_niche]

            chosen_rp = random.choice(candidate_rps)
            matching = [c for c in front_l_candidates if c["rp"] == chosen_rp]

            if niche_counts[chosen_rp] == 0:
                best_cand = min(matching, key=lambda c: c["dist"])
            else:
                best_cand = random.choice(matching)

            selected_pool.append(best_cand["ind"])
            niche_counts[chosen_rp] += 1
            front_l_candidates.remove(best_cand)
            needed_slots -= 1

        population = selected_pool

    # Extract final Rank-0 Pareto front
    final_fronts = fast_non_dominated_sort(population)
    pareto_front = final_fronts[0] if final_fronts else []
    unique_pareto, _ = deduplicate_population(pareto_front)

    unique_pareto.sort(key=lambda ind: tuple(ind.objectives))
    pareto_scheds = [ind.schedule for ind in unique_pareto]
    pareto_objs = np.array([ind.objectives for ind in unique_pareto], dtype=float)

    return pareto_scheds, pareto_objs
