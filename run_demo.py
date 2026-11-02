"""
run_demo.py: End-to-end demonstration of Dynamic Job Shop Scheduling under Machine Disruptions.
Demonstrates:
  1. Giffler-Thompson Active Dispatching Baselines (FCFS, SPT, EDD)
  2. Single-Objective GA Predictive Schedule (True Active Schedule Decoding)
  3. Physically Consistent Machine Breakdown Injection & State Partitioning
  4. Reactive Baselines: Right-Shift, Reactive FCFS, Reactive SPT, Reactive Single-Obj GA
  5. Warm-Started NSGA-II Multi-Objective Rescheduling (Makespan, Tardiness, Instability)
  6. Automated Constraint Verification (`schedule.validate()`) & Publication Gantt/Pareto Plots
"""

import os
import sys
import time
import numpy as np
from tabulate import tabulate

from src.problem import load_jsp_instance
from src.baselines import solve_fcfs, solve_spt, solve_edd
from src.ga import run_genetic_algorithm
from src.disruption import (
    BreakdownDisruption,
    apply_right_shift,
    build_disrupted_state,
    extract_warm_start_chromosomes,
)
from src.nsga2 import run_nsga2
from src.visualize import plot_gantt_chart, plot_pareto_front

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    print("=" * 88)
    print(" DYNAMIC JOB SHOP SCHEDULING UNDER MACHINE DISRUPTIONS: COMPLETE BENCHMARK DEMO")
    print("=" * 88)

    # 1. Load Benchmark Instance (FT06 - 6 jobs, 6 machines, 36 ops; known optimal C_max = 55)
    benchmark_path = "data/benchmarks/ft06.txt"
    print(f"\n[1] Loading Benchmark: {benchmark_path} ...")
    instance = load_jsp_instance(benchmark_path, tightness_factor=1.5)
    print(
        f"    - Jobs: {instance.num_jobs}, Machines: {instance.num_machines}, "
        f"Total Ops: {instance.total_operations}, Bottleneck Machine: M{instance.get_bottleneck_machine()}"
    )
    for j in instance.jobs:
        print(f"      Job {j.job_id}: Total Proc = {j.total_processing_time:2d}, Due Date = {j.due_date:2d}")

    # 2. Pre-Disruption Classical Dispatching Heuristics (Giffler-Thompson Active)
    print("\n[2] Evaluating Classical Active Dispatching Rules (Pre-Disruption)...")
    t0 = time.time()
    sched_fcfs = solve_fcfs(instance)
    fcfs_time = time.time() - t0

    t0 = time.time()
    sched_spt = solve_spt(instance)
    spt_time = time.time() - t0

    t0 = time.time()
    sched_edd = solve_edd(instance)
    edd_time = time.time() - t0

    assert sched_fcfs.validate() and sched_spt.validate() and sched_edd.validate()
    print(f"    - FCFS: Makespan = {sched_fcfs.makespan}, Tardiness = {sched_fcfs.total_tardiness}, Energy = {sched_fcfs.total_energy_consumption:.1f} kWh")
    print(f"    - SPT : Makespan = {sched_spt.makespan}, Tardiness = {sched_spt.total_tardiness}, Energy = {sched_spt.total_energy_consumption:.1f} kWh")
    print(f"    - EDD : Makespan = {sched_edd.makespan}, Tardiness = {sched_edd.total_tardiness}, Energy = {sched_edd.total_energy_consumption:.1f} kWh")

    # 3. Single-Objective GA: Predictive Baseline Schedule
    print("\n[3] Running Single-Objective GA (Active Decoding, Optimizing Makespan)...")
    t0 = time.time()
    initial_ga_sched, _, _ = run_genetic_algorithm(
        instance,
        pop_size=50,
        generations=80,
        crossover_rate=0.85,
        mutation_rate=0.20,
        seed=42,
    )
    ga_time = time.time() - t0
    assert initial_ga_sched.validate()
    print(
        f"    - GA Predictive: Makespan = {initial_ga_sched.makespan} (FT06 Optimal = 55), "
        f"Tardiness = {initial_ga_sched.total_tardiness}, Energy = {initial_ga_sched.total_energy_consumption:.1f} kWh ({ga_time:.2f}s)"
    )

    os.makedirs("results", exist_ok=True)
    plot_gantt_chart(
        initial_ga_sched,
        title="Initial Predictive Schedule (Single-Objective Active GA)",
        save_path="results/01_initial_ga_schedule.png",
    )

    # 4. Inject Machine Disruption on Machine 2 at t=20 for D=15 units
    disruption = BreakdownDisruption(machine_id=2, breakdown_start=20, duration=15)
    state = build_disrupted_state(initial_ga_sched, disruption, instance)
    print(f"\n[4] Injecting Disruption: {disruption}")
    print(
        f"    - Frozen Ops (Completed <= {disruption.breakdown_start} or In-Progress on Healthy Machines): "
        f"{len(state.frozen_ops)}"
    )
    print(f"    - Remaining Unscheduled Ops: {len(state.remaining_chromosome_pool)}")

    # 5. Evaluate Post-Disruption Reactive Baselines
    print("\n[5] Evaluating Post-Disruption Reactive Baselines...")
    t0 = time.time()
    sched_rs = apply_right_shift(initial_ga_sched, disruption, instance)
    rs_time = time.time() - t0

    t0 = time.time()
    sched_r_fcfs = solve_fcfs(instance, disrupted_state=state)
    rfcfs_time = time.time() - t0

    t0 = time.time()
    sched_r_spt = solve_spt(instance, disrupted_state=state)
    rspt_time = time.time() - t0

    ws_chroms = extract_warm_start_chromosomes(initial_ga_sched, disruption, instance)
    t0 = time.time()
    sched_r_ga, _, _ = run_genetic_algorithm(
        instance,
        pop_size=50,
        generations=70,
        disrupted_state=state,
        warm_start_chromosomes=ws_chroms,
        seed=42,
    )
    rga_time = time.time() - t0

    for s in (sched_rs, sched_r_fcfs, sched_r_spt, sched_r_ga):
        assert s.validate(), "Schedule failed feasibility validation!"

    rs_instab = sched_rs.compute_instability(initial_ga_sched, disruption.breakdown_start)
    rfcfs_instab = sched_r_fcfs.compute_instability(initial_ga_sched, disruption.breakdown_start)
    rspt_instab = sched_r_spt.compute_instability(initial_ga_sched, disruption.breakdown_start)
    rga_instab = sched_r_ga.compute_instability(initial_ga_sched, disruption.breakdown_start)

    plot_gantt_chart(
        sched_rs,
        disruption=disruption,
        baseline_schedule=initial_ga_sched,
        title="Post-Disruption Baseline: Right-Shift Rescheduling",
        save_path="results/02_right_shift_schedule.png",
    )

    # 6. Run Warm-Started NSGA-II Multi-Objective Rescheduling
    print("\n[6] Running Warm-Started NSGA-II (Objectives: Makespan, Tardiness, Instability)...")
    t0 = time.time()
    pareto_schedules, pareto_objs = run_nsga2(
        instance,
        baseline_schedule=initial_ga_sched,
        pop_size=60,
        generations=80,
        crossover_rate=0.85,
        mutation_rate=0.20,
        disrupted_state=state,
        warm_start=True,
        seed=42,
    )
    nsga2_time = time.time() - t0
    for s in pareto_schedules:
        assert s.validate(), "NSGA-II Pareto schedule failed validation!"

    print(
        f"    - NSGA-II completed in {nsga2_time:.2f}s | Discovered {len(pareto_schedules)} "
        f"validated non-dominated Pareto schedules."
    )

    # 7. Plot Pareto Trade-Off Frontier with ALL Reactive Baselines
    baseline_pts = {
        "Right-Shift": (sched_rs.makespan, sched_rs.total_tardiness, rs_instab),
        "Reactive FCFS": (sched_r_fcfs.makespan, sched_r_fcfs.total_tardiness, rfcfs_instab),
        "Reactive SPT": (sched_r_spt.makespan, sched_r_spt.total_tardiness, rspt_instab),
        "Reactive GA": (sched_r_ga.makespan, sched_r_ga.total_tardiness, rga_instab),
    }
    plot_pareto_front(
        pareto_objs,
        baseline_points=baseline_pts,
        save_path="results/03_pareto_frontier.png",
        title="FT06 Post-Disruption Pareto Frontier (NSGA-II) vs Reactive Baselines",
    )

    # Save Gantt charts for extreme Pareto solutions
    best_ms_idx = int(np.argmin(pareto_objs[:, 0]))
    best_ms_sched = pareto_schedules[best_ms_idx]
    plot_gantt_chart(
        best_ms_sched,
        disruption=disruption,
        baseline_schedule=initial_ga_sched,
        title=f"NSGA-II Pareto Solution (Min Makespan = {best_ms_sched.makespan})",
        save_path="results/04_nsga2_min_makespan.png",
    )

    best_stab_idx = int(np.argmin(pareto_objs[:, 2]))
    best_stab_sched = pareto_schedules[best_stab_idx]
    plot_gantt_chart(
        best_stab_sched,
        disruption=disruption,
        baseline_schedule=initial_ga_sched,
        title=f"NSGA-II Pareto Solution (Min Instability ΔS = {int(pareto_objs[best_stab_idx, 2])})",
        save_path="results/05_nsga2_min_instability.png",
    )

    # 8. Print Complete Comparative Summary Table
    print("\n" + "=" * 88)
    print(" COMPLETE COMPARATIVE SUMMARY (FT06, M2 BREAKDOWN [20, 35])")
    print("=" * 88)

    table_data = [
        ["FCFS (Pre-Disruption)", sched_fcfs.makespan, sched_fcfs.total_tardiness, "-", "-", f"{sched_fcfs.total_energy_consumption:.1f}", f"{fcfs_time*1000:.2f} ms"],
        ["SPT (Pre-Disruption)", sched_spt.makespan, sched_spt.total_tardiness, "-", "-", f"{sched_spt.total_energy_consumption:.1f}", f"{spt_time*1000:.2f} ms"],
        ["EDD (Pre-Disruption)", sched_edd.makespan, sched_edd.total_tardiness, "-", "-", f"{sched_edd.total_energy_consumption:.1f}", f"{edd_time*1000:.2f} ms"],
        ["Single-Obj GA (Predictive)", initial_ga_sched.makespan, initial_ga_sched.total_tardiness, "-", "-", f"{initial_ga_sched.total_energy_consumption:.1f}", f"{ga_time:.2f} s"],
        ["Reactive FCFS (Post-Disruption)", sched_r_fcfs.makespan, sched_r_fcfs.total_tardiness, rfcfs_instab, sched_r_fcfs.compute_sequence_instability(initial_ga_sched), f"{sched_r_fcfs.total_energy_consumption:.1f}", f"{rfcfs_time*1000:.2f} ms"],
        ["Reactive SPT (Post-Disruption)", sched_r_spt.makespan, sched_r_spt.total_tardiness, rspt_instab, sched_r_spt.compute_sequence_instability(initial_ga_sched), f"{sched_r_spt.total_energy_consumption:.1f}", f"{rspt_time*1000:.2f} ms"],
        ["Right-Shift (Post-Disruption)", sched_rs.makespan, sched_rs.total_tardiness, rs_instab, sched_rs.compute_sequence_instability(initial_ga_sched), f"{sched_rs.total_energy_consumption:.1f}", f"{rs_time*1000:.2f} ms"],
        ["Reactive GA (Post-Disruption)", sched_r_ga.makespan, sched_r_ga.total_tardiness, rga_instab, sched_r_ga.compute_sequence_instability(initial_ga_sched), f"{sched_r_ga.total_energy_consumption:.1f}", f"{rga_time:.2f} s"],
    ]

    for i, sol in enumerate(pareto_schedules):
        ms = int(pareto_objs[i, 0])
        tar = int(pareto_objs[i, 1])
        instab = int(pareto_objs[i, 2])
        seq_swaps = sol.compute_sequence_instability(initial_ga_sched)
        label = f"NSGA-II Pareto #{i+1}"
        if i == best_ms_idx:
            label += " [Min C_max]"
        elif i == best_stab_idx:
            label += " [Min ΔS]"
        table_data.append([label, ms, tar, instab, seq_swaps, f"{sol.total_energy_consumption:.1f}", f"{nsga2_time:.2f} s"])

    headers = ["Method / Solution", "Makespan", "Tardiness", "ΔS (Time)", "Seq Swaps", "Energy (kWh)", "Runtime"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 88)
    print("\nAll schedules passed `validate()`! Updated figures saved to `results/01_*.png` .. `05_*.png`.")


if __name__ == "__main__":
    main()
