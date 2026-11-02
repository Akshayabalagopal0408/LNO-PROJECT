"""
run_dynamic_horizon.py: Multi-Event Rolling-Horizon Rescheduling Experiment.

Addresses sequential real-world shop-floor disruptions by simulating a 3-stage
cascading operational horizon (Vieira et al., 2003; Ouelhadj & Petrovic, 2009):
  - Stage 0 (t = 0) : Predictive schedule built for 6 baseline jobs (FT06, C_max = 55).
  - Event 1 (t = 16): Breakdown #1 hits Machine 2 for 14 units [16, 30].
                      Shop floor executes U-NSGA-III and adopts the Knee schedule.
  - Event 2 (t = 34): Secondary Breakdown #2 hits Machine 4 for 14 units [34, 48]
                      AND an Urgent Rush-Order Job (J6, release_time = 34, tight due date)
                      arrives dynamically on the shop floor!
                      Rolling-horizon U-NSGA-III freezes all completed/in-progress work up
                      to t = 34, preserves both repair windows, integrates Rush Job J6,
                      and computes a new Pareto-optimal schedule vs. Right-Shift + FCFS.

Outputs `results/dynamic_horizon_summary.csv` and `results/09_multi_disruption_rush_order_gantt.png`.
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem import load_jsp_instance, Job, Operation
from src.schedule import Schedule, ScheduledOperation
from src.disruption import (
    BreakdownDisruption,
    DisruptedState,
    build_disrupted_state,
    apply_right_shift,
)
from src.baselines import solve_fcfs
from src.ga import run_genetic_algorithm
from src.nsga3 import run_nsga3

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def select_knee_schedule(schedules, objs) -> tuple:
    span = np.where(np.ptp(objs, axis=0) > 0, np.ptp(objs, axis=0), 1.0)
    norm = (objs - np.min(objs, axis=0)) / span
    idx = int(np.argmin(np.linalg.norm(norm, axis=1)))
    return schedules[idx], objs[idx]


def build_stage2_with_rush_job(
    active_schedule: Schedule,
    disruption1: BreakdownDisruption,
    disruption2: BreakdownDisruption,
    base_instance,
):
    """
    Creates augmented JSPInstance with urgent Rush Job J6 arriving at t = disruption2.breakdown_start,
    and constructs the multi-breakdown DisruptedState at t_d2.
    """
    aug_instance = copy.deepcopy(base_instance)
    t_d2 = disruption2.breakdown_start

    # Urgent Rush Order J6: 4 high-priority operations across machines [1, 3, 4, 0]
    rush_id = aug_instance.num_jobs
    rush_specs = [(1, 4), (3, 5), (4, 4), (0, 3)]
    rush_ops = [
        Operation(job_id=rush_id, op_idx=idx, machine_id=m, duration=d)
        for idx, (m, d) in enumerate(rush_specs)
    ]
    rush_proc = sum(d for _, d in rush_specs)
    rush_job = Job(
        job_id=rush_id,
        operations=rush_ops,
        release_time=t_d2,
        due_date=t_d2 + int(round(1.3 * rush_proc)),  # Tight urgent SLA (k = 1.3)
    )
    aug_instance.jobs.append(rush_job)
    aug_instance.num_jobs += 1

    # Partition active_schedule at t_d2 (preserving previous breakdown window on M2 as well)
    base_state2 = build_disrupted_state(active_schedule, disruption2, base_instance)

    unavail = {m: list(base_state2.unavailable_windows.get(m, [])) for m in range(aug_instance.num_machines)}
    unavail[disruption1.machine_id].append((disruption1.breakdown_start, disruption1.breakdown_end))
    for m in unavail:
        unavail[m] = sorted(set(unavail[m]))

    job_ready = dict(base_state2.job_ready_times)
    job_ready[rush_id] = t_d2

    job_offsets = dict(base_state2.job_op_offsets)
    job_offsets[rush_id] = 0

    rem_pool = list(base_state2.remaining_chromosome_pool) + [rush_id] * len(rush_ops)

    aug_state = DisruptedState(
        disruption=disruption2,
        frozen_ops=base_state2.frozen_ops,
        machine_ready_times=base_state2.machine_ready_times,
        job_ready_times=job_ready,
        job_op_offsets=job_offsets,
        remaining_chromosome_pool=rem_pool,
        unavailable_windows=unavail,
    )
    return aug_instance, aug_state


def apply_right_shift_plus_fifo_rush(
    stage1_schedule: Schedule,
    aug_instance,
    aug_state: DisruptedState,
    disruption2: BreakdownDisruption,
) -> Schedule:
    """
    Classical heuristic baseline at Stage 2: Right-Shifts existing jobs (J0..J5) around
    Breakdown #2 and appends urgent Rush Job J6 in FIFO order after queued work.
    """
    base_inst = stage1_schedule.instance
    rs_existing = apply_right_shift(stage1_schedule, disruption2, base_inst)

    all_ops = list(rs_existing.operations)
    m_end = {
        m: max([op.end_time for op in rs_existing.machine_schedules[m]], default=0)
        for m in range(aug_instance.num_machines)
    }
    rush_job = aug_instance.jobs[-1]
    j_avail = rush_job.release_time

    for op in rush_job.operations:
        m = op.machine_id
        start_t = max(m_end[m], j_avail, disruption2.breakdown_start)
        if m == disruption2.machine_id and start_t < disruption2.breakdown_end:
            start_t = disruption2.breakdown_end
        end_t = start_t + op.duration
        m_end[m] = end_t
        j_avail = end_t
        all_ops.append(
            ScheduledOperation(
                job_id=rush_job.job_id,
                op_idx=op.op_idx,
                machine_id=m,
                start_time=start_t,
                end_time=end_t,
                duration=op.duration,
                is_frozen=False,
            )
        )

    sched = Schedule(
        instance=aug_instance,
        operations=all_ops,
        unavailable_windows=aug_state.unavailable_windows,
    )
    sched.finalize()
    return sched


def plot_multi_event_gantt(
    schedule: Schedule,
    disruptions: list,
    rush_job_id: int,
    save_path: str,
):
    num_machines = schedule.instance.num_machines
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=180)
    cmap = matplotlib.colormaps["tab10"]

    for op in schedule.operations:
        y_pos = op.machine_id - 0.35
        is_rush = op.job_id == rush_job_id
        color = "#ffbe0b" if is_rush else cmap(op.job_id % 10)
        edge_w = 1.8 if is_rush else 1.1
        edge_s = "--" if op.is_frozen else "-"
        alpha_v = 0.72 if op.is_frozen else 0.95

        rect = patches.Rectangle(
            (op.start_time, y_pos),
            op.duration,
            0.70,
            linewidth=edge_w,
            linestyle=edge_s,
            edgecolor="#000000",
            facecolor=color,
            alpha=alpha_v,
            zorder=3,
        )
        ax.add_patch(rect)
        tag = f"RUSH\nJ{op.job_id}O{op.op_idx}" if is_rush and op.duration >= 4 else f"J{op.job_id}O{op.op_idx}"
        if op.duration >= 2:
            ax.text(
                op.start_time + op.duration / 2,
                y_pos + 0.35,
                tag,
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                zorder=4,
            )

    for idx, d in enumerate(disruptions, start=1):
        ax.axvline(x=d.breakdown_start, color="#c1121f", linestyle="--", linewidth=1.5, zorder=5)
        b_patch = patches.Rectangle(
            (d.breakdown_start, d.machine_id - 0.44),
            d.duration,
            0.88,
            linewidth=1.5,
            edgecolor="#9b0000",
            facecolor="#ff4d4d",
            hatch="//",
            alpha=0.68,
            zorder=5,
        )
        ax.add_patch(b_patch)
        ax.text(
            d.breakdown_start + d.duration / 2,
            d.machine_id,
            f"EVENT #{idx}: M{d.machine_id} DOWN\n[{d.breakdown_start}, {d.breakdown_end}]",
            ha="center",
            va="center",
            color="white",
            fontsize=7.0,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.80),
            zorder=6,
        )

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(num_machines)], fontweight="bold")
    ax.set_xlabel("Time (Rolling Shop-Floor Horizon)", fontweight="bold", fontsize=10)
    ax.set_ylabel("Production Machines", fontweight="bold", fontsize=10)
    ax.set_title(
        f"Rolling-Horizon U-NSGA-III Schedule Under Cascading Breakdowns + Rush Order J{rush_job_id} (Gold)\n"
        f"Makespan: {schedule.makespan}  |  Total Tardiness: {schedule.total_tardiness}  |  "
        f"Rush Job Completion: {schedule.job_completion_times[rush_job_id]} (Due: {schedule.instance.jobs[rush_job_id].due_date})  |  "
        f"Energy: {schedule.total_energy_consumption:.1f} kWh",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlim(0, schedule.makespan + 4)
    ax.set_ylim(-0.6, num_machines - 0.4)
    ax.grid(axis="x", linestyle="--", alpha=0.45)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    print("=" * 92)
    print(" ROLLING-HORIZON EXPERIMENT: CASCADING BREAKDOWNS + DYNAMIC RUSH-ORDER ARRIVAL")
    print("=" * 92)

    inst = load_jsp_instance("data/benchmarks/ft06.txt", tightness_factor=1.5)

    # Stage 0: Predictive Baseline (t = 0)
    sched_s0, _, _ = run_genetic_algorithm(inst, pop_size=50, generations=80, seed=42)
    assert sched_s0.validate()

    # Event 1 (t = 16): Breakdown #1 on Machine 2 for 14 units [16, 30]
    d1 = BreakdownDisruption(machine_id=2, breakdown_start=16, duration=14)
    state1 = build_disrupted_state(sched_s0, d1, inst)
    scheds_s1, objs_s1 = run_nsga3(
        inst, baseline_schedule=sched_s0, pop_size=56, generations=60,
        num_divisions=5, disrupted_state=state1, include_energy=True, seed=42,
    )
    sched_s1, _ = select_knee_schedule(scheds_s1, objs_s1)
    assert sched_s1.validate()

    # Event 2 (t = 34): Breakdown #2 on Machine 4 for 14 units [34, 48] + Rush Job J6 (due = 55)
    d2 = BreakdownDisruption(machine_id=4, breakdown_start=34, duration=14)
    aug_inst, state2 = build_stage2_with_rush_job(sched_s1, d1, d2, inst)
    rush_id = aug_inst.num_jobs - 1

    # Baselines at Event 2
    sched_rs_fifo = apply_right_shift_plus_fifo_rush(sched_s1, aug_inst, state2, d2)
    sched_r_fcfs = solve_fcfs(aug_inst, disrupted_state=state2)
    assert sched_rs_fifo.validate() and sched_r_fcfs.validate()

    # Rolling-Horizon U-NSGA-III at Event 2
    scheds_s2, objs_s2 = run_nsga3(
        aug_inst, baseline_schedule=sched_s1, pop_size=56, generations=65,
        num_divisions=5, disrupted_state=state2, include_energy=True, warm_start=False, seed=42,
    )
    for sc in scheds_s2:
        assert sc.validate()
    sched_s2_knee, _ = select_knee_schedule(scheds_s2, objs_s2)

    plot_multi_event_gantt(
        sched_s2_knee,
        disruptions=[d1, d2],
        rush_job_id=rush_id,
        save_path="results/09_multi_disruption_rush_order_gantt.png",
    )

    rush_due = aug_inst.jobs[rush_id].due_date
    rows = [
        {
            "Stage": "Stage 0 (t=0, 6 Jobs)",
            "Method": "Predictive Memetic GA",
            "Makespan": sched_s0.makespan,
            "Total_Tardiness": sched_s0.total_tardiness,
            "Rush_Job_Tardiness": "-",
            "Instability_vs_Prev": 0,
            "Energy_kWh": sched_s0.total_energy_consumption,
        },
        {
            "Stage": "Event 1 (t=16, M2 Down [16,30])",
            "Method": "Rolling U-NSGA-III (Knee)",
            "Makespan": sched_s1.makespan,
            "Total_Tardiness": sched_s1.total_tardiness,
            "Rush_Job_Tardiness": "-",
            "Instability_vs_Prev": sched_s1.compute_instability(sched_s0, 16),
            "Energy_kWh": sched_s1.total_energy_consumption,
        },
        {
            "Stage": "Event 2 (t=34, M4 Down + Rush J6)",
            "Method": "Right-Shift + FIFO Rush",
            "Makespan": sched_rs_fifo.makespan,
            "Total_Tardiness": sched_rs_fifo.total_tardiness,
            "Rush_Job_Tardiness": max(0, sched_rs_fifo.job_completion_times[rush_id] - rush_due),
            "Instability_vs_Prev": sched_rs_fifo.compute_instability(sched_s1, 34),
            "Energy_kWh": sched_rs_fifo.total_energy_consumption,
        },
        {
            "Stage": "Event 2 (t=34, M4 Down + Rush J6)",
            "Method": "Reactive FCFS",
            "Makespan": sched_r_fcfs.makespan,
            "Total_Tardiness": sched_r_fcfs.total_tardiness,
            "Rush_Job_Tardiness": max(0, sched_r_fcfs.job_completion_times[rush_id] - rush_due),
            "Instability_vs_Prev": sched_r_fcfs.compute_instability(sched_s1, 34),
            "Energy_kWh": sched_r_fcfs.total_energy_consumption,
        },
        {
            "Stage": "Event 2 (t=34, M4 Down + Rush J6)",
            "Method": "Rolling U-NSGA-III (Knee)",
            "Makespan": sched_s2_knee.makespan,
            "Total_Tardiness": sched_s2_knee.total_tardiness,
            "Rush_Job_Tardiness": max(0, sched_s2_knee.job_completion_times[rush_id] - rush_due),
            "Instability_vs_Prev": sched_s2_knee.compute_instability(sched_s1, 34),
            "Energy_kWh": sched_s2_knee.total_energy_consumption,
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv("results/dynamic_horizon_summary.csv", index=False)
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    print("\n[DONE] Saved `results/dynamic_horizon_summary.csv` and `results/09_multi_disruption_rush_order_gantt.png`.")


if __name__ == "__main__":
    main()

