"""
run_nsga3_comparison.py: Rigorous Apples-to-Apples Many-Objective Comparison
of NSGA-II (4 Objectives) vs. NSGA-III (4 Objectives) across FT06, LA01, and FT10.

Both MOEAs optimize the EXACT SAME 4-objective Dynamic Green Scheduling formulation:
  f1: Makespan (C_max)
  f2: Total Tardiness (sum T_j)
  f3: Schedule Instability (Delta S)
  f4: Total Energy Consumption (TEC in kWh)

Under identical population sizes (N = H = 56), generation budgets, genetic operators,
and random seeds, evaluated via:
  - Exact 4D Normalized Hypervolume (HV, higher is better)
  - Inverted Generational Distance (IGD to empirical reference front P*, lower is better)
  - Schott's Spacing Metric (SP, lower means more uniform front distribution)
  - Two-sided Wilcoxon Rank-Sum statistical significance tests (alpha = 0.05)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.problem import load_jsp_instance
from src.disruption import BreakdownDisruption, build_disrupted_state
from src.ga import run_genetic_algorithm
from src.nsga2 import run_nsga2
from src.nsga3 import run_nsga3
from src.metrics import (
    extract_nondominated,
    compute_hypervolume,
    compute_igd,
    compute_spacing,
    wilcoxon_rank_sum,
)
from src.visualize import plot_parallel_coordinates_4d, plot_hv_igd_boxplots

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


SCENARIOS = [
    {
        "label": "FT06 (6x6)",
        "path": "data/benchmarks/ft06.txt",
        "tightness": 1.5,
        "disruption_frac": 0.35,
        "duration_frac": 0.28,
        "ga_gen": 80,
        "moea_gen": 65,
    },
    {
        "label": "LA01 (10x5)",
        "path": "data/benchmarks/la01.txt",
        "tightness": 1.5,
        "disruption_frac": 0.35,
        "duration_frac": 0.25,
        "ga_gen": 100,
        "moea_gen": 70,
    },
    {
        "label": "FT10 (10x10)",
        "path": "data/benchmarks/ft10.txt",
        "tightness": 1.5,
        "disruption_frac": 0.35,
        "duration_frac": 0.25,
        "ga_gen": 120,
        "moea_gen": 75,
    },
]


def evaluate_4d_scenario(cfg: dict, num_seeds: int = 20, pop_size: int = 56, num_divisions: int = 5):
    instance = load_jsp_instance(cfg["path"], tightness_factor=cfg["tightness"])
    init_sched, _, _ = run_genetic_algorithm(
        instance, pop_size=60, generations=cfg["ga_gen"], seed=42
    )
    assert init_sched.validate()

    m_break = instance.get_bottleneck_machine()
    t_d = int(round(init_sched.makespan * cfg["disruption_frac"]))
    dur = int(round(init_sched.makespan * cfg["duration_frac"]))
    disruption = BreakdownDisruption(machine_id=m_break, breakdown_start=t_d, duration=dur)
    state = build_disrupted_state(init_sched, disruption, instance)

    print(f"\n[{cfg['label']}] Predictive C_max={init_sched.makespan} | {disruption} | Rem Ops={len(state.remaining_chromosome_pool)}")

    nsga2_fronts = []
    nsga3_fronts = []
    nsga2_times = []
    nsga3_times = []

    for s in range(num_seeds):
        run_seed = 1000 + s * 37

        t0 = time.process_time()
        scheds2, objs2 = run_nsga2(
            instance,
            baseline_schedule=init_sched,
            pop_size=pop_size,
            generations=cfg["moea_gen"],
            disrupted_state=state,
            include_energy=True,
            warm_start=True,
            seed=run_seed,
        )
        nsga2_times.append(time.process_time() - t0)
        assert all(sc.validate() for sc in scheds2)
        nsga2_fronts.append(objs2)

        t0 = time.process_time()
        scheds3, objs3 = run_nsga3(
            instance,
            baseline_schedule=init_sched,
            pop_size=pop_size,
            generations=cfg["moea_gen"],
            num_divisions=num_divisions,
            disrupted_state=state,
            include_energy=True,
            warm_start=True,
            seed=run_seed,
        )
        nsga3_times.append(time.process_time() - t0)
        assert all(sc.validate() for sc in scheds3)
        nsga3_fronts.append(objs3)

    # Construct empirical true Pareto front P* from union of all 2 * num_seeds fronts
    all_pts = np.vstack(nsga2_fronts + nsga3_fronts)
    ref_front = extract_nondominated(all_pts)
    ideal = np.min(all_pts, axis=0)
    nadir = np.max(ref_front, axis=0)
    # Ensure non-degenerate nadir if P* has identical values on an axis
    nadir = np.maximum(nadir, ideal + 1e-6)

    records = []
    hv2_list, hv3_list = [], []
    igd2_list, igd3_list = [], []
    sp2_list, sp3_list = [], []

    for s in range(num_seeds):
        f2 = nsga2_fronts[s]
        f3 = nsga3_fronts[s]

        hv2 = compute_hypervolume(f2, ideal, nadir)
        hv3 = compute_hypervolume(f3, ideal, nadir)
        igd2 = compute_igd(f2, ref_front, ideal, nadir)
        igd3 = compute_igd(f3, ref_front, ideal, nadir)
        sp2 = compute_spacing(f2, ideal, nadir)
        sp3 = compute_spacing(f3, ideal, nadir)

        hv2_list.append(hv2)
        hv3_list.append(hv3)
        igd2_list.append(igd2)
        igd3_list.append(igd3)
        sp2_list.append(sp2)
        sp3_list.append(sp3)

        records.append({
            "Instance": cfg["label"], "Algorithm": "NSGA-II (4D)", "Seed": s,
            "Front_Size": len(f2), "HV": hv2, "IGD": igd2, "Spacing": sp2,
            "Min_Makespan": float(np.min(f2[:, 0])), "Min_Tardiness": float(np.min(f2[:, 1])),
            "Min_Instability": float(np.min(f2[:, 2])), "Min_Energy_kWh": float(np.min(f2[:, 3])),
            "Runtime_sec": nsga2_times[s],
        })
        records.append({
            "Instance": cfg["label"], "Algorithm": "NSGA-III (4D)", "Seed": s,
            "Front_Size": len(f3), "HV": hv3, "IGD": igd3, "Spacing": sp3,
            "Min_Makespan": float(np.min(f3[:, 0])), "Min_Tardiness": float(np.min(f3[:, 1])),
            "Min_Instability": float(np.min(f3[:, 2])), "Min_Energy_kWh": float(np.min(f3[:, 3])),
            "Runtime_sec": nsga3_times[s],
        })

    stat_hv = wilcoxon_rank_sum(hv3_list, hv2_list, higher_is_better=True)
    stat_igd = wilcoxon_rank_sum(igd3_list, igd2_list, higher_is_better=False)
    stat_sp = wilcoxon_rank_sum(sp3_list, sp2_list, higher_is_better=False)

    return {
        "records": records,
        "ref_size": len(ref_front),
        "hv2": hv2_list, "hv3": hv3_list,
        "igd2": igd2_list, "igd3": igd3_list,
        "sp2": sp2_list, "sp3": sp3_list,
        "t2": nsga2_times, "t3": nsga3_times,
        "stat_hv": stat_hv, "stat_igd": stat_igd, "stat_sp": stat_sp,
        "sample_f2": nsga2_fronts[0],
        "sample_f3": nsga3_fronts[0],
    }


def main():
    print("=" * 96)
    print(" APPLES-TO-APPLES 4-OBJECTIVE STUDY: NSGA-II (4D) vs. NSGA-III (4D) [20 SEEDS PER BENCHMARK]")
    print(" Objectives: f1=Makespan, f2=Tardiness, f3=Instability (ΔS), f4=Total Energy (kWh)")
    print(" Budget    : pop_size = 56, Das-Dennis Reference Rays H = binom(4+5-1, 5) = 56")
    print("=" * 96)

    os.makedirs("results", exist_ok=True)
    num_seeds = 20
    all_records = []
    summary_rows = []

    for cfg in SCENARIOS:
        res = evaluate_4d_scenario(cfg, num_seeds=num_seeds, pop_size=56, num_divisions=5)
        all_records.extend(res["records"])

        summary_rows.append([
            cfg["label"],
            "NSGA-II (4D)",
            f"{np.mean(res['hv2']):.4f} ± {np.std(res['hv2']):.4f}",
            f"{np.mean(res['igd2']):.4f} ± {np.std(res['igd2']):.4f}",
            f"{np.mean(res['sp2']):.4f} ± {np.std(res['sp2']):.4f}",
            f"{np.mean(res['t2']):.2f} s",
            "-",
        ])
        sig_str = (
            f"HV:{res['stat_hv']['symbol']} (p={res['stat_hv']['p_value']:.3g}), "
            f"IGD:{res['stat_igd']['symbol']} (p={res['stat_igd']['p_value']:.3g})"
        )
        summary_rows.append([
            cfg["label"],
            "NSGA-III (4D)",
            f"{np.mean(res['hv3']):.4f} ± {np.std(res['hv3']):.4f}",
            f"{np.mean(res['igd3']):.4f} ± {np.std(res['igd3']):.4f}",
            f"{np.mean(res['sp3']):.4f} ± {np.std(res['sp3']):.4f}",
            f"{np.mean(res['t3']):.2f} s",
            sig_str,
        ])

        if "FT10" in cfg["label"]:
            plot_parallel_coordinates_4d(
                res["sample_f2"],
                res["sample_f3"],
                save_path="results/06_nsga2_vs_nsga3_4d_parallel.png",
                title="FT10 (10x10) 4-Objective Pareto Front Coverage: NSGA-II vs NSGA-III",
            )
            plot_hv_igd_boxplots(
                res["hv2"],
                res["hv3"],
                res["igd2"],
                res["igd3"],
                p_hv=res["stat_hv"]["p_value"],
                p_igd=res["stat_igd"]["p_value"],
                save_path="results/07_nsga2_vs_nsga3_hv_igd_boxplot.png",
                instance_label=cfg["label"],
            )

    df = pd.DataFrame(all_records)
    csv_path = "results/nsga2_vs_nsga3_4d_metrics.csv"
    df.to_csv(csv_path, index=False)

    headers = [
        "Benchmark",
        "Algorithm",
        "Hypervolume (↑)",
        "IGD (↓)",
        "Spacing (↓)",
        "Runtime",
        "Wilcoxon vs NSGA-II",
    ]
    print("\n" + "=" * 96)
    print(" 4-OBJECTIVE STATISTICAL SUMMARY (MEAN ± STD OVER 20 INDEPENDENT SEEDS)")
    print("=" * 96)
    print(tabulate(summary_rows, headers=headers, tablefmt="github"))
    print("=" * 96)
    print(f"\nSaved metrics CSV to `{csv_path}`")
    print("Saved plots to `results/06_nsga2_vs_nsga3_4d_parallel.png` & `results/07_nsga2_vs_nsga3_hv_igd_boxplot.png`.")


if __name__ == "__main__":
    main()
