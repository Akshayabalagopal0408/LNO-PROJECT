"""
run_experiments.py: Comprehensive Multi-Seed, Multi-Benchmark Experimental Suite.
Evaluates Classical Dispatching Rules (FCFS, SPT, EDD), Single-Objective GA,
Right-Shift Rescheduling, Reactive FCFS/SPT/GA, NSGA-II (3D & 4D), and NSGA-III (4D)
across FT06 (6x6), LA01 (10x5), LA02 (10x5), and FT10 (10x10) over 20 independent seeds.

Outputs publication-ready CSV tables with Mean ± Std, Hypervolume, IGD, Sequence Swaps,
and Wilcoxon Rank-Sum p-values to `results/`.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from tabulate import tabulate

# Ensure root workspace path is importable when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from src.nsga3 import run_nsga3
from src.metrics import (
    extract_nondominated,
    compute_hypervolume,
    compute_igd,
    wilcoxon_rank_sum,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


BENCHMARK_SUITE = [
    {
        "name": "FT06 (6x6)",
        "path": "data/benchmarks/ft06.txt",
        "disruption_machine": 2,
        "disruption_time": 20,
        "disruption_duration": 15,
        "ga_pop": 50,
        "ga_gen": 80,
        "moea_pop": 56,
        "moea_gen": 65,
    },
    {
        "name": "LA01 (10x5)",
        "path": "data/benchmarks/la01.txt",
        "disruption_machine": None,  # Auto-select bottleneck machine
        "disruption_time": 230,
        "disruption_duration": 160,
        "ga_pop": 55,
        "ga_gen": 90,
        "moea_pop": 56,
        "moea_gen": 65,
    },
    {
        "name": "LA02 (10x5)",
        "path": "data/benchmarks/la02.txt",
        "disruption_machine": None,
        "disruption_time": 230,
        "disruption_duration": 160,
        "ga_pop": 55,
        "ga_gen": 90,
        "moea_pop": 56,
        "moea_gen": 65,
    },
    {
        "name": "FT10 (10x10)",
        "path": "data/benchmarks/ft10.txt",
        "disruption_machine": None,
        "disruption_time": 340,
        "disruption_duration": 220,
        "ga_pop": 60,
        "ga_gen": 100,
        "moea_pop": 56,
        "moea_gen": 70,
    },
]


def evaluate_instance_experiment(cfg: dict, num_seeds: int = 20):
    benchmark_name = cfg["name"]
    instance = load_jsp_instance(cfg["path"], tightness_factor=1.5)
    m_break = cfg["disruption_machine"] if cfg["disruption_machine"] is not None else instance.get_bottleneck_machine()

    disruption = BreakdownDisruption(
        machine_id=m_break,
        breakdown_start=cfg["disruption_time"],
        duration=cfg["disruption_duration"],
    )

    print(f"\n=======================================================================")
    print(f" BENCHMARK: {benchmark_name} | Disruption: {disruption}")
    print(f"=======================================================================")

    records = []

    # 1. Deterministic Pre-Disruption Baselines
    for rule_name, solver in [("FCFS", solve_fcfs), ("SPT", solve_spt), ("EDD", solve_edd)]:
        t0 = time.process_time()
        sched = solver(instance)
        rt = time.process_time() - t0
        assert sched.validate()
        records.append({
            "Instance": benchmark_name,
            "Algorithm": rule_name,
            "Phase": "Pre-Disruption",
            "Seed": 0,
            "Makespan": sched.makespan,
            "Tardiness": sched.total_tardiness,
            "Instability": 0,
            "Seq_Swaps": 0,
            "Energy_kWh": sched.total_energy_consumption,
            "Runtime_sec": rt,
        })

    # Fixed predictive baseline schedule (seed=42) so all post-disruption seeds share a common disruption state
    init_ga_fixed, _, _ = run_genetic_algorithm(
        instance, pop_size=cfg["ga_pop"], generations=cfg["ga_gen"], seed=42
    )
    assert init_ga_fixed.validate()
    state = build_disrupted_state(init_ga_fixed, disruption, instance)
    ws_chroms = extract_warm_start_chromosomes(init_ga_fixed, disruption, instance)

    # Deterministic Post-Disruption Baselines on Fixed Predictive Schedule
    t0 = time.process_time()
    sched_rs = apply_right_shift(init_ga_fixed, disruption, instance)
    rs_rt = time.process_time() - t0
    assert sched_rs.validate()
    records.append({
        "Instance": benchmark_name,
        "Algorithm": "Right-Shift",
        "Phase": "Post-Disruption",
        "Seed": 0,
        "Makespan": sched_rs.makespan,
        "Tardiness": sched_rs.total_tardiness,
        "Instability": sched_rs.compute_instability(init_ga_fixed, disruption.breakdown_start),
        "Seq_Swaps": sched_rs.compute_sequence_instability(init_ga_fixed),
        "Energy_kWh": sched_rs.total_energy_consumption,
        "Runtime_sec": rs_rt,
    })

    for r_name, solver in [("Reactive FCFS", solve_fcfs), ("Reactive SPT", solve_spt)]:
        t0 = time.process_time()
        r_sched = solver(instance, disrupted_state=state)
        r_rt = time.process_time() - t0
        assert r_sched.validate()
        records.append({
            "Instance": benchmark_name,
            "Algorithm": r_name,
            "Phase": "Post-Disruption",
            "Seed": 0,
            "Makespan": r_sched.makespan,
            "Tardiness": r_sched.total_tardiness,
            "Instability": r_sched.compute_instability(init_ga_fixed, disruption.breakdown_start),
            "Seq_Swaps": r_sched.compute_sequence_instability(init_ga_fixed),
            "Energy_kWh": r_sched.total_energy_consumption,
            "Runtime_sec": r_rt,
        })

    # 2. Stochastic Evolutionary Runs across `num_seeds`
    nsga2_4d_fronts = []
    nsga3_4d_fronts = []

    for seed in range(num_seeds):
        run_seed = seed * 31 + 11

        # A. Pre-Disruption Single-Objective GA
        t0 = time.process_time()
        ga_pre, _, _ = run_genetic_algorithm(
            instance, pop_size=cfg["ga_pop"], generations=cfg["ga_gen"], seed=run_seed
        )
        ga_pre_rt = time.process_time() - t0
        assert ga_pre.validate()
        records.append({
            "Instance": benchmark_name,
            "Algorithm": "Single-Obj GA",
            "Phase": "Pre-Disruption",
            "Seed": seed,
            "Makespan": ga_pre.makespan,
            "Tardiness": ga_pre.total_tardiness,
            "Instability": 0,
            "Seq_Swaps": 0,
            "Energy_kWh": ga_pre.total_energy_consumption,
            "Runtime_sec": ga_pre_rt,
        })

        # B. Post-Disruption Reactive Single-Obj GA
        t0 = time.process_time()
        r_ga, _, _ = run_genetic_algorithm(
            instance,
            pop_size=cfg["ga_pop"],
            generations=cfg["moea_gen"],
            disrupted_state=state,
            warm_start_chromosomes=ws_chroms,
            seed=run_seed,
        )
        r_ga_rt = time.process_time() - t0
        assert r_ga.validate()
        records.append({
            "Instance": benchmark_name,
            "Algorithm": "Reactive GA",
            "Phase": "Post-Disruption",
            "Seed": seed,
            "Makespan": r_ga.makespan,
            "Tardiness": r_ga.total_tardiness,
            "Instability": r_ga.compute_instability(init_ga_fixed, disruption.breakdown_start),
            "Seq_Swaps": r_ga.compute_sequence_instability(init_ga_fixed),
            "Energy_kWh": r_ga.total_energy_consumption,
            "Runtime_sec": r_ga_rt,
        })

        # C. NSGA-II (3 Objectives)
        t0 = time.process_time()
        scheds_3d, objs_3d = run_nsga2(
            instance,
            baseline_schedule=init_ga_fixed,
            pop_size=cfg["moea_pop"],
            generations=cfg["moea_gen"],
            disrupted_state=state,
            include_energy=False,
            warm_start=True,
            seed=run_seed,
        )
        nsga2_3d_rt = time.process_time() - t0

        idx_ms = int(np.argmin(objs_3d[:, 0]))
        idx_st = int(np.argmin(objs_3d[:, 2]))
        # Normalized knee point (closest to ideal point in 3D)
        span_3d = np.where(np.ptp(objs_3d, axis=0) > 0, np.ptp(objs_3d, axis=0), 1.0)
        norm_3d = (objs_3d - np.min(objs_3d, axis=0)) / span_3d
        idx_knee = int(np.argmin(np.linalg.norm(norm_3d, axis=1)))

        for label, idx in [
            ("NSGA-II 3D (Min Makespan)", idx_ms),
            ("NSGA-II 3D (Knee Compromise)", idx_knee),
            ("NSGA-II 3D (Min Instability)", idx_st),
        ]:
            sol = scheds_3d[idx]
            records.append({
                "Instance": benchmark_name,
                "Algorithm": label,
                "Phase": "Post-Disruption",
                "Seed": seed,
                "Makespan": objs_3d[idx, 0],
                "Tardiness": objs_3d[idx, 1],
                "Instability": objs_3d[idx, 2],
                "Seq_Swaps": sol.compute_sequence_instability(init_ga_fixed),
                "Energy_kWh": sol.total_energy_consumption,
                "Runtime_sec": nsga2_3d_rt,
            })

        # D. NSGA-II (4 Objectives) vs NSGA-III (4 Objectives)
        t0 = time.process_time()
        scheds_2_4d, objs_2_4d = run_nsga2(
            instance,
            baseline_schedule=init_ga_fixed,
            pop_size=cfg["moea_pop"],
            generations=cfg["moea_gen"],
            disrupted_state=state,
            include_energy=True,
            warm_start=True,
            seed=run_seed,
        )
        rt_2_4d = time.process_time() - t0
        nsga2_4d_fronts.append(objs_2_4d)

        t0 = time.process_time()
        scheds_3_4d, objs_3_4d = run_nsga3(
            instance,
            baseline_schedule=init_ga_fixed,
            pop_size=cfg["moea_pop"],
            generations=cfg["moea_gen"],
            num_divisions=5,
            disrupted_state=state,
            include_energy=True,
            warm_start=True,
            seed=run_seed,
        )
        rt_3_4d = time.process_time() - t0
        nsga3_4d_fronts.append(objs_3_4d)

        for alg_label, sc_list, ob_arr, rt_val in [
            ("NSGA-II 4D (Knee)", scheds_2_4d, objs_2_4d, rt_2_4d),
            ("NSGA-III 4D (Knee)", scheds_3_4d, objs_3_4d, rt_3_4d),
        ]:
            sp4 = np.where(np.ptp(ob_arr, axis=0) > 0, np.ptp(ob_arr, axis=0), 1.0)
            n4 = (ob_arr - np.min(ob_arr, axis=0)) / sp4
            ik = int(np.argmin(np.linalg.norm(n4, axis=1)))
            records.append({
                "Instance": benchmark_name,
                "Algorithm": alg_label,
                "Phase": "Post-Disruption (4D)",
                "Seed": seed,
                "Makespan": ob_arr[ik, 0],
                "Tardiness": ob_arr[ik, 1],
                "Instability": ob_arr[ik, 2],
                "Seq_Swaps": sc_list[ik].compute_sequence_instability(init_ga_fixed),
                "Energy_kWh": ob_arr[ik, 3],
                "Runtime_sec": rt_val,
            })

    # Compute 4D HV and IGD for this benchmark
    all_4d = np.vstack(nsga2_4d_fronts + nsga3_4d_fronts)
    ref_4d = extract_nondominated(all_4d)
    ideal_4d = np.min(all_4d, axis=0)
    nadir_4d = np.maximum(np.max(ref_4d, axis=0), ideal_4d + 1e-6)

    hv2 = [compute_hypervolume(f, ideal_4d, nadir_4d) for f in nsga2_4d_fronts]
    hv3 = [compute_hypervolume(f, ideal_4d, nadir_4d) for f in nsga3_4d_fronts]
    igd2 = [compute_igd(f, ref_4d, ideal_4d, nadir_4d) for f in nsga2_4d_fronts]
    igd3 = [compute_igd(f, ref_4d, ideal_4d, nadir_4d) for f in nsga3_4d_fronts]

    w_hv = wilcoxon_rank_sum(hv3, hv2, higher_is_better=True)
    w_igd = wilcoxon_rank_sum(igd3, igd2, higher_is_better=False)

    moea_meta = {
        "Instance": benchmark_name,
        "Ref_Front_Size": len(ref_4d),
        "NSGA2_HV_Mean": float(np.mean(hv2)),
        "NSGA2_HV_Std": float(np.std(hv2)),
        "NSGA3_HV_Mean": float(np.mean(hv3)),
        "NSGA3_HV_Std": float(np.std(hv3)),
        "HV_Wilcoxon_p": w_hv["p_value"],
        "HV_Sig": w_hv["symbol"],
        "NSGA2_IGD_Mean": float(np.mean(igd2)),
        "NSGA2_IGD_Std": float(np.std(igd2)),
        "NSGA3_IGD_Mean": float(np.mean(igd3)),
        "NSGA3_IGD_Std": float(np.std(igd3)),
        "IGD_Wilcoxon_p": w_igd["p_value"],
        "IGD_Sig": w_igd["symbol"],
    }

    return pd.DataFrame(records), moea_meta


def main():
    os.makedirs("results", exist_ok=True)
    num_seeds = 20

    dfs = []
    moea_summaries = []

    for cfg in BENCHMARK_SUITE:
        df_inst, meta = evaluate_instance_experiment(cfg, num_seeds=num_seeds)
        dfs.append(df_inst)
        moea_summaries.append(meta)

        if "FT06" in cfg["name"]:
            df_inst.to_csv("results/experiment_summary_ft06.csv", index=False)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv("results/experiment_detailed_all.csv", index=False)

    agg = (
        full_df.groupby(["Instance", "Phase", "Algorithm"], sort=False)
        .agg(
            Makespan_Mean=("Makespan", "mean"),
            Makespan_Std=("Makespan", "std"),
            Tardiness_Mean=("Tardiness", "mean"),
            Tardiness_Std=("Tardiness", "std"),
            Instability_Mean=("Instability", "mean"),
            Seq_Swaps_Mean=("Seq_Swaps", "mean"),
            Energy_kWh_Mean=("Energy_kWh", "mean"),
            Runtime_Mean=("Runtime_sec", "mean"),
        )
        .fillna(0.0)
        .reset_index()
    )
    agg.to_csv("results/experiment_summary_all.csv", index=False)

    moea_df = pd.DataFrame(moea_summaries)
    moea_df.to_csv("results/experiment_4d_hv_igd_summary.csv", index=False)

    print("\n" + "=" * 105)
    print(" AGGREGATED MULTI-BENCHMARK SUMMARY (20 INDEPENDENT SEEDS PER STOCHASTIC METHOD)")
    print("=" * 105)
    print(agg.to_string(index=False))

    print("\n" + "=" * 105)
    print(" 4D MANY-OBJECTIVE STATISTICAL INDICATORS (NSGA-II vs NSGA-III)")
    print("=" * 105)
    print(tabulate(moea_df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    print("\n[DONE] Saved all CSV tables to `results/`.")


if __name__ == "__main__":
    main()
