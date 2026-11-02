"""
run_ablation_sensitivity.py: Ablation & Parameter Sensitivity Suite for Final Paper.

Executes four systematic studies over 15 independent random seeds:
  1. Warm-Start Ablation: `warm_start=True` vs `warm_start=False` on LA01 & FT10 (4D NSGA-III)
  2. Memetic Local Search Ablation: `local_search=True` (N5) vs `local_search=False` on FT10
  3. Disruption Timing Sensitivity: Early (20%), Mid-Peak (45%), and Late (70%) breakdowns on LA01
  4. Due-Date Tightness Sensitivity: Tight (k=1.3), Moderate (k=1.5), and Loose (k=1.7) TWK on LA01

Outputs CSVs (`results/ablation_study.csv`, `results/sensitivity_study.csv`) and a 2x2
publication figure (`results/08_ablation_and_sensitivity.png`).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem import load_jsp_instance
from src.disruption import BreakdownDisruption, build_disrupted_state, apply_right_shift
from src.ga import run_genetic_algorithm
from src.nsga3 import run_nsga3
from src.metrics import extract_nondominated, compute_hypervolume, compute_igd, wilcoxon_rank_sum

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def run_ablation_experiments(num_seeds: int = 15):
    records = []
    plot_data = {}

    # --- Study 1: Warm-Start vs Cold-Start on LA01 and FT10 ---
    for label, path, ga_g, moea_g in [
        ("LA01 (10x5)", "data/benchmarks/la01.txt", 90, 55),
        ("FT10 (10x10)", "data/benchmarks/ft10.txt", 100, 60),
    ]:
        inst = load_jsp_instance(path, tightness_factor=1.5)
        init_sched, _, _ = run_genetic_algorithm(inst, pop_size=55, generations=ga_g, seed=42)
        m_b = inst.get_bottleneck_machine()
        disr = BreakdownDisruption(
            machine_id=m_b,
            breakdown_start=int(round(0.35 * init_sched.makespan)),
            duration=int(round(0.25 * init_sched.makespan)),
        )
        state = build_disrupted_state(init_sched, disr, inst)

        warm_fronts, cold_fronts = [], []
        for s in range(num_seeds):
            rseed = 500 + s * 29
            _, f_warm = run_nsga3(
                inst, baseline_schedule=init_sched, pop_size=56, generations=moea_g,
                num_divisions=5, disrupted_state=state, include_energy=True,
                warm_start=True, local_search=True, seed=rseed,
            )
            _, f_cold = run_nsga3(
                inst, baseline_schedule=init_sched, pop_size=56, generations=moea_g,
                num_divisions=5, disrupted_state=state, include_energy=True,
                warm_start=False, local_search=True, seed=rseed,
            )
            warm_fronts.append(f_warm)
            cold_fronts.append(f_cold)

        all_pts = np.vstack(warm_fronts + cold_fronts)
        ref = extract_nondominated(all_pts)
        ideal = np.min(all_pts, axis=0)
        nadir = np.maximum(np.max(ref, axis=0), ideal + 1e-6)

        hv_w = [compute_hypervolume(f, ideal, nadir) for f in warm_fronts]
        hv_c = [compute_hypervolume(f, ideal, nadir) for f in cold_fronts]
        igd_w = [compute_igd(f, ref, ideal, nadir) for f in warm_fronts]
        igd_c = [compute_igd(f, ref, ideal, nadir) for f in cold_fronts]
        min_ds_w = [float(np.min(f[:, 2])) for f in warm_fronts]
        min_ds_c = [float(np.min(f[:, 2])) for f in cold_fronts]

        stat_hv = wilcoxon_rank_sum(hv_w, hv_c, higher_is_better=True)
        stat_ds = wilcoxon_rank_sum(min_ds_w, min_ds_c, higher_is_better=False)

        records.append({
            "Study": "Warm-Start Ablation",
            "Instance": label,
            "Variant": "Cold-Start (Random P0)",
            "HV_Mean": np.mean(hv_c), "HV_Std": np.std(hv_c),
            "IGD_Mean": np.mean(igd_c),
            "Best_Metric_Mean": np.mean(min_ds_c),
            "Metric_Label": "Min ΔS",
            "Wilcoxon_p": "-",
        })
        records.append({
            "Study": "Warm-Start Ablation",
            "Instance": label,
            "Variant": "Warm-Start (Proposed)",
            "HV_Mean": np.mean(hv_w), "HV_Std": np.std(hv_w),
            "IGD_Mean": np.mean(igd_w),
            "Best_Metric_Mean": np.mean(min_ds_w),
            "Metric_Label": "Min ΔS",
            "Wilcoxon_p": f"HV p={stat_hv['p_value']:.3g} ({stat_hv['symbol']}), ΔS p={stat_ds['p_value']:.3g} ({stat_ds['symbol']})",
        })

        if "LA01" in label:
            plot_data["ws_ds"] = (min_ds_c, min_ds_w, stat_ds["p_value"], hv_c, hv_w, stat_hv["p_value"])

    # --- Study 2: Memetic N5 Local Search Ablation on FT10 ---
    inst_ft10 = load_jsp_instance("data/benchmarks/ft10.txt", tightness_factor=1.5)
    init_ft10, _, _ = run_genetic_algorithm(inst_ft10, pop_size=55, generations=100, local_search=True, seed=42)
    disr_ft10 = BreakdownDisruption(
        machine_id=inst_ft10.get_bottleneck_machine(),
        breakdown_start=int(round(0.35 * init_ft10.makespan)),
        duration=int(round(0.25 * init_ft10.makespan)),
    )
    state_ft10 = build_disrupted_state(init_ft10, disr_ft10, inst_ft10)

    mem_fronts, pure_fronts = [], []
    for s in range(num_seeds):
        rseed = 900 + s * 31
        _, f_mem = run_nsga3(
            inst_ft10, baseline_schedule=init_ft10, pop_size=56, generations=60,
            num_divisions=5, disrupted_state=state_ft10, include_energy=True,
            warm_start=True, local_search=True, seed=rseed,
        )
        _, f_pure = run_nsga3(
            inst_ft10, baseline_schedule=init_ft10, pop_size=56, generations=60,
            num_divisions=5, disrupted_state=state_ft10, include_energy=True,
            warm_start=True, local_search=False, seed=rseed,
        )
        mem_fronts.append(f_mem)
        pure_fronts.append(f_pure)

    all_mp = np.vstack(mem_fronts + pure_fronts)
    ref_mp = extract_nondominated(all_mp)
    id_mp = np.min(all_mp, axis=0)
    nad_mp = np.maximum(np.max(ref_mp, axis=0), id_mp + 1e-6)

    hv_m = [compute_hypervolume(f, id_mp, nad_mp) for f in mem_fronts]
    hv_p = [compute_hypervolume(f, id_mp, nad_mp) for f in pure_fronts]
    igd_m = [compute_igd(f, ref_mp, id_mp, nad_mp) for f in mem_fronts]
    igd_p = [compute_igd(f, ref_mp, id_mp, nad_mp) for f in pure_fronts]
    ms_m = [float(np.min(f[:, 0])) for f in mem_fronts]
    ms_p = [float(np.min(f[:, 0])) for f in pure_fronts]

    stat_mem_hv = wilcoxon_rank_sum(hv_m, hv_p, higher_is_better=True)
    stat_mem_ms = wilcoxon_rank_sum(ms_m, ms_p, higher_is_better=False)

    records.append({
        "Study": "Memetic N5 Ablation",
        "Instance": "FT10 (10x10)",
        "Variant": "Pure Genetic (No N5)",
        "HV_Mean": np.mean(hv_p), "HV_Std": np.std(hv_p),
        "IGD_Mean": np.mean(igd_p),
        "Best_Metric_Mean": np.mean(ms_p),
        "Metric_Label": "Min C_max",
        "Wilcoxon_p": "-",
    })
    records.append({
        "Study": "Memetic N5 Ablation",
        "Instance": "FT10 (10x10)",
        "Variant": "Memetic N5 (Proposed)",
        "HV_Mean": np.mean(hv_m), "HV_Std": np.std(hv_m),
        "IGD_Mean": np.mean(igd_m),
        "Best_Metric_Mean": np.mean(ms_m),
        "Metric_Label": "Min C_max",
        "Wilcoxon_p": f"HV p={stat_mem_hv['p_value']:.3g} ({stat_mem_hv['symbol']}), Cmax p={stat_mem_ms['p_value']:.3g} ({stat_mem_ms['symbol']})",
    })
    plot_data["n5_hv"] = (hv_p, hv_m, stat_mem_hv["p_value"])

    return pd.DataFrame(records), plot_data


def run_sensitivity_experiments(num_seeds: int = 15):
    sens_records = []

    # --- Sensitivity A: Disruption Timing (Early 20%, Mid 45%, Late 70%) on LA01 ---
    inst = load_jsp_instance("data/benchmarks/la01.txt", tightness_factor=1.5)
    init_sched, _, _ = run_genetic_algorithm(inst, pop_size=55, generations=90, seed=42)
    m_b = inst.get_bottleneck_machine()
    dur = int(round(0.25 * init_sched.makespan))

    timing_labels = []
    rs_ms_vals, moea_ms_vals = [], []
    rs_tard_vals, moea_tard_vals = [], []

    for frac_label, frac in [("Early (20%)", 0.20), ("Mid-Peak (45%)", 0.45), ("Late (70%)", 0.70)]:
        t_d = int(round(frac * init_sched.makespan))
        disr = BreakdownDisruption(machine_id=m_b, breakdown_start=t_d, duration=dur)
        state = build_disrupted_state(init_sched, disr, inst)
        rs_sched = apply_right_shift(init_sched, disr, inst)

        knee_ms, knee_td, knee_ds = [], [], []
        for s in range(num_seeds):
            _, objs = run_nsga3(
                inst, baseline_schedule=init_sched, pop_size=56, generations=55,
                num_divisions=5, disrupted_state=state, include_energy=True, seed=700 + s * 19,
            )
            sp = np.where(np.ptp(objs, axis=0) > 0, np.ptp(objs, axis=0), 1.0)
            ik = int(np.argmin(np.linalg.norm((objs - np.min(objs, axis=0)) / sp, axis=1)))
            knee_ms.append(objs[ik, 0])
            knee_td.append(objs[ik, 1])
            knee_ds.append(objs[ik, 2])

        sens_records.append({
            "Factor": "Breakdown Timing",
            "Setting": frac_label,
            "Rem_Ops": len(state.remaining_chromosome_pool),
            "RS_Makespan": rs_sched.makespan,
            "MOEA_Knee_Makespan": float(np.mean(knee_ms)),
            "RS_Tardiness": rs_sched.total_tardiness,
            "MOEA_Knee_Tardiness": float(np.mean(knee_td)),
            "Tardiness_Reduction_Pct": 100.0 * (rs_sched.total_tardiness - float(np.mean(knee_td))) / max(1.0, rs_sched.total_tardiness),
            "RS_Instability": rs_sched.compute_instability(init_sched, t_d),
            "MOEA_Knee_Instability": float(np.mean(knee_ds)),
        })
        timing_labels.append(frac_label)
        rs_tard_vals.append(rs_sched.total_tardiness)
        moea_tard_vals.append(float(np.mean(knee_td)))

    # --- Sensitivity B: Due-Date Tightness (k = 1.3, 1.5, 1.7) on LA01 ---
    k_labels, k_rs_td, k_moea_td = [], [], []
    for k_name, k_val in [("Tight (k=1.3)", 1.3), ("Moderate (k=1.5)", 1.5), ("Loose (k=1.7)", 1.7)]:
        inst_k = load_jsp_instance("data/benchmarks/la01.txt", tightness_factor=k_val)
        init_k, _, _ = run_genetic_algorithm(inst_k, pop_size=55, generations=90, seed=42)
        t_d = int(round(0.35 * init_k.makespan))
        disr_k = BreakdownDisruption(
            machine_id=inst_k.get_bottleneck_machine(),
            breakdown_start=t_d,
            duration=int(round(0.25 * init_k.makespan)),
        )
        state_k = build_disrupted_state(init_k, disr_k, inst_k)
        rs_k = apply_right_shift(init_k, disr_k, inst_k)

        knee_ms, knee_td, knee_ds = [], [], []
        for s in range(num_seeds):
            _, objs = run_nsga3(
                inst_k, baseline_schedule=init_k, pop_size=56, generations=55,
                num_divisions=5, disrupted_state=state_k, include_energy=True, seed=800 + s * 23,
            )
            sp = np.where(np.ptp(objs, axis=0) > 0, np.ptp(objs, axis=0), 1.0)
            ik = int(np.argmin(np.linalg.norm((objs - np.min(objs, axis=0)) / sp, axis=1)))
            knee_ms.append(objs[ik, 0])
            knee_td.append(objs[ik, 1])
            knee_ds.append(objs[ik, 2])

        sens_records.append({
            "Factor": "Due-Date Tightness",
            "Setting": k_name,
            "Rem_Ops": len(state_k.remaining_chromosome_pool),
            "RS_Makespan": rs_k.makespan,
            "MOEA_Knee_Makespan": float(np.mean(knee_ms)),
            "RS_Tardiness": rs_k.total_tardiness,
            "MOEA_Knee_Tardiness": float(np.mean(knee_td)),
            "Tardiness_Reduction_Pct": 100.0 * (rs_k.total_tardiness - float(np.mean(knee_td))) / max(1.0, rs_k.total_tardiness),
            "RS_Instability": rs_k.compute_instability(init_k, t_d),
            "MOEA_Knee_Instability": float(np.mean(knee_ds)),
        })
        k_labels.append(k_name)
        k_rs_td.append(rs_k.total_tardiness)
        k_moea_td.append(float(np.mean(knee_td)))

    sens_plot = {
        "timing": (timing_labels, rs_tard_vals, moea_tard_vals),
        "tightness": (k_labels, k_rs_td, k_moea_td),
    }
    return pd.DataFrame(sens_records), sens_plot


def save_combined_plot(ab_plot: dict, sens_plot: dict, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), dpi=180)

    # (0, 0) Warm-Start Min Instability / HV Boxplot
    ds_c, ds_w, p_ds, hv_c, hv_w, p_ws = ab_plot["ws_ds"]
    bp1 = axes[0, 0].boxplot(
        [hv_c, hv_w], tick_labels=["Cold-Start (Random P0)", "Warm-Start (5% Seed)"],
        patch_artist=True, showmeans=True, widths=0.45,
    )
    bp1["boxes"][0].set_facecolor("#fcbba1")
    bp1["boxes"][1].set_facecolor("#a1d99b")
    axes[0, 0].set_title(f"(a) Warm-Start Seeding Ablation on LA01 (4D HV ↑)\nWilcoxon p = {p_ws:.3g}", fontweight="bold", fontsize=10)
    axes[0, 0].set_ylabel("Normalized 4D Hypervolume", fontweight="bold")
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # (0, 1) Memetic N5 4D Hypervolume Boxplot on FT10
    hv_p, hv_m, p_n5 = ab_plot["n5_hv"]
    bp2 = axes[0, 1].boxplot(
        [hv_p, hv_m], tick_labels=["Pure Genetic (No N5)", "Memetic + N5 (Proposed)"],
        patch_artist=True, showmeans=True, widths=0.45,
    )
    bp2["boxes"][0].set_facecolor("#fdd0a2")
    bp2["boxes"][1].set_facecolor("#9ecae1")
    axes[0, 1].set_title(f"(b) Memetic N5 Critical-Block Ablation on FT10 (4D HV ↑)\nWilcoxon p = {p_n5:.3g}", fontweight="bold", fontsize=10)
    axes[0, 1].set_ylabel("Normalized 4D Hypervolume", fontweight="bold")
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

    # (1, 0) Breakdown Timing Sensitivity Bar Chart
    t_labels, t_rs, t_moea = sens_plot["timing"]
    x = np.arange(len(t_labels))
    w = 0.34
    axes[1, 0].bar(x - w / 2, t_rs, width=w, color="#e63946", edgecolor="black", label="Right-Shift Baseline")
    axes[1, 0].bar(x + w / 2, t_moea, width=w, color="#2a9d8f", edgecolor="black", label="U-NSGA-III Knee")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(t_labels, fontweight="bold")
    axes[1, 0].set_title("(c) Sensitivity to Breakdown Epoch t_d (LA01)", fontweight="bold", fontsize=10)
    axes[1, 0].set_ylabel("Total Tardiness (Σ T_j)", fontweight="bold")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # (1, 1) Due-Date Tightness Sensitivity Bar Chart
    k_labels, k_rs, k_moea = sens_plot["tightness"]
    xk = np.arange(len(k_labels))
    axes[1, 1].bar(xk - w / 2, k_rs, width=w, color="#e63946", edgecolor="black", label="Right-Shift Baseline")
    axes[1, 1].bar(xk + w / 2, k_moea, width=w, color="#2b5c8f", edgecolor="black", label="U-NSGA-III Knee")
    axes[1, 1].set_xticks(xk)
    axes[1, 1].set_xticklabels(k_labels, fontweight="bold")
    axes[1, 1].set_title("(d) Sensitivity to Due-Date Tightness Factor k (LA01)", fontweight="bold", fontsize=10)
    axes[1, 1].set_ylabel("Total Tardiness (Σ T_j)", fontweight="bold")
    axes[1, 1].legend(fontsize=8.5)
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle("Ablation & Parameter Sensitivity Analysis (15 Independent Seeds)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    print("=" * 95)
    print(" RUNNING ABLATION & SENSITIVITY STUDIES (15 SEEDS PER CONFIGURATION)")
    print("=" * 95)

    t0 = time.process_time()
    df_ab, ab_plot = run_ablation_experiments(num_seeds=15)
    df_sens, sens_plot = run_sensitivity_experiments(num_seeds=15)
    elapsed = time.process_time() - t0

    df_ab.to_csv("results/ablation_study.csv", index=False)
    df_sens.to_csv("results/sensitivity_study.csv", index=False)
    save_combined_plot(ab_plot, sens_plot, "results/08_ablation_and_sensitivity.png")

    print("\n[1] ABLATION STUDY RESULTS:")
    print(tabulate(df_ab, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))

    print("\n[2] PARAMETER SENSITIVITY RESULTS:")
    print(tabulate(df_sens, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    print(f"\n[DONE in {elapsed:.1f}s CPU] Saved `results/ablation_study.csv`, `results/sensitivity_study.csv`, and `results/08_ablation_and_sensitivity.png`.")


if __name__ == "__main__":
    main()

