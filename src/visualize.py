"""
visualize.py: Publication-grade visualization suite for Dynamic Job Shop Scheduling.
Produces high-resolution Gantt charts, 3D Pareto Trade-Off Frontiers,
4D Parallel Coordinates Plots, and Multi-Seed HV / IGD Statistical Boxplots.
"""

from typing import List, Dict, Optional, Tuple
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.schedule import Schedule
from src.disruption import BreakdownDisruption


def plot_gantt_chart(
    schedule: Schedule,
    disruption: Optional[BreakdownDisruption] = None,
    baseline_schedule: Optional[Schedule] = None,
    title: str = "Production Schedule Gantt Chart",
    save_path: Optional[str] = None,
    show_plot: bool = False,
):
    """
    Renders a publication-quality Gantt chart of the complete job shop schedule.
    Highlights frozen pre-disruption operations, the disruption epoch t_d, and
    the machine repair window [t_d, t_d + D].
    """
    num_machines = schedule.instance.num_machines
    fig, ax = plt.subplots(figsize=(12.5, max(4.2, num_machines * 0.78)), dpi=180)

    cmap_name = "tab20" if schedule.instance.num_jobs > 10 else "tab10"
    cmap = matplotlib.colormaps[cmap_name]
    job_colors = {j: cmap(j % 20) for j in range(schedule.instance.num_jobs)}

    for op in schedule.operations:
        y_pos = op.machine_id - 0.35
        height = 0.70
        width = op.duration
        x_pos = op.start_time

        color = job_colors[op.job_id]
        edge_style = "-" if not op.is_frozen else "--"
        alpha_val = 0.92 if not op.is_frozen else 0.72

        rect = patches.Rectangle(
            (x_pos, y_pos),
            width,
            height,
            linewidth=1.1,
            linestyle=edge_style,
            edgecolor="#1a1a1a",
            facecolor=color,
            alpha=alpha_val,
            zorder=3,
        )
        ax.add_patch(rect)

        if width >= max(1.5, schedule.makespan * 0.018):
            text_label = f"J{op.job_id}O{op.op_idx}"
            ax.text(
                x_pos + width / 2,
                y_pos + height / 2,
                text_label,
                ha="center",
                va="center",
                color="black",
                fontsize=7.5 if schedule.instance.num_jobs <= 10 else 6.5,
                fontweight="bold",
                zorder=4,
            )

    if disruption is not None:
        # Vertical dashed line marking disruption epoch t_d
        ax.axvline(
            x=disruption.breakdown_start,
            color="#c1121f",
            linestyle="--",
            linewidth=1.6,
            zorder=5,
            label=f"Disruption Epoch (t_d={disruption.breakdown_start})",
        )

        y_pos = disruption.machine_id - 0.44
        height = 0.88
        x_pos = disruption.breakdown_start
        width = disruption.duration

        breakdown_patch = patches.Rectangle(
            (x_pos, y_pos),
            width,
            height,
            linewidth=1.6,
            edgecolor="#9b0000",
            facecolor="#ff4d4d",
            hatch="//",
            alpha=0.65,
            zorder=5,
        )
        ax.add_patch(breakdown_patch)
        ax.text(
            x_pos + width / 2,
            y_pos + height / 2,
            f"M{disruption.machine_id} DOWN\n[{x_pos}, {x_pos + width}]",
            ha="center",
            va="center",
            color="white",
            fontsize=7.2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#111111", alpha=0.78),
            zorder=6,
        )

    subtitle_parts = [
        f"Makespan: {schedule.makespan}",
        f"Tardiness: {schedule.total_tardiness}",
        f"Energy: {schedule.total_energy_consumption:.1f} kWh",
    ]
    if baseline_schedule is not None and disruption is not None:
        instab = schedule.compute_instability(
            baseline_schedule, impacted_only_since=disruption.breakdown_start
        )
        seq_inv = schedule.compute_sequence_instability(baseline_schedule)
        subtitle_parts.insert(2, f"Instability (ΔS): {instab} (Seq Swaps: {seq_inv})")

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(num_machines)], fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (Scheduling Units)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Shop-Floor Machines", fontsize=10, fontweight="bold")
    ax.set_title(
        f"{title}\n" + "  |  ".join(subtitle_parts),
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlim(0, max(schedule.makespan * 1.05, 20))
    ax.set_ylim(-0.6, num_machines - 0.4)
    ax.grid(axis="x", linestyle="--", alpha=0.45)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_pareto_front(
    pareto_objs: np.ndarray,
    baseline_points: Optional[Dict[str, Tuple[float, float, float]]] = None,
    save_path: Optional[str] = None,
    title: str = "NSGA-II Pareto Trade-Off Frontier vs Reactive Baselines",
):
    """
    Plots Makespan vs Total Tardiness colored by Schedule Instability (Delta S),
    overlaying discrete reactive baselines (Right-Shift, Reactive FCFS/SPT/GA).
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)

    sc = ax.scatter(
        pareto_objs[:, 0],
        pareto_objs[:, 1],
        c=pareto_objs[:, 2],
        cmap="viridis",
        s=130,
        edgecolors="black",
        linewidths=1.4,
        alpha=0.92,
        label="NSGA-II Non-Dominated Front",
        zorder=4,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Schedule Instability (ΔS)", fontsize=10, fontweight="bold")

    sorted_order = np.lexsort((pareto_objs[:, 1], pareto_objs[:, 0]))
    ax.plot(
        pareto_objs[sorted_order, 0],
        pareto_objs[sorted_order, 1],
        linestyle="--",
        color="#2b5c8f",
        alpha=0.55,
        zorder=3,
    )

    if baseline_points:
        markers = {
            "Right-Shift": "s",
            "Reactive FCFS": "^",
            "Reactive SPT": "D",
            "Reactive GA": "X",
            "FCFS": "^",
            "SPT": "D",
            "Single-Obj GA": "X",
        }
        colors = {
            "Right-Shift": "#e63946",
            "Reactive FCFS": "#e76f51",
            "Reactive SPT": "#f4a261",
            "Reactive GA": "#2a9d8f",
            "FCFS": "#e76f51",
            "SPT": "#f4a261",
            "Single-Obj GA": "#2a9d8f",
        }

        all_x = list(pareto_objs[:, 0]) + [pt[0] for pt in baseline_points.values()]
        x_mid = 0.5 * (min(all_x) + max(all_x))

        for idx, (name, pt) in enumerate(baseline_points.items()):
            mk = markers.get(name, "o")
            col = colors.get(name, "#333333")
            ax.scatter(
                pt[0],
                pt[1],
                marker=mk,
                color=col,
                s=165,
                edgecolors="black",
                linewidths=1.6,
                label=f"{name} (Cmax={int(pt[0])}, Tard={int(pt[1])}, ΔS={int(pt[2])})",
                zorder=5,
            )
            on_right = pt[0] > x_mid
            offset_x = -8 if on_right else 8
            ha_align = "right" if on_right else "left"
            offset_y = 6 if idx % 2 == 0 else -12
            ax.annotate(
                name,
                (pt[0], pt[1]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha=ha_align,
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.88, edgecolor=col),
                zorder=6,
            )

    ax.set_xlabel("Makespan (C_max)  →  [Minimize]", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Total Tardiness (Σ T_j)  →  [Minimize]", fontsize=10.5, fontweight="bold")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close(fig)


def plot_parallel_coordinates_4d(
    objs_nsga2: np.ndarray,
    objs_nsga3: np.ndarray,
    save_path: str,
    title: str = "4-Objective Pareto Front Distribution: NSGA-II vs NSGA-III",
):
    """
    Side-by-side normalized Parallel Coordinates Plot (IEEE TEVC standard for M >= 4)
    showing objective trade-off coverage across Makespan, Tardiness, Instability, and Energy.
    """
    combined = np.vstack([objs_nsga2, objs_nsga3])
    ideal = np.min(combined, axis=0)
    nadir = np.max(combined, axis=0)
    span = np.where(nadir - ideal > 1e-9, nadir - ideal, 1.0)

    norm_2 = (objs_nsga2 - ideal) / span
    norm_3 = (objs_nsga3 - ideal) / span

    labels = [
        f"f1: Makespan\n[{ideal[0]:.0f}, {nadir[0]:.0f}]",
        f"f2: Tardiness\n[{ideal[1]:.0f}, {nadir[1]:.0f}]",
        f"f3: Instability\n[{ideal[2]:.0f}, {nadir[2]:.0f}]",
        f"f4: Energy (kWh)\n[{ideal[3]:.1f}, {nadir[3]:.1f}]",
    ]
    x = np.arange(4)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180, sharey=True)

    for ax, data, name, col in [
        (axes[0], norm_2, f"NSGA-II (4D Crowding, |F0|={len(objs_nsga2)})", "#d95f02"),
        (axes[1], norm_3, f"NSGA-III (4D Ref-Point, |F0|={len(objs_nsga3)})", "#1b9e77"),
    ]:
        for row in data:
            ax.plot(x, row, color=col, alpha=0.48, linewidth=1.5, marker="o", markersize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.8, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(name, fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.45)

    axes[0].set_ylabel("Normalized Objective Value  [0 = Ideal, 1 = Nadir]", fontsize=9.5, fontweight="bold")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_hv_igd_boxplots(
    hv_nsga2: List[float],
    hv_nsga3: List[float],
    igd_nsga2: List[float],
    igd_nsga3: List[float],
    p_hv: float,
    p_igd: float,
    save_path: str,
    instance_label: str = "FT10 (10x10)",
):
    """
    Renders side-by-side statistical boxplots of 4D Hypervolume (HV) and IGD
    across independent random seeds, annotated with Wilcoxon rank-sum p-values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)

    # Panel 1: Hypervolume (Higher is better)
    bp1 = axes[0].boxplot(
        [hv_nsga2, hv_nsga3],
        tick_labels=["NSGA-II (4D)", "NSGA-III (4D)"],
        patch_artist=True,
        showmeans=True,
        widths=0.45,
    )
    bp1["boxes"][0].set_facecolor("#fdd0a2")
    bp1["boxes"][1].set_facecolor("#a1d99b")
    axes[0].set_title(
        f"Normalized 4D Hypervolume (↑ Better)\nWilcoxon p = {p_hv:.4g}",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0].set_ylabel("Hypervolume (HV)", fontsize=9.5, fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 2: IGD (Lower is better)
    bp2 = axes[1].boxplot(
        [igd_nsga2, igd_nsga3],
        tick_labels=["NSGA-II (4D)", "NSGA-III (4D)"],
        patch_artist=True,
        showmeans=True,
        widths=0.45,
    )
    bp2["boxes"][0].set_facecolor("#fdd0a2")
    bp2["boxes"][1].set_facecolor("#a1d99b")
    axes[1].set_title(
        f"Inverted Generational Distance (↓ Better)\nWilcoxon p = {p_igd:.4g}",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].set_ylabel("IGD to Reference Front P*", fontsize=9.5, fontweight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(
        f"Many-Objective Statistical Comparison ({instance_label}, {len(hv_nsga2)} Independent Seeds)",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
