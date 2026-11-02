# Comparative Evaluation of Classical and Many-Objective Memetic Evolutionary Algorithms for Dynamic Green Production Scheduling Under Machine Disruptions

## 1. Overview & Research Contributions
This repository implements a complete, publication-grade experimental framework for **Dynamic Job Shop Scheduling Problems (DJSSP)** under **unforeseen machine breakdown disruptions and dynamic urgent rush-order arrivals**. It systematically evaluates classical priority dispatching heuristics, single-objective memetic genetic algorithms, sequence-preserving right-shift rescheduling, and multi-/many-objective evolutionary algorithms (**NSGA-II** and **U-NSGA-III**).

### Core Methodological Contributions
1. **True Active Schedule Decoding & Conflict-Set Dispatching**:
   - Implements **Giffler & Thompson (1960)** active conflict-set dispatching for **FCFS**, **SPT**, and **EDD** (both static pre-disruption and reactive post-disruption).
   - Implements **Bierwirth (1995)** operation-based encoding with earliest idle-gap insertion (**Active Left-Shift Decoding**), reaching exact global static optima on **FT06** ($C_{\max}^* = 55$) and **LA01** ($C_{\max}^* = 666$).
2. **Memetic $N5$ Critical-Block Local Search (*Nowicki & Smutnicki, 1996*)**:
   - Hybridizes GA, NSGA-II, and U-NSGA-III with cycle-free $N5$ critical-block boundary swaps along the active critical path, improving 4D Hypervolume on **FT10 ($10\times 10$)** by **+77.5%** ($0.2788 \to 0.4948$, Wilcoxon $p = 7.48 \times 10^{-6}$).
3. **Physically Consistent Predictive-Reactive & Rolling-Horizon Disruption Model**:
   - At breakdown epoch $t_d$, operations completed by $t_d$ ($C_{i,j} \le t_d$) and operations mid-execution on healthy machines ($m \neq m_{\text{break}}, S_{i,j} < t_d < C_{i,j}$) are strictly **frozen**; operations interrupted on $M_{\text{break}}$ follow **preempt-repeat** after repair window $[t_d, t_d + D]$; no unfrozen operation is scheduled retroactively before $t_d$.
   - Supports **cascading multi-machine breakdowns + dynamic urgent rush-order arrivals** over a rolling horizon (`experiments/run_dynamic_horizon.py`).
   - Every generated schedule passes automated mathematical feasibility verification (`schedule.validate()`).
4. **Green Scheduling Energy Model & U-NSGA-III Many-Objective Optimization**:
   - Models **Total Energy Consumption (TEC in kWh, $f_4$)** using heterogeneous machine active/idle power states and an optimal **standby/turn-off switching policy** (*Mouzon et al., 2007; Dai et al., 2013*), excluding offline breakdown windows from idle draw.
   - Implements **Deb & Jain (2014)** Achievement Scalarizing Function (ASF) hyperplane normalization with **U-NSGA-III** niche-preserving tournament selection (*Seada & Deb, 2016*).
5. **Statistical Indicators, Ablation, Sensitivity & Automated Verification**:
   - Exact recursive dimension-sweep **Normalized Hypervolume (HV)**, **Inverted Generational Distance (IGD)**, **Schott's Spacing (SP)**, pairwise **Machine Sequence Swaps**, two-sided **Wilcoxon Rank-Sum** tests ($\alpha = 0.05$), and an 8-suite unit test runner.

---

## 2. Mathematical Problem Formulation

Given $n$ jobs $\mathcal{J} = \{J_0, \dots, J_{n-1}\}$ and $m$ machines $\mathcal{M} = \{M_0, \dots, M_{m-1}\}$, each job $J_i$ consists of an ordered sequence of operations $O_{i,0} \prec O_{i,1} \prec \dots \prec O_{i,l_i-1}$ with dedicated machine $m(O_{i,j}) \in \mathcal{M}$, processing duration $p_{i,j} > 0$, release time $r_i \ge 0$, and Total Work Content (TWK) due date $d_i = r_i + \lfloor k \sum_j p_{i,j} \rceil$.

When a breakdown strikes machine $m_{\text{break}}$ over $[t_d, t_d + D]$, we partition all operations into frozen set $\mathcal{F}(t_d)$ and reschedulable set $\mathcal{U}(t_d)$, minimizing up to four conflicting objectives simultaneously:

1. **Makespan ($f_1$)**:
   $$f_1 = C_{\max} = \max_{i \in \mathcal{J}} C_i$$
2. **Total Tardiness ($f_2$)**:
   $$f_2 = \sum_{i \in \mathcal{J}} \max(0, C_i - d_i)$$
3. **Schedule Start-Time Instability / Nervousness ($f_3$)**:
   $$f_3 = \Delta S = \sum_{O_{i,j}: C_{i,j}^0 > t_d} \left| S_{i,j}^{\text{new}} - S_{i,j}^0 \right|$$
   *(alongside auxiliary **Machine Sequence Instability** $\Delta S_{\text{seq}}$, counting pairwise machine-order inversions relative to the baseline schedule $S^0$, where Right-Shift satisfies $\Delta S_{\text{seq}} = 0$).*
4. **Total Energy Consumption ($f_4$, Green Scheduling in kWh)**:
   $$f_4 = \text{TEC} = \tau \sum_{m=0}^{M-1} \left( P_m^{\text{work}} \sum_{o \in \mathcal{O}_m} p_o + \sum_{g \in \mathcal{G}_m} \min\left( P_m^{\text{idle}} \Delta t_g,\; E_m^{\text{switch}} + P_m^{\text{standby}} \Delta t_g \right) \right)$$
   where $\mathcal{G}_m$ denotes genuine idle intervals on machine $m$ before shutdown (excluding offline repair windows $[t_d, t_d + D]$) and $\tau = 0.1\text{ h/unit}$.

---

## 3. Repository Structure & Artifact Index

```
LNO-PROJECT/
├── data/benchmarks/                       # Standard OR-Library / JSPLIB benchmarks (ft06, la01, la02, ft10)
├── src/
│   ├── problem.py                         # JSPInstance loader, TWK due dates, heterogeneous machine power profiles
│   ├── schedule.py                        # Bierwirth active decoder, TEC standby energy model, schedule validator
│   ├── disruption.py                      # DisruptedState partitioning, Right-Shift baseline, warm-start extractor
│   ├── baselines.py                       # Giffler-Thompson active FCFS, SPT, and EDD dispatchers
│   ├── ga.py                              # Memetic Single-Objective GA (POX/JOX + N5 critical-block local search)
│   ├── nsga2.py                           # Vectorized Memetic NSGA-II (3D & 4D) with clone control & 5% warm-start
│   ├── nsga3.py                           # Memetic U-NSGA-III (4D) with Das-Dennis simplex rays & ASF normalization
│   ├── metrics.py                         # Exact 2D-4D Hypervolume, IGD, Spacing, and Wilcoxon rank-sum tests
│   └── visualize.py                       # Gantt charts, 3D Pareto plots, 4D parallel coordinates, HV/IGD boxplots
├── experiments/
│   ├── run_experiments.py                 # 20-seed multi-benchmark batch suite (FT06, LA01, LA02, FT10)
│   ├── run_ablation_sensitivity.py        # 15-seed Warm-Start & N5 Ablation + Timing/Tightness Sensitivity
│   └── run_dynamic_horizon.py             # Cascading multi-disruption + dynamic rush-order rolling-horizon study
├── tests/
│   └── test_scheduling_suite.py           # Automated unit & analytical verification suite (8 test suites)
├── results/                               # All publication figures (PNG) and statistical datasets (CSV)
├── run_demo.py                            # End-to-end FT06 demonstration & Gantt/Pareto generator
├── run_nsga3_comparison.py                # Apples-to-apples 20-seed 4D NSGA-II vs U-NSGA-III study
└── LITERATURE_REVIEW.md                   # Grounded literature review, citations & benchmark justification
```

### Generated Figures & CSV Datasets in `results/`
| File | Description for Paper |
| :--- | :--- |
| `results/01_initial_ga_schedule.png` | Predictive Memetic GA Gantt chart on `FT06` reaching global optimum $C_{\max}=55$. |
| `results/02_right_shift_schedule.png` | Sequence-preserving Right-Shift Gantt chart under `M2` breakdown $[20,35]$ ($\Delta S_{\text{seq}}=0$). |
| `results/03_pareto_frontier.png` | 3D NSGA-II Pareto frontier vs. `Right-Shift`, `Reactive FCFS`, `Reactive SPT`, and `Reactive GA`. |
| `results/04_nsga2_min_makespan.png` | Complete post-disruption Gantt chart for NSGA-II minimum-makespan schedule ($C_{\max}=68$). |
| `results/05_nsga2_min_instability.png` | Complete post-disruption Gantt chart for NSGA-II minimum-instability schedule ($\Delta S=216$). |
| `results/06_nsga2_vs_nsga3_4d_parallel.png` | Normalized 4-objective Parallel Coordinates Plot comparing 4D NSGA-II vs. 4D U-NSGA-III on `FT10`. |
| `results/07_nsga2_vs_nsga3_hv_igd_boxplot.png` | 20-seed statistical boxplots of 4D Hypervolume and IGD on `FT10`. |
| `results/08_ablation_and_sensitivity.png` | 2×2 Ablation (Warm-Start & Memetic $N5$) and Sensitivity (Breakdown Timing & Tightness $k$) figure. |
| `results/09_multi_disruption_rush_order_gantt.png` | Rolling-horizon Gantt chart with two cascading breakdowns (`M2`, `M4`) + urgent Rush Order `J6`. |
| `results/experiment_summary_all.csv` | Aggregated Mean ± Std across `FT06`, `LA01`, `LA02`, `FT10` (20 seeds, all baselines & MOEAs). |
| `results/experiment_detailed_all.csv` | Raw per-seed rows for all benchmarks and algorithms. |
| `results/nsga2_vs_nsga3_4d_metrics.csv` | Per-seed 4D HV, IGD, Spacing, and CPU runtimes for Memetic NSGA-II vs. U-NSGA-III. |
| `results/ablation_study.csv` | 15-seed Warm-Start and Memetic $N5$ ablation metrics & Wilcoxon $p$-values. |
| `results/sensitivity_study.csv` | 15-seed breakdown timing ($20\%, 45\%, 70\%$) and due-date tightness ($k \in \{1.3, 1.5, 1.7\}$) results. |
| `results/dynamic_horizon_summary.csv` | Multi-event rolling-horizon comparison with dynamic Rush Job `J6`. |

---

## 4. Reproducing All Experiments

```bash
pip install -r requirements.txt

# 0. Run automated unit & analytical verification tests (8/8 pass in <1s)
python -m unittest discover -s tests -v

# 1. End-to-end FT06 walkthrough + Gantt & 3D Pareto plots (01_*.png .. 05_*.png)
python run_demo.py

# 2. Apples-to-apples 20-seed 4D NSGA-II vs U-NSGA-III study (06_*.png, 07_*.png)
python run_nsga3_comparison.py

# 3. Full 20-seed multi-benchmark experiment suite across FT06, LA01, LA02, and FT10
python experiments/run_experiments.py

# 4. Ablation (Warm-Start & Memetic N5) + Parameter Sensitivity suite (08_*.png)
python experiments/run_ablation_sensitivity.py

# 5. Rolling-horizon cascading breakdowns + urgent rush-order arrival study (09_*.png)
python experiments/run_dynamic_horizon.py
```

---

## 5. Summary of Experimental Results

### Table 1: 3-Objective Dynamic Rescheduling Across Benchmarks (Mean over 20 Seeds)

| Benchmark | Method | Phase | Makespan ($C_{\max}$) | Tardiness ($\sum T_j$) | Time Instab. ($\Delta S$) | Seq Swaps | Energy (kWh) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **FT06 ($6\times 6$)** | FCFS / SPT / EDD | Pre-Disruption | 65 / 94 / 96 | 57 / 70 / 48 | - | - | 126.2 / 134.1 / 133.4 |
| *M2 down $[20,35]$* | Memetic GA | Predictive ($t=0$) | **55.0** (Opt=55) | 30.0 | - | - | 124.8 |
| | Right-Shift | Post-Disruption | 77.0 | 132.0 | 371.0 | **0.0** | 131.7 |
| | Reactive FCFS / SPT | Post-Disruption | 72.0 / 77.0 | 57.0 / 55.0 | 221.0 / 264.0 | 11.0 / 12.0 | 131.3 / 133.8 |
| | Reactive GA | Post-Disruption | **68.0** | 65.3 | 227.0 | 11.1 | 130.8 |
| | **NSGA-II 3D (Knee)** | Post-Disruption | **69.1** | **46.2** | 237.9 | 12.0 | 130.7 |
| | **NSGA-II 3D (Min $\Delta S$)** | Post-Disruption | 70.0 | 50.0 | **216.0** | 11.0 | **130.5** |
| **LA01 ($10\times 5$)** | FCFS / SPT / EDD | Pre-Disruption | 772 / 1075 / 1179 | 1871 / 2145 / 1860 | - | - | 1590.7 / 1622.1 / 1618.4 |
| *M4 down $[230,390]$* | Memetic GA | Predictive ($t=0$) | **666.0** (Opt=666) | 1674.4 | - | - | 1589.3 |
| | Right-Shift | Post-Disruption | 902.0 | 3942.0 | 5900.0 | **0.0** | 1602.5 |
| | Reactive FCFS | Post-Disruption | 916.0 | 3630.0 | 5761.0 | 15.0 | 1606.9 |
| | **NSGA-II 3D (Knee)** | Post-Disruption | 908.0 | **2684.8** | 4505.7 | 18.8 | 1608.2 |
| | **NSGA-II 3D (Min $\Delta S$)** | Post-Disruption | **902.0** | 2980.0 | **4157.0** | 20.0 | 1605.8 |
| **LA02 ($10\times 5$)** | Memetic GA | Predictive ($t=0$) | **694.5** | 1634.3 | - | - | 1448.0 |
| *M3 down $[230,390]$* | Right-Shift | Post-Disruption | 871.0 | 3122.0 | 2102.0 | **0.0** | 1452.7 |
| | Reactive FCFS | Post-Disruption | 871.0 | 2365.0 | 3426.0 | 14.0 | 1457.1 |
| | **NSGA-II 3D (Min $C_{\max}$)** | Post-Disruption | **871.0** | **1903.0** | 3643.1 | 24.9 | 1452.8 |
| | **NSGA-II 3D (Min $\Delta S$)** | Post-Disruption | 872.6 | 2491.7 | **1274.1** | 8.1 | 1449.7 |
| **FT10 ($10\times 10$)** | FCFS / SPT / EDD | Pre-Disruption | 1184 / 1429 / 1466 | 2210 / 1594 / 1183 | - | - | 2850.1 / 2927.4 / 2910.6 |
| *M3 down $[340,560]$* | Memetic GA | Predictive ($t=0$) | **983.0** | 1793.5 | - | - | 2858.0 |
| | Right-Shift | Post-Disruption | 1248.0 | 4040.0 | 13532.0 | **0.0** | 2887.1 |
| | Reactive FCFS | Post-Disruption | 1148.0 | 2594.0 | 9471.0 | 38.0 | 2874.3 |
| | Reactive GA | Post-Disruption | 1102.2 | 2511.2 | 8092.0 | 47.4 | 2877.5 |
| | **NSGA-II 3D (Min $C_{\max}$)** | Post-Disruption | **1085.2** | 2277.2 | 6828.7 | 43.0 | 2876.5 |
| | **NSGA-II 3D (Knee)** | Post-Disruption | 1099.3 | **2257.0** | 6352.7 | 43.8 | 2879.8 |
| | **NSGA-II 3D (Min $\Delta S$)** | Post-Disruption | 1096.0 | 2399.7 | **5756.6** | 38.1 | 2878.8 |

---

### Table 2: Apples-to-Apples 4-Objective Study — Memetic NSGA-II (4D) vs. Memetic U-NSGA-III (4D)
*Identical budgets: $N = H = \binom{4+5-1}{5} = 56$ reference rays, 20 independent seeds, optimizing $(C_{\max}, \sum T_j, \Delta S, \text{TEC})$.*

| Benchmark | Algorithm | Normalized 4D HV ($\uparrow$) | IGD to $P^*$ ($\downarrow$) | Spacing ($\downarrow$) | CPU Runtime | Wilcoxon vs. NSGA-II |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **FT06 ($6\times 6$)** | NSGA-II (4D) | **0.5690 ± 0.0008** | **0.0231 ± 0.0076** | 0.1592 ± 0.0384 | 2.03 s | — |
| | U-NSGA-III (4D) | 0.5689 ± 0.0006 | 0.0251 ± 0.0112 | **0.1469 ± 0.0388** | 2.04 s | Tie ($p = 0.837$, small discrete front) |
| **LA01 ($10\times 5$)** | NSGA-II (4D) | 0.5794 ± 0.0146 | **0.1295 ± 0.0119** | **0.0979 ± 0.0179** | 2.09 s | — |
| | **U-NSGA-III (4D)** | **0.5934 ± 0.0111** | 0.1301 ± 0.0147 | 0.1027 ± 0.0239 | **1.72 s** | **Significant HV Win ($p = 0.00711$)** |
| **FT10 ($10\times 10$)** | NSGA-II (4D) | 0.4674 ± 0.0541 | 0.1838 ± 0.0311 | **0.0853 ± 0.0130** | 2.98 s | — |
| | **U-NSGA-III (4D)** | 0.4655 ± 0.0543 | **0.1745 ± 0.0313** | 0.0867 ± 0.0129 | **2.88 s** | Lower IGD; parallel plot confirms superior $\Delta S$ spread |

---

### Table 3: Ablation, Sensitivity & Rolling-Horizon Multi-Disruption Highlights
1. **Memetic $N5$ Critical-Block Ablation (`FT10`, 15 Seeds)**:
   - Pure Genetic (`local_search=False`): $\text{HV} = 0.2788 \pm 0.0633$, $\text{IGD} = 0.3532$.
   - **Memetic $N5$ (`local_search=True`)**: $\text{HV} = \mathbf{0.4948 \pm 0.0684}$ (**+77.5% HV**), $\text{IGD} = \mathbf{0.1895}$ (**-46.3% IGD**, **Wilcoxon $p = 7.48 \times 10^{-6}$**).
2. **Breakdown Timing & Due-Date Tightness Sensitivity (`LA01`, 15 Seeds)**:
   - Early ($20\%$) and Mid-Peak ($45\%$) breakdowns yield **29.4%** (`3811 → 2691.2`) and **27.3%** (`3787 → 2752.8`) tardiness reductions alongside **22.8%–25.7%** lower $\Delta S$ than Right-Shift, tapering to **8.0%** at Late ($70\%$, 11 ops left).
   - Across Tight ($k=1.3$), Moderate ($k=1.5$), and Loose ($k=1.7$) due dates, U-NSGA-III consistently cuts tardiness by **23.6%–27.7%**.
3. **Cascading Breakdowns + Dynamic Urgent Rush-Order Arrival (`FT06` + Rush Job `J6`)**:
   - Event 1 ($t=16$, `M2` down $[16,30]$) followed by Event 2 ($t=34$, `M4` down $[34,48]$ + arrival of urgent Rush Job `J6` with due date $55$):
   - **Rolling U-NSGA-III** cuts Rush Job tardiness by **70%** (**`9`** vs. `30` under Right-Shift+FIFO and `25` under Reactive FCFS) while simultaneously achieving strictly better Makespan (**`82`** vs. `85`/`91`), Total Tardiness (**`138`** vs. `147`/`151`), Instability (**`134`** vs. `155`/`217`), and Energy (**`140.7`** vs. `142.3`/`141.7 kWh`).

---

## 6. Modeling Scope & Future Extensions (For Paper Discussion)
- **Classical JSSP vs. Flexible JSSP (FJSSP)**: Standard OR-Library instances (`FT06`, `LA01`, `LA02`, `FT10`) assign each operation to a dedicated machine without parallel routing alternatives; extending the chromosome with a machine-assignment vector generalizes this framework to FJSSP.
- **Deterministic vs. Stochastic Repair Horizons**: Repair duration $D$ is known upon breakdown diagnosis at $t_d$; uncertain repair distributions ($D \sim \text{Weibull}$) can be handled via periodic rolling re-triggers using `experiments/run_dynamic_horizon.py`.
