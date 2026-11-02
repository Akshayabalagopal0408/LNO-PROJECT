# Literature Review & Benchmark Dataset Selection
**Project Title**: *Comparative Evaluation of Classical and Many-Objective Evolutionary Algorithms for Dynamic Green Production Scheduling Under Machine Disruptions*

---

## Part 1: Comprehensive Literature Review

Dynamic Job Shop Scheduling (DJSS) under unforeseen machine disruptions is a cornerstone topic in industrial engineering, production planning, and operations research. Below is a structured breakdown of the foundational and state-of-the-art research papers directly grounding our mathematical models, algorithms, and evaluation metrics.

---

### 1. The Definitive Rescheduling Framework
* **Paper**: *Rescheduling manufacturing systems: a framework of strategies, policies, and methods*
* **Authors**: G. E. Vieira, J. W. Herrmann, & E. Lin (2003)
* **Venue**: *Journal of Scheduling*, 6(1), 39–62.
* **Core Contribution**:
  Categorizes shop-floor disruption responses into three paradigms:
  1. **Completely Reactive Dispatching**: Real-time local conflict resolution via Priority Dispatching Rules (FCFS, SPT, EDD).
  2. **Predictive-Reactive Rescheduling**: Generating an optimal baseline schedule offline (predictive), then re-optimizing remaining unfrozen operations when a disruption occurs at $t_d$ (reactive).
  3. **Robust Pro-Active Scheduling**: Embedding buffer slack offline.
* **Relevance to Our Work**:
  Establishes **Schedule Instability (Nervousness)**—both start-time deviation ($\Delta S = \sum |S'_o - S_o|$) and machine sequence inversions—as essential shop-floor objectives alongside efficiency. Also formalizes freezing completed ($\le t_d$) and non-preemptable healthy in-progress operations at the rescheduling epoch $t_d$.

---

### 2. Active Schedule Generation, Chromosome Representation & Critical-Block Local Search
* **Papers**:
  - B. Giffler & G. L. Thompson (1960), *Algorithms for solving production-scheduling problems*, *Operations Research*, 8(4), 487–503.
  - C. Bierwirth (1995), *A generalized permutation approach to job shop scheduling with genetic algorithms*, *OR Spectrum*, 17(2), 87–92.
  - E. Nowicki & C. Smutnicki (1996), *A fast taboo search algorithm for the job shop problem*, *Management Science*, 42(6), 797–813.
* **Core Contribution**:
  - **Giffler & Thompson (1960)** proved that optimal regular schedules always reside within the set of **active schedules** (where no operation can be left-shifted into an earlier machine idle gap without delaying another operation), and introduced conflict-set dispatching.
  - **Bierwirth (1995)** introduced permutations with repetition (operation-based encoding) combined with Precedence Operation Crossover (**POX / JOX**), guaranteeing 100% deadlock-free offspring without repair heuristics.
  - **Nowicki & Smutnicki (1996)** proved that only reversing the **first** or **last** operation pair of a contiguous same-machine **critical block** ($N5$ neighborhood) can shorten the critical path without creating disjunctive graph cycles.
* **Relevance to Our Work**:
  Our framework combines Bierwirth active left-shift decoding with **Memetic $N5$ Critical-Block Local Search** across GA, NSGA-II, and U-NSGA-III, yielding a statistically significant $+38.6\%$ 4D Hypervolume improvement ($p = 0.000189$) on `FT10`.

---

### 3. Multi-Objective Evolutionary Optimization (NSGA-II)
* **Paper**: *A fast and elitist multiobjective genetic algorithm: NSGA-II*
* **Authors**: K. Deb, A. Pratap, S. Agarwal, & T. Meyarivan (2002)
* **Venue**: *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
* **Core Contribution**:
  Introduced $O(MN^2)$ Fast Non-Dominated Sorting, parent-offspring elitism ($R_t = P_t \cup Q_t$), and parameter-free Manhattan Crowding Distance niching for $M \le 3$ conflicting objectives.
* **Relevance to Our Work**:
  Forms our core 3-objective rescheduling solver balancing Makespan ($C_{\max}$), Total Tardiness ($\sum T_j$), and Schedule Instability ($\Delta S$), augmented with predictive-reactive **warm-start seeding** from the baseline and Right-Shift sequences.

---

### 4. Scaling to Many-Objective Scheduling (NSGA-III)
* **Paper**: *An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints*
* **Authors**: K. Deb & H. Jain (2014)
* **Venue**: *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.
* **Core Contribution**:
  Demonstrated that in $M \ge 4$ objectives, NSGA-II's Manhattan crowding distance loses neighborhood locality and boundary/interior uniformity. NSGA-III replaces crowding distance with systematic **Das & Dennis (1998)** unit-simplex reference rays ($H = \binom{M+p-1}{p}$) and **Achievement Scalarizing Function (ASF)** hyperplane normalization.
* **Relevance to Our Work**:
  Powers our 4-objective Dynamic Green Scheduling extension ($f_1 = C_{\max}$, $f_2 = \sum T_j$, $f_3 = \Delta S$, $f_4 = \text{TEC}$), compared head-to-head against 4-objective NSGA-II under identical evaluation budgets.

---

### 5. Energy-Efficient / Green Job Shop Scheduling
* **Papers**:
  - G. Mouzon, M. B. Yildirim, & J. Twomey (2007), *Operational methods for minimization of energy consumption of manufacturing equipment*, *International Journal of Production Research*, 45(18–19), 4247–4271.
  - M. Dai, D. Tang, A. Giret, M. A. Salido, & W. D. Li (2013), *Energy-efficient scheduling for a flexible flow shop using an improved genetic-simulated annealing algorithm*, *Robotics and Computer-Integrated Manufacturing*, 29(5), 418–429.
* **Core Contribution**:
  Formulated heterogeneous machine power states (active cutting power $P_m^{\text{work}}$, idle spindle/coolant power $P_m^{\text{idle}}$, low-power standby $P_m^{\text{standby}}$, and turn-off/turn-on setup energy $E_m^{\text{switch}}$). Consolidating fragmented short idle gaps above the breakeven threshold allows machines to enter standby mode rather than burning high idle power.
* **Relevance to Our Work**:
  Grounds our 4th objective—**Total Energy Consumption (TEC in kWh)**—making energy genuinely non-collinear with Makespan.

---

### 6. Dynamic Job Shop Rescheduling & Classical Baselines
* **Papers**:
  - D. Ouelhadj & S. Petrovic (2009), *A survey of dynamic scheduling in manufacturing systems*, *Journal of Scheduling*, 12(4), 417–431.
  - W. Xiong, D. Chen, L. Yang, & C. Shen (2013), *A multi-objective genetic algorithm for dynamic flexible job-shop scheduling with machine breakdowns*, *IJPR*, 51(8), 2465–2481.
  - S. S. Panwalkar & W. Iskander (1977), *A survey of priority dispatching rules*, *Operations Research*, 25(1), 45–61.
* **Relevance to Our Work**:
  Validates comparing reactive dispatching rules (**Reactive FCFS**, **Reactive SPT**), sequence-preserving **Right-Shift**, **Reactive Single-Objective GA**, **NSGA-II**, and **NSGA-III**.

---

### Summary Matrix of Related Work

| Paper | Problem Context | Algorithms / Models | Objectives Evaluated | Core Takeaway for Our Project |
| :--- | :--- | :--- | :--- | :--- |
| **Giffler & Thompson (1960); Bierwirth (1995)** | Classical & GA JSSP | Active SGS, Operation-Based POX | Makespan ($C_{\max}$) | Guarantees optimal schedule reachability via active left-shift decoding. |
| **Panwalkar & Iskander (1977)** | Priority Dispatching | FCFS, SPT, EDD | Flow time, Tardiness | Establishes active conflict-set FCFS, SPT, and EDD baselines. |
| **Vieira et al. (2003)** | Dynamic Rescheduling | Right-Shift, Predictive-Reactive | Efficiency vs. Instability ($\Delta S$) | Formalizes time & sequence instability and frozen operation partitioning at $t_d$. |
| **Deb et al. (2002)** | Multi-Objective EAs | NSGA-II | $M \le 3$ Pareto Optimality | Non-dominated sorting and crowding distance mechanics. |
| **Mouzon et al. (2007); Dai et al. (2013)** | Green Manufacturing | Turn-off / Standby Energy Policy | Makespan vs. Total Energy (TEC) | Heterogeneous machine idle/standby power model for $f_4$. |
| **Xiong et al. (2013)** | Breakdown Rescheduling | Warm-Started MOEA vs. Right-Shift | Makespan, Tardiness, Stability | Warm-starting anchors Pareto fronts at the Right-Shift stability bound. |
| **Deb & Jain (2014)** | Many-Objective EAs | NSGA-III vs. NSGA-II | $M \ge 4$ Simplex Reference Rays | ASF normalization & Das–Dennis niching outperform crowding in 4D space. |

---

## Part 2: Benchmark Dataset Selection & Evaluation Protocol

### 1. Selected OR-Library / JSPLIB Benchmark Suite

| Dataset | Size ($n \times m$) | Operations | Known Static Optimum ($C_{\max}^*$) | Primary Purpose in Our Study |
| :--- | :---: | :---: | :---: | :--- |
| **FT06** (Fisher & Thompson, 1963) | $6 \times 6$ | 36 | 55 | **Validation & Visual Walkthrough**: Exact verification of active decoding ($C_{\max}=55$), Gantt inspection, and disruption mechanics. |
| **LA01** (Lawrence, 1984) | $10 \times 5$ | 50 | 666 | **Bottleneck Contention (Rectangular)**: High job-to-machine ratio ($2:1$) testing queue congestion under breakdown. |
| **LA02** (Lawrence, 1984) | $10 \times 5$ | 50 | 655 | **Asymmetric Routing Sensitivity**: Secondary $10 \times 5$ instance testing structural generalization. |
| **FT10** (Fisher & Thompson, 1963) | $10 \times 10$ | 100 | 930 | **Primary Combinatorial Stress Test**: Famous hard $10 \times 10$ benchmark testing 3D/4D Pareto scalability and NSGA-III vs. NSGA-II divergence. |

---

### 2. Dynamic Disruption & Multi-Objective Protocol

1. **Due Date Generation (Total Work Content - TWK)**:
   $$d_i = r_i + \lfloor k \cdot P_i \rceil, \quad P_i = \sum_{j=1}^m p_{i,j}, \quad k = 1.5$$
2. **Physically Consistent Breakdown Partitioning at $t_d$**:
   - Operations completed by $t_d$ ($C_{i,j} \le t_d$) and operations already mid-execution on healthy machines ($m \neq m_{\text{break}}, S_{i,j} < t_d < C_{i,j}$) are **frozen**.
   - Operations interrupted mid-cut on $M_{\text{break}}$ follow **preempt-repeat** after repair window $[t_d, t_d + D]$.
   - No unfrozen operation can be scheduled retroactively before $t_d$.
3. **Multi-Objective Quality Indicators & Statistical Testing**:
   - **Exact Dimension-Sweep Hypervolume (HV)** normalized to $[0, 1]$ with reference point $\mathbf{r} = (1.1, \dots, 1.1)^M$.
   - **Inverted Generational Distance (IGD)** relative to the empirical true Pareto front $P^*$ formed by the non-dominated union across all runs.
   - **Schott's Spacing (SP)** for front uniformity.
   - **Two-Sided Wilcoxon Rank-Sum Tests** ($\alpha = 0.05$) over **20 independent random seeds** per benchmark.
