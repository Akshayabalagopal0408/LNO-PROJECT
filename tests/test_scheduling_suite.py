"""
test_scheduling_suite.py: Comprehensive automated unit and property test suite
verifying mathematical invariants across all project modules.

Run via:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.problem import load_jsp_instance
from src.schedule import decode_chromosome, create_random_chromosome
from src.disruption import (
    BreakdownDisruption,
    build_disrupted_state,
    apply_right_shift,
)
from src.baselines import solve_fcfs, solve_spt, solve_edd
from src.ga import run_genetic_algorithm, generate_n5_critical_neighbors
from src.nsga2 import run_nsga2
from src.nsga3 import run_nsga3, generate_reference_points
from src.metrics import (
    extract_nondominated,
    _hv_exact_recursive,
    compute_hypervolume,
    compute_igd,
    wilcoxon_rank_sum,
)


class TestDynamicSchedulingFramework(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ft06 = load_jsp_instance("data/benchmarks/ft06.txt", tightness_factor=1.5)
        cls.la01 = load_jsp_instance("data/benchmarks/la01.txt", tightness_factor=1.5)

    def test_01_instance_loading_and_power_profiles(self):
        self.assertEqual(self.ft06.num_jobs, 6)
        self.assertEqual(self.ft06.num_machines, 6)
        self.assertEqual(self.ft06.total_operations, 36)
        self.assertEqual(len(self.ft06.machine_work_power), 6)
        # Heterogeneous idle power ensures non-collinearity with makespan
        self.assertGreater(len(set(self.ft06.machine_idle_power.values())), 1)

    def test_02_active_decoding_and_ft06_global_optimum(self):
        for s in range(5):
            np.random.seed(s)
            chrom = create_random_chromosome(self.ft06)
            sched_active = decode_chromosome(chrom, self.ft06, active=True)
            sched_semi = decode_chromosome(chrom, self.ft06, active=False)
            self.assertTrue(sched_active.validate())
            self.assertTrue(sched_semi.validate())
            self.assertLessEqual(sched_active.makespan, sched_semi.makespan)

        ga_sched, _, _ = run_genetic_algorithm(
            self.ft06, pop_size=45, generations=70, seed=42
        )
        self.assertTrue(ga_sched.validate())
        self.assertEqual(ga_sched.makespan, 55, "Memetic Active GA should hit exact FT06 optimum 55")

    def test_03_giffler_thompson_dispatchers_pre_and_post_disruption(self):
        for solver in (solve_fcfs, solve_spt, solve_edd):
            sched = solver(self.ft06)
            self.assertTrue(sched.validate())

        init_sched, _, _ = run_genetic_algorithm(self.ft06, pop_size=40, generations=50, seed=42)
        disruption = BreakdownDisruption(machine_id=2, breakdown_start=20, duration=15)
        state = build_disrupted_state(init_sched, disruption, self.ft06)

        for solver in (solve_fcfs, solve_spt, solve_edd):
            r_sched = solver(self.ft06, disrupted_state=state)
            self.assertTrue(r_sched.validate())
            for op in r_sched.operations:
                if not op.is_frozen:
                    self.assertGreaterEqual(op.start_time, disruption.breakdown_start)

    def test_04_disruption_partitioning_and_right_shift_invariants(self):
        init_sched, _, _ = run_genetic_algorithm(self.ft06, pop_size=45, generations=60, seed=42)
        disruption = BreakdownDisruption(machine_id=2, breakdown_start=20, duration=15)
        state = build_disrupted_state(init_sched, disruption, self.ft06)

        # Verify in-progress healthy machine ops straddling t_d=20 are frozen and preserved
        straddling_healthy = [
            op for op in init_sched.operations
            if op.machine_id != 2 and op.start_time < 20 < op.end_time
        ]
        frozen_keys = {op.op_key for op in state.frozen_ops}
        for s_op in straddling_healthy:
            self.assertIn(s_op.op_key, frozen_keys)

        rs_sched = apply_right_shift(init_sched, disruption, self.ft06)
        self.assertTrue(rs_sched.validate())
        self.assertEqual(
            rs_sched.compute_sequence_instability(init_sched),
            0,
            "Right-Shift must preserve exact machine sequences (0 inversions)",
        )

    def test_05_n5_critical_block_neighbors_validity(self):
        sched = solve_fcfs(self.la01)
        neighbors = generate_n5_critical_neighbors(sched)
        self.assertGreater(len(neighbors), 0)
        for nb in neighbors:
            self.assertEqual(len(nb), self.la01.total_operations)
            nb_sched = decode_chromosome(nb, self.la01, active=True)
            self.assertTrue(nb_sched.validate())

    def test_06_exact_hypervolume_2d_3d_4d_analytical(self):
        # 2D analytical union: [1, 5]x[4, 5] (area 4) U [3, 5]x[2, 5] (area 6), overlap [3,5]x[4,5] (area 2) => 8
        pts_2d = np.array([[1.0, 4.0], [3.0, 2.0]])
        ref_2d = np.array([5.0, 5.0])
        self.assertAlmostEqual(_hv_exact_recursive(pts_2d, ref_2d), 8.0, places=9)

        # 4D analytical inclusion-exclusion check for 2 overlapping hyper-boxes
        a = np.array([0.2, 0.4, 0.1, 0.5])
        b = np.array([0.5, 0.2, 0.3, 0.2])
        ref_4d = np.array([1.0, 1.0, 1.0, 1.0])
        vol_a = float(np.prod(ref_4d - a))
        vol_b = float(np.prod(ref_4d - b))
        vol_inter = float(np.prod(ref_4d - np.maximum(a, b)))
        expected_4d = vol_a + vol_b - vol_inter
        computed_4d = _hv_exact_recursive(np.vstack([a, b]), ref_4d)
        self.assertAlmostEqual(computed_4d, expected_4d, places=9)

    def test_07_nsga2_and_nsga3_pareto_validity_and_warm_start(self):
        init_sched, _, _ = run_genetic_algorithm(self.ft06, pop_size=40, generations=50, seed=42)
        disruption = BreakdownDisruption(machine_id=2, breakdown_start=20, duration=15)
        state = build_disrupted_state(init_sched, disruption, self.ft06)
        rs_sched = apply_right_shift(init_sched, disruption, self.ft06)
        rs_instab = rs_sched.compute_instability(init_sched, 20)

        rps = generate_reference_points(4, 5)
        self.assertEqual(len(rps), 56)  # binom(4 + 5 - 1, 5) = 56

        scheds2, objs2 = run_nsga2(
            self.ft06, baseline_schedule=init_sched, pop_size=36, generations=25,
            disrupted_state=state, include_energy=True, warm_start=True, seed=7,
        )
        scheds3, objs3 = run_nsga3(
            self.ft06, baseline_schedule=init_sched, pop_size=36, generations=25,
            num_divisions=3, disrupted_state=state, include_energy=True, warm_start=True, seed=7,
        )

        for s in scheds2 + scheds3:
            self.assertTrue(s.validate())

        # Returned fronts must be strictly mutually non-dominated
        self.assertEqual(len(extract_nondominated(objs2)), len(objs2))
        self.assertEqual(len(extract_nondominated(objs3)), len(objs3))

        # Warm-started fronts must achieve instability <= Right-Shift instability
        self.assertLessEqual(float(np.min(objs2[:, 2])), rs_instab)
        self.assertLessEqual(float(np.min(objs3[:, 2])), rs_instab)

    def test_08_igd_and_wilcoxon_metrics(self):
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        ideal = np.array([0.0, 0.0])
        nadir = np.array([1.0, 1.0])
        self.assertAlmostEqual(compute_igd(ref, ref, ideal, nadir), 0.0, places=9)
        self.assertAlmostEqual(compute_hypervolume(np.array([[0.0, 0.0]]), ideal, nadir), 1.0, places=9)

        stat = wilcoxon_rank_sum([0.8, 0.82, 0.85, 0.81, 0.84], [0.5, 0.52, 0.49, 0.51, 0.53], higher_is_better=True)
        self.assertTrue(stat["significant"])
        self.assertEqual(stat["symbol"], "+")


if __name__ == "__main__":
    unittest.main()

