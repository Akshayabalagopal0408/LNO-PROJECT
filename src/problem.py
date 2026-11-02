"""
problem.py: Core data models and instance loader for Job Shop Scheduling Problems (JSSP).
Supports standard OR-Library / JSPLIB formats, Total Work Content (TWK) due dates,
and heterogeneous machine power profiles for Green / Energy-Aware Scheduling.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import os


@dataclass
class Operation:
    job_id: int
    op_idx: int
    machine_id: int
    duration: int

    def __repr__(self):
        return f"Op(J{self.job_id}, #{self.op_idx}, M{self.machine_id}, dur={self.duration})"


@dataclass
class Job:
    job_id: int
    operations: List[Operation] = field(default_factory=list)
    due_date: int = 0
    release_time: int = 0

    @property
    def total_processing_time(self) -> int:
        return sum(op.duration for op in self.operations)


@dataclass
class JSPInstance:
    name: str
    num_jobs: int
    num_machines: int
    jobs: List[Job] = field(default_factory=list)
    # Heterogeneous machine power parameters (in kW and kW*time_unit) per Mouzon et al. (2007) / Dai et al. (2013)
    machine_work_power: Dict[int, float] = field(default_factory=dict)
    machine_idle_power: Dict[int, float] = field(default_factory=dict)
    machine_standby_power: Dict[int, float] = field(default_factory=dict)
    machine_switch_energy: Dict[int, float] = field(default_factory=dict)

    @property
    def total_operations(self) -> int:
        return sum(len(job.operations) for job in self.jobs)

    def assign_due_dates(self, tightness_factor: float = 1.5, min_due_date: int = 0):
        """
        Total Work Content (TWK) rule: Due date = release_time + k * total_processing_time.
        Standard academic formula for introducing tardiness in JSSP benchmarks.
        """
        for job in self.jobs:
            raw_due = int(round(job.release_time + tightness_factor * job.total_processing_time))
            job.due_date = max(min_due_date, raw_due)

    def assign_machine_power_profiles(self):
        """
        Assigns deterministic, heterogeneous machine power profiles representing distinct
        shop-floor machine classes (e.g., high-power CNC milling/turning vs. light finishing).
        Includes turn-off/turn-on setup energy (Mouzon et al., 2007; Dai et al., 2013) so
        consolidating idle intervals beyond the breakeven threshold saves idle energy.
        """
        # Archetypes: (work_kW, idle_kW, standby_kW, switch_energy_kW_time)
        archetypes = [
            (6.5, 2.2, 0.15, 14.0),  # Heavy CNC Machining Center (high idle draw)
            (4.2, 0.9, 0.10, 6.0),   # Precision Lathe
            (7.8, 2.6, 0.20, 18.0),  # Thermal / Grinding Station (very high idle draw)
            (3.5, 0.6, 0.08, 4.0),   # Light Drilling / Assembly
            (5.4, 1.6, 0.12, 10.0),  # Multi-Axis Milling
        ]
        for m in range(self.num_machines):
            work_p, idle_p, stby_p, sw_e = archetypes[m % len(archetypes)]
            self.machine_work_power[m] = work_p
            self.machine_idle_power[m] = idle_p
            self.machine_standby_power[m] = stby_p
            self.machine_switch_energy[m] = sw_e

    def get_bottleneck_machine(self) -> int:
        """Returns the machine_id with the highest total processing workload."""
        loads = {m: 0 for m in range(self.num_machines)}
        for job in self.jobs:
            for op in job.operations:
                loads[op.machine_id] += op.duration
        return max(loads, key=lambda m: (loads[m], -m))


def load_jsp_instance(file_path: str, tightness_factor: float = 1.5) -> JSPInstance:
    """
    Parses standard JSPLib / OR-Library JSSP benchmark text files.
    Format:
      # Comments starting with #
      num_jobs num_machines
      For each job row: pairs of (machine_id, duration)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    instance_name = os.path.splitext(os.path.basename(file_path))[0]
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)

    if not lines:
        raise ValueError(f"No valid data lines found in {file_path}")

    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1])

    jobs = []
    job_lines = lines[1 : num_jobs + 1]

    for j_id, line in enumerate(job_lines):
        tokens = [int(tok) for tok in line.split()]
        ops = []
        op_idx = 0
        for i in range(0, len(tokens), 2):
            m_id = tokens[i]
            dur = tokens[i + 1]
            ops.append(Operation(job_id=j_id, op_idx=op_idx, machine_id=m_id, duration=dur))
            op_idx += 1
        jobs.append(Job(job_id=j_id, operations=ops))

    instance = JSPInstance(
        name=instance_name,
        num_jobs=num_jobs,
        num_machines=num_machines,
        jobs=jobs,
    )
    instance.assign_due_dates(tightness_factor=tightness_factor)
    instance.assign_machine_power_profiles()
    return instance
