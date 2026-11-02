"""
schedule.py: Schedule representation, multi-objective evaluation, and True Active
Operation-Based Chromosome Decoder (Bierwirth, 1995; Giffler & Thompson, 1960).

Guarantees strictly feasible, deadlock-free ACTIVE schedules via earliest idle-interval
insertion (left-shifting), while respecting frozen in-progress/completed operations,
machine breakdown windows, and non-retroactive rescheduling epochs (t >= t_d).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import bisect
import numpy as np
from src.problem import JSPInstance


@dataclass
class ScheduledOperation:
    job_id: int
    op_idx: int
    machine_id: int
    start_time: int
    end_time: int
    duration: int
    is_frozen: bool = False

    @property
    def op_key(self) -> Tuple[int, int]:
        return (self.job_id, self.op_idx)


def find_earliest_slot(
    occupied_intervals: List[Tuple[int, int]],
    earliest_start: int,
    duration: int,
) -> int:
    """
    Scans sorted, non-overlapping occupied intervals on a machine to find the
    earliest feasible start time t >= earliest_start such that [t, t + duration]
    intersects no occupied or breakdown interval (Active Left-Shift).
    """
    t = earliest_start
    for iv_start, iv_end in occupied_intervals:
        if t + duration <= iv_start:
            break
        if t < iv_end:
            t = iv_end
    return t


@dataclass
class Schedule:
    instance: JSPInstance
    operations: List[ScheduledOperation] = field(default_factory=list)
    unavailable_windows: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    machine_schedules: Dict[int, List[ScheduledOperation]] = field(default_factory=dict)
    job_completion_times: Dict[int, int] = field(default_factory=dict)
    op_lookup: Dict[Tuple[int, int], ScheduledOperation] = field(default_factory=dict)

    def finalize(self):
        """Build index maps and calculate per-job completion times."""
        self.machine_schedules = {m: [] for m in range(self.instance.num_machines)}
        self.op_lookup = {}
        self.job_completion_times = {j: 0 for j in range(self.instance.num_jobs)}

        for op in self.operations:
            self.machine_schedules[op.machine_id].append(op)
            self.op_lookup[op.op_key] = op
            if op.end_time > self.job_completion_times[op.job_id]:
                self.job_completion_times[op.job_id] = op.end_time

        for m in self.machine_schedules:
            self.machine_schedules[m].sort(key=lambda x: (x.start_time, x.end_time))

    @property
    def makespan(self) -> int:
        if not self.operations:
            return 0
        return max(op.end_time for op in self.operations)

    @property
    def total_tardiness(self) -> int:
        """Sum of max(0, C_i - d_i) across all jobs."""
        tardiness = 0
        for job in self.instance.jobs:
            c_i = self.job_completion_times.get(job.job_id, 0)
            tardiness += max(0, c_i - job.due_date)
        return tardiness

    @property
    def total_energy_consumption(self) -> float:
        """
        Total Energy Consumption (TEC in kWh) - 4th Objective for Green Scheduling.
        Based on Mouzon et al. (2007) and Dai et al. (2013):
          - Heterogeneous active processing power P_work[m] and idle power P_idle[m].
          - Optimal turn-off / standby policy on idle gaps: if an idle gap exceeds the
            breakeven threshold, the machine switches to standby mode (E_switch + P_standby * gap).
          - Offline breakdown windows do NOT consume idle power.
          - Time scale: 1 scheduling time unit = 0.1 hours (6 minutes).
        """
        TIME_TO_HOURS = 0.1
        total_energy_kwh = 0.0

        for m, ops in self.machine_schedules.items():
            if not ops:
                continue

            p_work = self.instance.machine_work_power.get(m, 5.0)
            p_idle = self.instance.machine_idle_power.get(m, 1.2)
            p_stby = self.instance.machine_standby_power.get(m, 0.1)
            e_switch = self.instance.machine_switch_energy.get(m, 8.0)

            m_completion = max(op.end_time for op in ops)
            work_time = sum(op.duration for op in ops)
            machine_energy = p_work * work_time

            # Build busy/offline intervals in [0, m_completion] to extract true idle gaps
            busy_blocks: List[Tuple[int, int]] = [(op.start_time, op.end_time) for op in ops]
            for w_start, w_end in self.unavailable_windows.get(m, []):
                clipped_start = max(0, min(m_completion, w_start))
                clipped_end = max(0, min(m_completion, w_end))
                if clipped_end > clipped_start:
                    busy_blocks.append((clipped_start, clipped_end))

            busy_blocks.sort(key=lambda x: (x[0], x[1]))

            # Merge overlapping/adjacent busy or offline intervals and evaluate idle gaps
            cursor = 0
            for b_start, b_end in busy_blocks:
                if b_start > cursor:
                    gap = b_start - cursor
                    idle_cost = p_idle * gap
                    standby_cost = e_switch + p_stby * gap
                    machine_energy += min(idle_cost, standby_cost)
                if b_end > cursor:
                    cursor = b_end

            total_energy_kwh += machine_energy * TIME_TO_HOURS

        return round(total_energy_kwh, 2)

    def compute_instability(self, baseline_schedule: Optional["Schedule"], impacted_only_since: int = 0) -> int:
        """
        Calculates Schedule Start-Time Instability (Nervousness, Delta S) relative to baseline:
        Sum of |start_time_new - start_time_baseline| across all operations not completed before t_d.
        (Frozen operations have identical start times and contribute 0.)
        """
        if baseline_schedule is None or not baseline_schedule.operations:
            return 0

        instability = 0
        for key, new_op in self.op_lookup.items():
            if key in baseline_schedule.op_lookup:
                base_op = baseline_schedule.op_lookup[key]
                if base_op.end_time > impacted_only_since:
                    instability += abs(new_op.start_time - base_op.start_time)
        return instability

    def compute_sequence_instability(self, baseline_schedule: Optional["Schedule"]) -> int:
        """
        Calculates Machine Sequence Instability: number of pairwise job-order inversions
        across all machines compared to baseline_schedule. (Right-Shift guarantees 0.)
        """
        if baseline_schedule is None or not baseline_schedule.operations:
            return 0

        inversions = 0
        for m in range(self.instance.num_machines):
            base_seq = [op.op_key for op in baseline_schedule.machine_schedules.get(m, [])]
            base_pos = {key: idx for idx, key in enumerate(base_seq)}
            new_seq = [op.op_key for op in self.machine_schedules.get(m, []) if op.op_key in base_pos]
            for i in range(len(new_seq)):
                for j in range(i + 1, len(new_seq)):
                    if base_pos[new_seq[i]] > base_pos[new_seq[j]]:
                        inversions += 1
        return inversions

    def validate(self) -> bool:
        """Verifies schedule completeness, job precedence, non-overlap, and breakdown compliance."""
        if len(self.operations) != self.instance.total_operations:
            return False

        # Check job precedence
        for job in self.instance.jobs:
            prev_end = job.release_time
            for op in job.operations:
                sched_op = self.op_lookup.get((job.job_id, op.op_idx))
                if sched_op is None:
                    return False
                if sched_op.end_time - sched_op.start_time != op.duration:
                    return False
                if sched_op.start_time < prev_end:
                    return False
                prev_end = sched_op.end_time

        # Check machine non-overlap and breakdown windows
        for m, ops in self.machine_schedules.items():
            for i in range(len(ops) - 1):
                if ops[i].end_time > ops[i + 1].start_time:
                    return False
            for w_start, w_end in self.unavailable_windows.get(m, []):
                for op in ops:
                    if op.start_time < w_end and op.end_time > w_start:
                        return False

        return True


def decode_chromosome(
    chromosome: List[int],
    instance: JSPInstance,
    machine_ready_times: Optional[Dict[int, int]] = None,
    job_ready_times: Optional[Dict[int, int]] = None,
    job_op_offsets: Optional[Dict[int, int]] = None,
    frozen_ops: Optional[List[ScheduledOperation]] = None,
    unavailable_windows: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    min_start_time: int = 0,
    active: bool = True,
) -> Schedule:
    """
    Decodes an operation-based chromosome (Bierwirth permutation with repetition) into
    a complete, strictly valid Schedule.

    When active=True (default), performs earliest idle-gap insertion (Active Schedule Decoding),
    guaranteeing the optimal schedule remains reachable in the search space.
    Preserves `frozen_ops` and respects `unavailable_windows` (machine breakdowns) as well
    as `min_start_time` (rescheduling decision epoch t_d).
    """
    occupied: Dict[int, List[Tuple[int, int]]] = {m: [] for m in range(instance.num_machines)}
    unavail_copy: Dict[int, List[Tuple[int, int]]] = {
        m: list(unavailable_windows.get(m, [])) if unavailable_windows else []
        for m in range(instance.num_machines)
    }

    for m, windows in unavail_copy.items():
        for w in windows:
            bisect.insort(occupied[m], w)

    j_ready = {job.job_id: job.release_time for job in instance.jobs}
    op_ptr = {j: 0 for j in range(instance.num_jobs)}

    scheduled_ops: List[ScheduledOperation] = []

    if frozen_ops:
        for f_op in frozen_ops:
            copied_op = ScheduledOperation(
                job_id=f_op.job_id,
                op_idx=f_op.op_idx,
                machine_id=f_op.machine_id,
                start_time=f_op.start_time,
                end_time=f_op.end_time,
                duration=f_op.duration,
                is_frozen=True,
            )
            scheduled_ops.append(copied_op)
            bisect.insort(occupied[f_op.machine_id], (f_op.start_time, f_op.end_time))
            if f_op.end_time > j_ready[f_op.job_id]:
                j_ready[f_op.job_id] = f_op.end_time
            if f_op.op_idx + 1 > op_ptr[f_op.job_id]:
                op_ptr[f_op.job_id] = f_op.op_idx + 1

    # Backward-compatible overrides when called without frozen_ops
    if machine_ready_times and not frozen_ops:
        for m, ready_t in machine_ready_times.items():
            if ready_t > 0:
                bisect.insort(occupied[m], (0, ready_t))
    if job_ready_times:
        for j, ready_t in job_ready_times.items():
            if ready_t > j_ready[j]:
                j_ready[j] = ready_t
    if job_op_offsets:
        for j, off in job_op_offsets.items():
            if off > op_ptr[j]:
                op_ptr[j] = off

    jobs_list = instance.jobs

    for job_id in chromosome:
        current_op_idx = op_ptr[job_id]
        job = jobs_list[job_id]

        if current_op_idx >= len(job.operations):
            continue

        op = job.operations[current_op_idx]
        m_id = op.machine_id
        dur = op.duration
        earliest_t = j_ready[job_id]
        if min_start_time > earliest_t:
            earliest_t = min_start_time

        m_intervals = occupied[m_id]
        if active:
            start_time = find_earliest_slot(m_intervals, earliest_t, dur)
        else:
            last_end = m_intervals[-1][1] if m_intervals else 0
            start_time = max(last_end, earliest_t)

        end_time = start_time + dur

        bisect.insort(m_intervals, (start_time, end_time))
        j_ready[job_id] = end_time
        op_ptr[job_id] = current_op_idx + 1

        scheduled_ops.append(
            ScheduledOperation(
                job_id=job_id,
                op_idx=current_op_idx,
                machine_id=m_id,
                start_time=start_time,
                end_time=end_time,
                duration=dur,
                is_frozen=False,
            )
        )

    sched = Schedule(
        instance=instance,
        operations=scheduled_ops,
        unavailable_windows=unavail_copy,
    )
    sched.finalize()
    return sched


def create_random_chromosome(instance: JSPInstance) -> List[int]:
    """Generates a random valid operation-based chromosome."""
    chrom = []
    for job in instance.jobs:
        chrom.extend([job.job_id] * len(job.operations))
    np.random.shuffle(chrom)
    return chrom
