"""
disruption.py: Models machine breakdown disruptions, DisruptedState partitioning,
and the Right-Shift Rescheduling baseline.
Implements predictive-reactive rescheduling mechanics formulated by Vieira et al. (2003)
and Xiong et al. (2013) under consistent physical shop-floor rules:
  1. Operations completed at or before t_d (end_time <= t_d) are frozen.
  2. Operations in-progress at t_d on healthy machines (m != m_break, start_time < t_d < end_time)
     continue uninterrupted and are frozen.
  3. Any operation interrupted mid-processing on the failed machine (m == m_break,
     start_time < t_d < end_time) is aborted at t_d (preempt-repeat) and rescheduled after repair.
  4. No unfrozen rescheduled operation may start earlier than t_d.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from src.problem import JSPInstance
from src.schedule import Schedule, ScheduledOperation


@dataclass
class BreakdownDisruption:
    machine_id: int
    breakdown_start: int  # t_d
    duration: int         # D

    @property
    def breakdown_end(self) -> int:
        return self.breakdown_start + self.duration

    def __repr__(self):
        return (
            f"Breakdown(M{self.machine_id} offline from t={self.breakdown_start} "
            f"to t={self.breakdown_end}, dur={self.duration})"
        )


@dataclass
class DisruptedState:
    """Encapsulates the exact shop-floor state at disruption time t_d."""
    disruption: BreakdownDisruption
    frozen_ops: List[ScheduledOperation]
    machine_ready_times: Dict[int, int]
    job_ready_times: Dict[int, int]
    job_op_offsets: Dict[int, int]
    remaining_chromosome_pool: List[int]
    unavailable_windows: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)

    @property
    def min_start_time(self) -> int:
        return self.disruption.breakdown_start


def build_disrupted_state(
    initial_schedule: Schedule,
    disruption: BreakdownDisruption,
    instance: JSPInstance,
) -> DisruptedState:
    """
    Constructs a physically consistent DisruptedState at t = disruption.breakdown_start.
    """
    t_d = disruption.breakdown_start
    m_break = disruption.machine_id

    frozen_ops: List[ScheduledOperation] = []
    job_op_offsets = {j: 0 for j in range(instance.num_jobs)}
    machine_ready = {m: t_d for m in range(instance.num_machines)}
    job_ready = {j: max(t_d, instance.jobs[j].release_time) for j in range(instance.num_jobs)}

    machine_ready[m_break] = disruption.breakdown_end
    unavailable_windows: Dict[int, List[Tuple[int, int]]] = {
        m: [] for m in range(instance.num_machines)
    }
    unavailable_windows[m_break].append((disruption.breakdown_start, disruption.breakdown_end))

    # Identify frozen operations in job precedence order
    for job in instance.jobs:
        j_id = job.job_id
        for op in job.operations:
            sched_op = initial_schedule.op_lookup.get((j_id, op.op_idx))
            if sched_op is None:
                break

            completed_before_td = sched_op.end_time <= t_d
            running_on_healthy = (
                sched_op.machine_id != m_break
                and sched_op.start_time < t_d < sched_op.end_time
            )

            if completed_before_td or running_on_healthy:
                frozen_copy = ScheduledOperation(
                    job_id=sched_op.job_id,
                    op_idx=sched_op.op_idx,
                    machine_id=sched_op.machine_id,
                    start_time=sched_op.start_time,
                    end_time=sched_op.end_time,
                    duration=sched_op.duration,
                    is_frozen=True,
                )
                frozen_ops.append(frozen_copy)
                job_op_offsets[j_id] = op.op_idx + 1
                if sched_op.end_time > machine_ready[sched_op.machine_id]:
                    machine_ready[sched_op.machine_id] = sched_op.end_time
                if sched_op.end_time > job_ready[j_id]:
                    job_ready[j_id] = sched_op.end_time
            else:
                # Subsequent operations of this job cannot have started at or before t_d
                break

    frozen_ops.sort(key=lambda x: (x.start_time, x.end_time, x.job_id))

    remaining_chrom_pool: List[int] = []
    for job in instance.jobs:
        rem_count = len(job.operations) - job_op_offsets[job.job_id]
        remaining_chrom_pool.extend([job.job_id] * rem_count)

    return DisruptedState(
        disruption=disruption,
        frozen_ops=frozen_ops,
        machine_ready_times=machine_ready,
        job_ready_times=job_ready,
        job_op_offsets=job_op_offsets,
        remaining_chromosome_pool=remaining_chrom_pool,
        unavailable_windows=unavailable_windows,
    )


def partition_disrupted_state(
    initial_schedule: Schedule,
    disruption: BreakdownDisruption,
    instance: JSPInstance,
) -> Tuple[List[ScheduledOperation], Dict[int, int], Dict[int, int], Dict[int, int], List[int]]:
    """
    Partitions the schedule at t = breakdown_start (returns 5-tuple for API compatibility):
      - frozen_ops: operations completed <= t_d or mid-execution on healthy machines across t_d
      - machine_ready_times: machine availability after frozen ops and breakdown repair
      - job_ready_times: job availability for remaining operations (>= t_d)
      - job_op_offsets: next operation index to schedule per job
      - remaining_chromosome_pool: multiset of job_ids remaining to be scheduled
    """
    state = build_disrupted_state(initial_schedule, disruption, instance)
    return (
        state.frozen_ops,
        state.machine_ready_times,
        state.job_ready_times,
        state.job_op_offsets,
        state.remaining_chromosome_pool,
    )


def apply_right_shift(
    initial_schedule: Schedule,
    disruption: BreakdownDisruption,
    instance: JSPInstance,
) -> Schedule:
    """
    Implements Right-Shift Rescheduling (the standard sequence-preserving reactive baseline):
    Preserves the EXACT operation sequence on all machines from `initial_schedule`,
    shifting unfrozen operations rightwards to accommodate the machine breakdown window
    and cascading precedence delays. Guarantees 0 machine sequence inversions.
    """
    state = build_disrupted_state(initial_schedule, disruption, instance)
    frozen_keys = {op.op_key for op in state.frozen_ops}

    # Topological order of initial_schedule (preserves both job and machine sequences)
    sorted_ops = sorted(
        initial_schedule.operations,
        key=lambda x: (x.start_time, x.op_idx, x.job_id),
    )

    new_ops: List[ScheduledOperation] = []
    machine_avail = {m: 0 for m in range(instance.num_machines)}
    job_avail = {j: instance.jobs[j].release_time for j in range(instance.num_jobs)}

    t_d = disruption.breakdown_start
    t_end = disruption.breakdown_end
    m_break = disruption.machine_id

    for op in sorted_ops:
        m = op.machine_id
        j = op.job_id
        dur = op.duration

        if op.op_key in frozen_keys:
            start_t = op.start_time
            end_t = op.end_time
            is_frozen = True
        else:
            earliest_t = max(machine_avail[m], job_avail[j], t_d)
            if m == m_break and earliest_t < t_end:
                start_t = t_end
            else:
                start_t = earliest_t
            end_t = start_t + dur
            is_frozen = False

        if end_t > machine_avail[m]:
            machine_avail[m] = end_t
        if end_t > job_avail[j]:
            job_avail[j] = end_t

        new_ops.append(
            ScheduledOperation(
                job_id=j,
                op_idx=op.op_idx,
                machine_id=m,
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                is_frozen=is_frozen,
            )
        )

    rescheduled = Schedule(
        instance=instance,
        operations=new_ops,
        unavailable_windows=state.unavailable_windows,
    )
    rescheduled.finalize()
    return rescheduled


def extract_warm_start_chromosomes(
    initial_schedule: Schedule,
    disruption: BreakdownDisruption,
    instance: JSPInstance,
) -> List[List[int]]:
    """
    Extracts baseline-preserving remaining operation chromosomes from both the
    initial predictive schedule and the Right-Shift schedule to warm-start MOEAs.
    """
    state = build_disrupted_state(initial_schedule, disruption, instance)
    frozen_keys = {op.op_key for op in state.frozen_ops}

    init_rem = [
        op.job_id
        for op in sorted(initial_schedule.operations, key=lambda x: (x.start_time, x.op_idx, x.job_id))
        if op.op_key not in frozen_keys
    ]

    rs_sched = apply_right_shift(initial_schedule, disruption, instance)
    rs_rem = [
        op.job_id
        for op in sorted(rs_sched.operations, key=lambda x: (x.start_time, x.op_idx, x.job_id))
        if op.op_key not in frozen_keys
    ]

    seeds = []
    if init_rem:
        seeds.append(init_rem)
    if rs_rem and rs_rem != init_rem:
        seeds.append(rs_rem)
    return seeds
