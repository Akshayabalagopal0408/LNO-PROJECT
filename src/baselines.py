"""
baselines.py: Classical Priority Dispatching Rules (FCFS, SPT, EDD) implemented via the
Giffler & Thompson (1960) Active Schedule Generation Scheme.

Supports both:
  - Pre-Disruption static dispatching (from t = 0)
  - Post-Disruption reactive dispatching (from DisruptedState at t = t_d, preserving frozen ops)
"""

from typing import List, Dict, Tuple, Optional, Callable
import bisect
from src.problem import JSPInstance, Operation
from src.schedule import Schedule, ScheduledOperation, find_earliest_slot
from src.disruption import DisruptedState


def _dispatch_giffler_thompson(
    instance: JSPInstance,
    priority_key: Callable[[int, Operation, int, int], Tuple],
    disrupted_state: Optional[DisruptedState] = None,
    initial_machine_avail: Optional[Dict[int, int]] = None,
    initial_job_avail: Optional[Dict[int, int]] = None,
    initial_op_ptr: Optional[Dict[int, int]] = None,
) -> Schedule:
    """
    Core Giffler & Thompson (1960) active dispatcher parameterized by `priority_key`.
    `priority_key(job_id, op, arrival_time, est_start)` returns a tuple minimized over conflict set C.
    """
    occupied: Dict[int, List[Tuple[int, int]]] = {m: [] for m in range(instance.num_machines)}
    unavail_copy: Dict[int, List[Tuple[int, int]]] = {m: [] for m in range(instance.num_machines)}
    j_ready = {job.job_id: job.release_time for job in instance.jobs}
    op_ptr = {j: 0 for j in range(instance.num_jobs)}
    scheduled_ops: List[ScheduledOperation] = []
    min_start_time = 0

    if disrupted_state is not None:
        min_start_time = disrupted_state.min_start_time
        for m, windows in disrupted_state.unavailable_windows.items():
            unavail_copy[m] = list(windows)
            for w in windows:
                bisect.insort(occupied[m], w)
        for f_op in disrupted_state.frozen_ops:
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
    else:
        if initial_machine_avail:
            for m, ready_t in initial_machine_avail.items():
                if ready_t > 0:
                    bisect.insort(occupied[m], (0, ready_t))
        if initial_job_avail:
            for j, ready_t in initial_job_avail.items():
                if ready_t > j_ready[j]:
                    j_ready[j] = ready_t
        if initial_op_ptr:
            op_ptr.update(initial_op_ptr)

    remaining_ops = instance.total_operations - sum(op_ptr.values())

    for _ in range(remaining_ops):
        # 1. Gather next schedulable operation for each incomplete job
        # Tuple: (job_id, op, arrival_time, est_start, est_end)
        schedulable: List[Tuple[int, Operation, int, int, int]] = []
        for j_id in range(instance.num_jobs):
            idx = op_ptr[j_id]
            if idx < len(instance.jobs[j_id].operations):
                op = instance.jobs[j_id].operations[idx]
                arrival_t = j_ready[j_id]
                earliest_req = max(arrival_t, min_start_time)
                est_start = find_earliest_slot(occupied[op.machine_id], earliest_req, op.duration)
                est_end = est_start + op.duration
                schedulable.append((j_id, op, arrival_t, est_start, est_end))

        if not schedulable:
            break

        # 2. Identify critical operation with minimum earliest completion time c*
        crit = min(schedulable, key=lambda c: (c[4], c[3], c[0]))
        m_star = crit[1].machine_id
        c_star = crit[4]

        # 3. Build Giffler-Thompson conflict set on machine m*
        conflict_set = [
            c for c in schedulable
            if c[1].machine_id == m_star and (c[3] < c_star or c[0] == crit[0])
        ]

        # 4. Dispatch highest-priority operation in conflict set
        winner = min(conflict_set, key=lambda c: priority_key(c[0], c[1], c[2], c[3]))
        w_job, w_op, _, w_start, w_end = winner

        bisect.insort(occupied[m_star], (w_start, w_end))
        j_ready[w_job] = w_end
        op_ptr[w_job] += 1

        scheduled_ops.append(
            ScheduledOperation(
                job_id=w_job,
                op_idx=w_op.op_idx,
                machine_id=m_star,
                start_time=w_start,
                end_time=w_end,
                duration=w_op.duration,
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


def solve_fcfs(
    instance: JSPInstance,
    initial_machine_avail: Optional[Dict[int, int]] = None,
    initial_job_avail: Optional[Dict[int, int]] = None,
    initial_op_ptr: Optional[Dict[int, int]] = None,
    disrupted_state: Optional[DisruptedState] = None,
) -> Schedule:
    """
    First-Come, First-Served (FCFS) Active Dispatching Rule:
    Resolves machine conflicts in favor of the job that arrived earliest in queue.
    """
    return _dispatch_giffler_thompson(
        instance=instance,
        priority_key=lambda j_id, op, arr_t, est_s: (arr_t, est_s, j_id),
        disrupted_state=disrupted_state,
        initial_machine_avail=initial_machine_avail,
        initial_job_avail=initial_job_avail,
        initial_op_ptr=initial_op_ptr,
    )


def solve_spt(
    instance: JSPInstance,
    initial_machine_avail: Optional[Dict[int, int]] = None,
    initial_job_avail: Optional[Dict[int, int]] = None,
    initial_op_ptr: Optional[Dict[int, int]] = None,
    disrupted_state: Optional[DisruptedState] = None,
) -> Schedule:
    """
    Shortest Processing Time (SPT) Active Dispatching Rule:
    Resolves machine conflicts in favor of the shortest operation duration.
    """
    return _dispatch_giffler_thompson(
        instance=instance,
        priority_key=lambda j_id, op, arr_t, est_s: (op.duration, est_s, j_id),
        disrupted_state=disrupted_state,
        initial_machine_avail=initial_machine_avail,
        initial_job_avail=initial_job_avail,
        initial_op_ptr=initial_op_ptr,
    )


def solve_edd(
    instance: JSPInstance,
    disrupted_state: Optional[DisruptedState] = None,
) -> Schedule:
    """
    Earliest Due Date (EDD) Active Dispatching Rule:
    Resolves machine conflicts in favor of the job with the earliest due date.
    """
    return _dispatch_giffler_thompson(
        instance=instance,
        priority_key=lambda j_id, op, arr_t, est_s: (instance.jobs[j_id].due_date, op.duration, j_id),
        disrupted_state=disrupted_state,
    )
