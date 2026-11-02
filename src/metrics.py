"""
metrics.py: Publication-grade Multi-Objective Quality Indicators and Non-Parametric
Statistical Hypothesis Testing for MOEA evaluation (IEEE TEVC / IJPR standards).

Implements:
  - Exact Dimension-Sweep Hypervolume (HV) for 2D, 3D, and 4D Pareto fronts
  - Inverted Generational Distance (IGD) relative to the empirical true Pareto front P*
  - Schott's Spacing Metric (SP) for front uniformity
  - Wilcoxon Rank-Sum (Mann-Whitney U) two-sided statistical significance test
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy import stats


def extract_nondominated(points: np.ndarray) -> np.ndarray:
    """
    Given an (N, M) array of objective vectors (minimization), removes duplicates
    and returns only the non-dominated Pareto-optimal rows.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 0), dtype=float)

    pts = np.unique(np.round(np.asarray(points, dtype=float), 6), axis=0)
    dom_matrix = np.all(pts[:, None, :] <= pts[None, :, :], axis=2) & np.any(
        pts[:, None, :] < pts[None, :, :], axis=2
    )
    is_nondom = np.sum(dom_matrix, axis=0) == 0
    return pts[is_nondom]


def normalize_points(
    points: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> np.ndarray:
    """Normalizes objective vectors into [0, 1]^M given ideal and nadir bounds."""
    span = np.where(nadir - ideal > 1e-9, nadir - ideal, 1.0)
    return (np.asarray(points, dtype=float) - ideal[None, :]) / span[None, :]


def _hv_exact_recursive(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact dimension-sweep Hypervolume for minimization problems (arbitrary M >= 1)."""
    valid = points[np.all(points < ref[None, :], axis=1)]
    if len(valid) == 0:
        return 0.0

    m = valid.shape[1]
    if m == 1:
        return float(ref[0] - np.min(valid[:, 0]))

    if m == 2:
        order = np.lexsort((valid[:, 1], valid[:, 0]))
        pts = valid[order]
        area = 0.0
        min_y = float(ref[1])
        ref_x = float(ref[0])
        for x, y in pts:
            if y < min_y:
                area += (ref_x - x) * (min_y - y)
                min_y = float(y)
        return area

    # Dimension sweep along last axis m-1
    order = np.argsort(valid[:, -1], kind="mergesort")
    pts = valid[order]
    z = pts[:, -1]
    k = len(pts)
    volume = 0.0
    sub_ref = ref[:-1]

    for i in range(k):
        if i + 1 < k and z[i] == z[i + 1]:
            continue
        z_next = float(z[i + 1]) if i + 1 < k else float(ref[-1])
        dz = z_next - float(z[i])
        if dz > 0.0:
            slice_hv = _hv_exact_recursive(pts[: i + 1, :-1], sub_ref)
            volume += dz * slice_hv

    return volume


def compute_hypervolume(
    front_objs: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    ref_margin: float = 1.1,
) -> float:
    """
    Computes exact Normalized Hypervolume (HV in [0, 1]) after normalizing `front_objs`
    by [ideal, nadir] and using reference point (ref_margin, ..., ref_margin)^M.
    Higher HV indicates simultaneously superior convergence and diversity.
    """
    if front_objs is None or len(front_objs) == 0:
        return 0.0

    nd_pts = extract_nondominated(front_objs)
    norm_pts = normalize_points(nd_pts, ideal, nadir)
    m = norm_pts.shape[1]
    ref = np.full(m, ref_margin, dtype=float)
    raw_hv = _hv_exact_recursive(norm_pts, ref)
    max_possible_hv = float(ref_margin ** m)
    return raw_hv / max_possible_hv


def compute_igd(
    front_objs: np.ndarray,
    reference_front: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> float:
    """
    Computes normalized Inverted Generational Distance (IGD):
      IGD(A, P*) = (1 / |P*|) * sum_{v in P*} min_{a in A} ||v_norm - a_norm||_2
    Lower IGD indicates closer proximity and more complete coverage of the true Pareto front P*.
    """
    if front_objs is None or len(front_objs) == 0 or len(reference_front) == 0:
        return float("inf")

    norm_a = normalize_points(front_objs, ideal, nadir)
    norm_ref = normalize_points(reference_front, ideal, nadir)

    # Pairwise Euclidean distances: (|P*|, |A|)
    dists = np.linalg.norm(norm_ref[:, None, :] - norm_a[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    return float(np.mean(min_dists))


def compute_spacing(
    front_objs: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> float:
    """
    Computes Schott's Spacing Metric (SP) on normalized non-dominated solutions:
      SP = sqrt( (1 / (|A| - 1)) * sum_i (d_i - d_mean)^2 )
    where d_i is the L1 distance to the nearest neighbor in A. Lower SP means more uniform spread.
    """
    nd_pts = extract_nondominated(front_objs)
    if len(nd_pts) < 2:
        return 0.0

    norm_a = normalize_points(nd_pts, ideal, nadir)
    l1_dists = np.sum(np.abs(norm_a[:, None, :] - norm_a[None, :, :]), axis=2)
    np.fill_diagonal(l1_dists, float("inf"))
    nn_dists = np.min(l1_dists, axis=1)
    d_mean = float(np.mean(nn_dists))
    return float(np.sqrt(np.sum((nn_dists - d_mean) ** 2) / (len(nd_pts) - 1)))


def wilcoxon_rank_sum(
    sample_a: List[float],
    sample_b: List[float],
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Performs two-sided Mann-Whitney U / Wilcoxon Rank-Sum test between independent seed runs
    of Algorithm A and Algorithm B.
    Returns dict with p_value and significance symbol ('+' A better, '-' A worse, '≈' tie).
    """
    arr_a = np.asarray(sample_a, dtype=float)
    arr_b = np.asarray(sample_b, dtype=float)

    if len(arr_a) < 2 or len(arr_b) < 2 or np.allclose(arr_a, arr_b):
        return {"p_value": 1.0, "symbol": "≈", "significant": False}

    stat_res = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    p_val = float(stat_res.pvalue)
    sig = p_val < alpha

    if not sig:
        symbol = "≈"
    else:
        mean_a, mean_b = float(np.mean(arr_a)), float(np.mean(arr_b))
        if higher_is_better:
            symbol = "+" if mean_a > mean_b else "-"
        else:
            symbol = "+" if mean_a < mean_b else "-"

    return {"p_value": p_val, "symbol": symbol, "significant": sig}

