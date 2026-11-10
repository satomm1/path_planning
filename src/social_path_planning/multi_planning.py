"""Dense-waypoint multi-robot timing MILP (sequential / simultaneous / combined).

Also hosts the segment-conflict detection and z-assignment helpers that the
event-waypoint planner (``event_multi_planning``) reuses.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from social_path_planning.mapf_comparison.motion import DEFAULT_MAX_VELOCITY_MPS

if TYPE_CHECKING:
    from social_path_planning.occupancy_grid import StochOccupancyGrid2D

NOMINAL_VELOCITY = 0.35  # m/s
TIME_STEP = 5  # seconds
MAX_VELOCITY = DEFAULT_MAX_VELOCITY_MPS
ROBOT_DIAMETER = 0.5  # meters
M = 60  # Big-M constant for constraints
DELTA = 1  # Safety margin in seconds
# Max factor k between consecutive segment speeds v_i = d_i / (t_{i+1} - t_i):
# v_{i+1} <= k * v_i and v_i <= k * v_{i+1} (linear in waypoint times).
MAX_VELOCITY_CHANGE_FACTOR = 1.5
GEOM_MATCH_TOL = 1e-3  # meters; tolerate grid / DDS float slop when matching vertices or edges

# Set True to save per-pair MILP z-assignment maps when MultiAgentSimultaneousPlanner.plan() runs.
# Off by default: the plots are written inside plan(), so leaving this on adds matplotlib
# time to every benchmarked dense-MILP solve.
VIZ_MILP_Z_ASSIGNMENT = False
VIZ_MILP_Z_OUTPUT_DIR = "results/milp_z_viz"


def _scalar_times_from_solver(prob, time_var, context):
    """Read one cvxpy time vector after ``prob.solve()``; raise if infeasible / no primal."""
    val = time_var.value
    if val is None:
        raise RuntimeError(
            f"{context}: no solution (cvxpy status={getattr(prob, 'status', None)!r}); likely infeasible or unbounded"
        )
    return val.tolist()


def _multi_agent_times_from_solver(prob, agent_times, context):
    """Read per-agent time vectors after ``prob.solve()``; raise if any primal is missing."""
    out = []
    for i, at in enumerate(agent_times):
        val = at.value
        if val is None:
            raise RuntimeError(
                f"{context}: no solution for agent {i} (cvxpy status={getattr(prob, 'status', None)!r}); likely infeasible or unbounded"
            )
        out.append(val.tolist())
    return out


def _segment_lengths(path):
    """Return Euclidean length d_i for each segment (waypoint i -> i+1)."""
    return [
        float(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])))
        for i in range(len(path) - 1)
    ]


def _add_velocity_change_constraints(constraints, time_var, path, k):
    """
    Limit consecutive segment speed changes with linear constraints (1-indexed i = 1..N-2):

      d_i * (t_{i+2} - t_{i+1}) - k * d_{i+1} * (t_{i+1} - t_i) <= 0
      d_{i+1} * (t_{i+1} - t_i) - k * d_i * (t_{i+2} - t_{i+1}) <= 0

    Equivalently bounds v_i = d_i / (t_{i+1}-t_i) so v_{i+1} <= k*v_i and v_i <= k*v_{i+1}.
    Long holds (large dt) are allowed; only relative speeds between adjacent segments are limited.
    """
    dists = _segment_lengths(path)
    if len(dists) < 2:
        return
    for i in range(len(dists) - 1):
        d_i = dists[i]
        d_ip1 = dists[i + 1]
        if d_i <= 0.0 and d_ip1 <= 0.0:
            continue
        dt_i = time_var[i + 1] - time_var[i]
        dt_ip1 = time_var[i + 2] - time_var[i + 1]
        constraints += [d_i * dt_ip1 - k * d_ip1 * dt_i <= 0]
        constraints += [d_ip1 * dt_i - k * d_i * dt_ip1 <= 0]


def _add_max_velocity_constraints(constraints, time_var, path, v):
    """
    Per-segment speed cap ``t_{i+1} - t_i >= |p_{i+1} - p_i| / v`` (which also keeps
    times non-decreasing), followed by the consecutive speed-change limits.
    """
    for i in range(len(path) - 1):
        delta_pos = np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
        constraints += [time_var[i + 1] - time_var[i] >= delta_pos / v]
    _add_velocity_change_constraints(constraints, time_var, path, MAX_VELOCITY_CHANGE_FACTOR)


def _segment_endpoint_conflict(p0, p1, q0, q1, threshold):
    """
    True if any endpoint of segment (p0, p1) is within threshold of any endpoint of (q0, q1).

    Endpoint-only C-space overlap; does not detect crossing segments whose interiors overlap
    while all four endpoints remain farther apart than threshold.
    """
    for pa in (p0, p1):
        for qb in (q0, q1):
            if np.linalg.norm(np.array(pa) - np.array(qb)) <= threshold:
                return True
    return False


def _points_close(a, b, tol=GEOM_MATCH_TOL):
    """True if two 2D points are within ``tol`` in Euclidean distance."""
    return float(np.linalg.norm(np.array(a) - np.array(b))) <= tol


def _point_to_segment_distance(point, seg_a, seg_b):
    """Perpendicular distance from ``point`` to segment ``seg_a -> seg_b``."""
    p = np.array(point, dtype=float)
    a = np.array(seg_a, dtype=float)
    b = np.array(seg_b, dtype=float)
    ab = b - a
    n = float(np.linalg.norm(ab))
    if n <= 1e-12:
        return float(np.linalg.norm(p - a))
    return float(np.abs((ab[0] * (a[1] - p[1])) - (a[0] - p[0]) * ab[1])) / n


def _segments_colinear(p0, p1, q0, q1, tol=GEOM_MATCH_TOL):
    """True when two segments lie on the same infinite line (within ``tol``)."""
    v1 = np.array(p1, dtype=float) - np.array(p0, dtype=float)
    v2 = np.array(q1, dtype=float) - np.array(q0, dtype=float)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= tol or n2 <= tol:
        return False
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    return cross <= tol * max(n1, n2)


def _segments_opposite_direction(p0, p1, q0, q1, tol=GEOM_MATCH_TOL):
    """True when segment directions differ by ~180 degrees."""
    v1 = np.array(p1, dtype=float) - np.array(p0, dtype=float)
    v2 = np.array(q1, dtype=float) - np.array(q0, dtype=float)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= tol or n2 <= tol:
        return False
    cos_angle = float(np.dot(v1, v2) / (n1 * n2))
    return cos_angle < -1.0 + tol


def _segments_overlap_on_axis(p0, p1, q0, q1, tol=GEOM_MATCH_TOL):
    """True when segment interiors overlap after projection onto the first segment axis."""
    v1 = np.array(p1, dtype=float) - np.array(p0, dtype=float)
    n1 = float(np.linalg.norm(v1))
    if n1 <= tol:
        return False
    u = v1 / n1
    origin = np.array(p0, dtype=float)

    def _proj(point):
        return float(np.dot(np.array(point, dtype=float) - origin, u))

    a0, a1 = _proj(p0), _proj(p1)
    b0, b1 = _proj(q0), _proj(q1)
    lo_a, hi_a = min(a0, a1), max(a0, a1)
    lo_b, hi_b = min(b0, b1), max(b0, b1)
    return min(hi_a, hi_b) - max(lo_a, lo_b) > tol


def _segments_opposite_same_edge(p0, p1, q0, q1, tol=GEOM_MATCH_TOL):
    """True when two segments share the same physical edge with opposite orientation."""
    return _points_close(p0, q1, tol) and _points_close(p1, q0, tol)


def _segments_opposite_for_coupling(p0, p1, q0, q1, tol=GEOM_MATCH_TOL):
    """True for opposite travel on the same corridor, including mismatched segment lengths."""
    if _segments_opposite_same_edge(p0, p1, q0, q1, tol=tol):
        return True
    if not (
        _segments_colinear(p0, p1, q0, q1, tol=tol)
        and _segments_opposite_direction(p0, p1, q0, q1, tol=tol)
        and _segments_overlap_on_axis(p0, p1, q0, q1, tol=tol)
    ):
        return False
    for qpt in (q0, q1):
        if _point_to_segment_distance(qpt, p0, p1) > tol:
            return False
    return True


def _tag_opposite_segment_pairs(path_a, path_b, segment_pairs, tol=GEOM_MATCH_TOL):
    """Return the subset of ``segment_pairs`` that are opposite-direction corridor conflicts."""
    return {
        (int(i), int(j))
        for (i, j) in segment_pairs
        if _segments_opposite_for_coupling(
            path_a[int(i)], path_a[int(i) + 1], path_b[int(j)], path_b[int(j) + 1], tol=tol
        )
    }


def _maximal_consecutive_runs(values):
    """Return maximal inclusive integer runs from ``values`` (gaps split runs)."""
    values = sorted({int(v) for v in values})
    if not values:
        return []
    runs = []
    run_start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((run_start, prev))
        run_start = value
        prev = value
    runs.append((run_start, prev))
    return runs


def _consecutive_interval(values):
    """Return inclusive ``(lo, hi)`` when ``values`` is a consecutive integer set, else ``None``."""
    if not values:
        return None
    lo, hi = min(values), max(values)
    if hi - lo + 1 != len(values):
        return None
    return lo, hi


def _cluster_pairs_8conn(pair_set):
    """Group ``(i,j)`` pairs connected by 8-neighbor steps in index space."""
    pair_set = set(pair_set)
    if not pair_set:
        return []
    visited = set()
    clusters = []
    for seed in sorted(pair_set):
        if seed in visited:
            continue
        stack = [seed]
        cluster = set()
        while stack:
            i, j = stack.pop()
            if (i, j) in visited:
                continue
            visited.add((i, j))
            cluster.add((i, j))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    nb = (i + di, j + dj)
                    if nb in pair_set and nb not in visited:
                        stack.append(nb)
        clusters.append(cluster)
    return clusters


def _opposite_window_from_cluster(cluster):
    """
    Return ``(i_start, i_end, j_start, j_end)`` when anti-diagonal corners are present.

    Opposite encounters have low-i/high-j and high-i/low-j corners in index space.
    """
    if len(cluster) < 2:
        return None
    i_start = min(i for i, _ in cluster)
    i_end = max(i for i, _ in cluster)
    j_start = min(j for _, j in cluster)
    j_end = max(j for _, j in cluster)
    if (i_start, j_end) not in cluster or (i_end, j_start) not in cluster:
        return None
    return i_start, i_end, j_start, j_end


def _same_direction_bands(pair_set):
    """
    Partition ``pair_set`` into compressible same-direction bands.

    Each band is a set of ``(i,j)`` pairs sharing one z: multi-row marching bands,
    then single-row or single-column consecutive streaks.
    """
    bands = []
    remaining = set(pair_set)

    row_to_js = defaultdict(set)
    for i, j in remaining:
        row_to_js[i].add(j)
    rows = sorted(row_to_js)
    start = 0
    while start < len(rows):
        end = start
        while end + 1 < len(rows):
            row_i = rows[end]
            row_ip1 = rows[end + 1]
            if row_ip1 != row_i + 1:
                break
            linked = any(
                (row_i, j) in remaining and (row_ip1, j + 1) in remaining for _i, j in remaining if _i == row_i
            )
            if not linked:
                break
            end += 1

        band_rows = rows[start : end + 1]
        if len(band_rows) >= 2:
            intervals = [_consecutive_interval(row_to_js[row]) for row in band_rows]
            if all(interval is not None for interval in intervals):
                band = set()
                for row, (j_lo, j_hi) in zip(band_rows, intervals):
                    for j in range(j_lo, j_hi + 1):
                        pair = (row, j)
                        if pair in remaining:
                            band.add(pair)
                if len(band) >= 2:
                    bands.append(band)
                    remaining -= band
                    start = end + 1
                    continue
        start = end + 1

    by_i = defaultdict(set)
    for i, j in remaining:
        by_i[i].add(j)
    for i in sorted(by_i):
        interval = _consecutive_interval(by_i[i])
        if interval is None or interval[1] == interval[0]:
            continue
        j_lo, j_hi = interval
        band = {(i, j) for j in range(j_lo, j_hi + 1) if (i, j) in remaining}
        if len(band) >= 2:
            bands.append(band)
            remaining -= band

    by_j = defaultdict(set)
    for i, j in remaining:
        by_j[j].add(i)
    for j in sorted(by_j):
        interval = _consecutive_interval(by_j[j])
        if interval is None or interval[1] == interval[0]:
            continue
        i_lo, i_hi = interval
        band = {(i, j) for i in range(i_lo, i_hi + 1) if (i, j) in remaining}
        if len(band) >= 2:
            bands.append(band)
            remaining -= band

    return bands


_ORDERING_MODES = frozenset({"both", "a_first", "b_first"})


def _add_ordering_constraints(constraints, times_a, times_b, i, j, z_var, z_index, mode):
    """Append Big-M segment-ordering constraints; ``mode`` selects which disjunct(s) to add."""
    if z_var is None:
        return
    if mode in ("both", "a_first"):
        constraints.append(times_a[i + 1] <= times_b[j] - DELTA + M * z_var[z_index])
    if mode in ("both", "b_first"):
        constraints.append(times_b[j + 1] <= times_a[i] - DELTA + M * (1 - z_var[z_index]))


def _parse_collision_entry(entry):
    """Return ``(leading, i, j, z_index, mode)`` for a collision-pair tuple."""
    if len(entry) >= 2 and entry[-1] in _ORDERING_MODES:
        return entry[:-4], int(entry[-4]), int(entry[-3]), int(entry[-2]), entry[-1]
    return entry[:-3], int(entry[-3]), int(entry[-2]), int(entry[-1]), "both"


def _append_assigned_collision_pairs(collision_pairs, prefix, assigned):
    """Expand z-assignment output into collision-pair tuples (with optional half-disjunct modes)."""
    for entry in assigned:
        if entry[0] == "opp_win":
            _, i_start, i_end, j_start, j_end, z = entry
            collision_pairs.append(prefix + (int(i_end), int(j_start), int(z), "a_first"))
            collision_pairs.append(prefix + (int(i_start), int(j_end), int(z), "b_first"))
        else:
            i, j, z = entry
            collision_pairs.append(prefix + (int(i), int(j), int(z), "both"))


def _assign_z_to_segment_pairs(segment_pairs, opposite_pairs=None, start_z_index=0):
    """Assign ``z_index`` to ``(i,j)`` pairs using index-only region compression."""
    segment_pairs = sorted({(int(i), int(j)) for i, j in segment_pairs})
    if not segment_pairs:
        return [], start_z_index

    pair_set = set(segment_pairs)
    opposite_pairs = {(int(i), int(j)) for (i, j) in (opposite_pairs or ())}
    pair_to_z = {}
    opposite_covered = set()
    window_entries = []
    z_index = int(start_z_index)

    for cluster in _cluster_pairs_8conn(opposite_pairs):
        window = _opposite_window_from_cluster(cluster)
        if window is None:
            continue
        i_start, i_end, j_start, j_end = window
        opposite_covered |= cluster
        window_entries.append(("opp_win", i_start, i_end, j_start, j_end, z_index))
        z_index += 1

    same_pairs = pair_set - opposite_covered
    for band in _same_direction_bands(same_pairs):
        for pair in band:
            pair_to_z[pair] = z_index
        z_index += 1

    for i, j in segment_pairs:
        if (i, j) in opposite_covered or (i, j) in pair_to_z:
            continue
        pair_to_z[(i, j)] = z_index
        z_index += 1

    assigned = list(window_entries)
    assigned.extend((i, j, pair_to_z[(i, j)]) for (i, j) in segment_pairs if (i, j) in pair_to_z)
    return assigned, z_index


_VECTOR_NORM_EXACT = None  # cached result of _vectorized_norm_is_exact()
_CONFLICT_BLOCK_ELEMENTS = 4_000_000  # vertex pairs per vectorized block (bounds memory)


def _vectorized_norm_is_exact():
    """
    True if ``sqrt(dx*dx + dy*dy)`` equals ``np.linalg.norm`` of the 2-vector on this
    platform (it does unless numpy's BLAS fuses the multiply-add). The vectorized
    conflict detection relies on it to reproduce ``_segment_endpoint_conflict`` exactly.
    """
    global _VECTOR_NORM_EXACT
    if _VECTOR_NORM_EXACT is None:
        rng = np.random.default_rng(12345)
        res = 0.2
        diffs = np.concatenate([
            res * rng.integers(-3000, 3000, 4000) - res * rng.integers(-3000, 3000, 4000),
            rng.uniform(-60.0, 60.0, 4000),
        ])
        dx, dy = diffs, np.roll(diffs, 1)
        vectorized = np.sqrt(dx * dx + dy * dy)
        _VECTOR_NORM_EXACT = all(
            float(np.linalg.norm(np.array([x, y]))) == v
            for x, y, v in zip(dx.tolist(), dy.tolist(), vectorized.tolist())
        )
    return _VECTOR_NORM_EXACT


def _iter_conflicting_segment_pairs_scalar(path_a, path_b, threshold):
    """Reference implementation: test every segment pair with ``_segment_endpoint_conflict``."""
    if len(path_a) < 2 or len(path_b) < 2:
        return
    for i in range(len(path_a) - 1):
        for j in range(len(path_b) - 1):
            if _segment_endpoint_conflict(
                path_a[i], path_a[i + 1], path_b[j], path_b[j + 1], threshold
            ):
                yield i, j


def _iter_conflicting_segment_pairs(path_a, path_b, threshold):
    """
    Yield (segment_index_i, segment_index_j) for conflicting segment pairs, i.e. pairs
    where some endpoint of segment i of ``path_a`` is within ``threshold`` of some
    endpoint of segment j of ``path_b``.

    Vectorized over all vertex pairs with the same per-element arithmetic as the scalar
    reference (:func:`_iter_conflicting_segment_pairs_scalar`), so the yielded pairs and
    their order (i ascending, then j) are identical. Falls back to the scalar loop for
    non-2-D points or if the platform's ``np.linalg.norm`` cannot be reproduced exactly.
    """
    if len(path_a) < 2 or len(path_b) < 2:
        return
    a = np.asarray(path_a, dtype=float)
    b = np.asarray(path_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2 or not _vectorized_norm_is_exact():
        yield from _iter_conflicting_segment_pairs_scalar(path_a, path_b, threshold)
        return
    bx, by = b[:, 0], b[:, 1]
    rows_per_block = max(2, _CONFLICT_BLOCK_ELEMENTS // max(1, len(b)))
    n_segments_a = len(a) - 1
    start = 0
    while start < n_segments_a:
        stop = min(n_segments_a, start + rows_per_block - 1)
        block = a[start:stop + 1]  # vertices start..stop (segments start..stop-1)
        dx = block[:, 0:1] - bx[None, :]
        dy = block[:, 1:2] - by[None, :]
        close = np.sqrt(dx * dx + dy * dy) <= threshold
        conflict = close[:-1, :-1] | close[:-1, 1:] | close[1:, :-1] | close[1:, 1:]
        for i, j in np.argwhere(conflict).tolist():
            yield start + i, j
        start = stop


def _collect_segment_collision_pairs(
    collision_pairs, path_a, path_b, threshold, prefix=(), start_z_index=0
):
    """
    Append segment collision tuples with z_index (opposite-window + same-direction coupling).

    Returns:
        num_z: total number of binary z variables required after this batch
    """
    segment_pairs = list(_iter_conflicting_segment_pairs(path_a, path_b, threshold))
    opposite_pairs = _tag_opposite_segment_pairs(path_a, path_b, segment_pairs)
    assigned, z_index = _assign_z_to_segment_pairs(segment_pairs, opposite_pairs, start_z_index)
    _append_assigned_collision_pairs(collision_pairs, prefix, assigned)
    return z_index


def _print_compressed_z_pair_summary(collision_pairs, num_agents, total_z):
    """Print distinct ordering z variables per robot-robot pair (after compression)."""
    z_by_pair = defaultdict(set)
    for entry in collision_pairs:
        leading, _i, _j, z_idx, _mode = _parse_collision_entry(entry)
        if len(leading) == 2:
            z_by_pair[(int(leading[0]), int(leading[1]))].add(int(z_idx))

    parts = []
    for a1 in range(num_agents):
        for a2 in range(a1 + 1, num_agents):
            parts.append(f"({a1},{a2})={len(z_by_pair.get((a1, a2), set()))}")
    summary = ", ".join(parts) if parts else "no agent pairs"
    print(f"MILP z per pair (compressed): {summary}; total z={int(total_z)}")


def _group_collision_pairs_by_agent_pair(collision_pairs):
    grouped = defaultdict(list)
    for entry in collision_pairs:
        leading, i, j, z_idx, mode = _parse_collision_entry(entry)
        if len(leading) == 2:
            grouped[(int(leading[0]), int(leading[1]))].append((i, j, z_idx, mode))
    return dict(grouped)


def _append_segment_pairs_with_z(
    collision_pairs, path_a, path_b, segment_pairs, prefix=(), start_z_index=0
):
    """Append pre-detected ``(seg_i, seg_j)`` tuples with coupled z assignment."""
    opposite_pairs = _tag_opposite_segment_pairs(path_a, path_b, segment_pairs)
    assigned, z_index = _assign_z_to_segment_pairs(segment_pairs, opposite_pairs, start_z_index)
    _append_assigned_collision_pairs(collision_pairs, prefix, assigned)
    return z_index


def detect_collision_pairs_for_agent_pair(path_a, path_b, a_idx, b_idx, threshold=0.5):
    """Return ``(a_idx, b_idx, seg_i, seg_j)`` for one agent pair (detection only)."""
    segment_i = []
    segment_j = []
    for i, j in _iter_conflicting_segment_pairs(path_a, path_b, threshold):
        segment_i.append(int(i))
        segment_j.append(int(j))
    return int(a_idx), int(b_idx), segment_i, segment_j


def merge_collision_reports(paths, reports, threshold=0.5):
    """Merge distributed segment reports into full ``(a1, a2, i, j, z_index)`` tuples.

    ``reports`` is a list of dicts with keys ``a1``, ``a2``, ``segment_i``, ``segment_j``.
    """
    collision_pairs = []
    z_index = 0
    ordered = sorted(reports, key=lambda row: (int(row["a1"]), int(row["a2"])))
    for row in ordered:
        a1 = int(row["a1"])
        a2 = int(row["a2"])
        seg_i = row.get("segment_i") or []
        seg_j = row.get("segment_j") or []
        if len(seg_i) != len(seg_j):
            raise ValueError(f"segment_i/segment_j length mismatch for pair ({a1}, {a2})")
        segment_pairs = [(int(i), int(j)) for i, j in zip(seg_i, seg_j)]
        z_index = _append_segment_pairs_with_z(
            collision_pairs,
            paths[a1],
            paths[a2],
            segment_pairs,
            prefix=(a1, a2),
            start_z_index=z_index,
        )
    return collision_pairs, z_index

def subsample_path_by_stride(path, stride):
    """Return path vertices at indices 0, stride, 2*stride, ... always including the last point.

    ``stride`` must be >= 1. ``stride=1`` returns the input unchanged (shallow copy of list).
    """
    if path is None:
        return []
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if len(path) <= 1 or stride == 1:
        return list(path)
    indices = list(range(0, len(path), stride))
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)
    return [path[i] for i in indices]


class MultiAgentPlanner:

    def __init__(self, occupancy_grid: StochOccupancyGrid2D, v):
        """
        Initialize the multi-agent planner.

        Args:
            occupancy_grid (StochOccupancyGrid2D): The occupancy grid of the environment.
        """
        self.occupancy_grid = occupancy_grid
        self.v = v if v is not None else MAX_VELOCITY

    def assign_path(self, path):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def assign_velocities(self, v):
        """
        Assign the velocities for all agents.

        Args:
            v (list of floats): Max velocities for each agent.
        Returns:
            None
        """
        self.v = v

    def find_collision_pairs(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def plan(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class MultiAgentSequentialPlanner(MultiAgentPlanner):

    def __init__(self, occupancy_grid: StochOccupancyGrid2D, other_agent_paths, other_agent_times, path=None, v=None):
        """
        Initialize the Sequential multi-agent planner.

        Args:
            occupancy_grid (StochOccupancyGrid2D): The occupancy grid of the environment.
            other_agent_paths (list of list of tuples): Paths of other agents in the environment.
            other_agent_times (list of list of floats): Time steps corresponding to other agent paths.
            path (list of tuples, optional): The path for the agent to follow. Defaults to None.
        """
        super().__init__(occupancy_grid, v)

        self.path = path  # Ego path
        self.other_agent_paths = other_agent_paths  # List of other agent paths
        self.other_agent_times = other_agent_times  # List of other agent arrival times

    def assign_path(self, path):
        """
        Assign the ego path for the planner.
        """
        self.path = path

    def find_collision_pairs(self):
        """
        Identify conflicting segment pairs with fixed other-agent schedules.

        Returns:
            collision_pairs: list of (other_idx, i, j, z_index)
                other_idx: index into other_agent_paths / other_agent_times
                i: ego segment index (occupancy [t_i, t_{i+1}])
                j: other-agent segment index (fixed [other_times[j], other_times[j+1]])
                z_index: binary variable for the ordering disjunction
            num_z: total number of z variables needed
        """
        collision_pairs = []
        z_index = 0
        for other_idx, other_path in enumerate(self.other_agent_paths):
            z_index = _collect_segment_collision_pairs(
                collision_pairs,
                self.path,
                other_path,
                ROBOT_DIAMETER,
                prefix=(other_idx,),
                start_z_index=z_index,
            )
        return collision_pairs, z_index

    def plan(self, verbose=False):
        if self.path is None:
            raise ValueError("Path not assigned. Please assign a path before planning.")

        n = len(self.path)
        if n < 1:
            raise ValueError("ego path is empty for sequential planner")
        if n == 1:
            return [0.0]

        t = cp.Variable(n)  # CP variable for time to reach each waypoint
        constraints = []
        constraints += [t[0] == 0]  # Start at time 0

        _add_max_velocity_constraints(constraints, t, self.path, self.v)

        # Collision Avoiding Constraints using Big-M method (segment occupancy intervals)
        collision_pairs, max_z = self.find_collision_pairs()
        z = cp.Variable(max_z, boolean=True) if max_z > 0 else None
        for entry in collision_pairs:
            leading, i, j, z_index, mode = _parse_collision_entry(entry)
            if z is None or len(leading) != 1:
                continue
            other_times = self.other_agent_times[leading[0]]
            _add_ordering_constraints(constraints, t, other_times, i, j, z, z_index, mode)

        objective = cp.Minimize(t[-1])  # Minimize time to reach final point
        prob = cp.Problem(objective, constraints)
        print("Starting to solve multi-agent planning problem...")
        prob.solve(verbose=verbose, solver=cp.ECOS_BB)
        return _scalar_times_from_solver(prob, t, "MultiAgentSequentialPlanner")


class MultiAgentSimultaneousPlanner(MultiAgentPlanner):

    def __init__(self, occupancy_grid: StochOccupancyGrid2D, paths=None, norm=1, v=None):
        """
        Initialize the Simultaneous multi-agent planner.

        Args:
            occupancy_grid (StochOccupancyGrid2D): The occupancy grid of the environment.
            paths (list of list of tuples, optional): Paths for all agents. Defaults to None.
            norm (int, optional): Norm to minimize (1, 2, or inf). Defaults to 1.
            v (list of floats, optional): Max velocities for each agent. Defaults to None.
        """
        super().__init__(occupancy_grid, v)
        self.paths = paths  # List of paths for all agents
        self.norm = norm  # Norm to minimize (1, 2, or inf)
        self.constraint_timing_records = []

    def assign_path(self, paths):
        """
        Assign the paths for all agents.

        Args:
            paths (list of list of tuples): Paths for all agents.
                Each path is a list of (x, y) tuples.
                paths is a list of such paths.

        Returns:
            None
        """
        self.paths = paths

    def find_collision_pairs(self):
        """
        Identify conflicting segment pairs between agents (Section 3.3 interval occupancy).

        Returns:
            collision_pairs: list of (a1, a2, i, j, z_index)
                a1, a2: agent indices
                i, j: segment indices (occupancy [t_i, t_{i+1}], [t_j, t_{j+1}])
                z_index: binary variable for the ordering disjunction
            num_z: total number of z variables needed
        """
        collision_pairs = []
        z_index = 0
        num_agents = len(self.paths)
        pair_detection_stats = {}
        count_before = 0

        for a1 in range(num_agents):
            for a2 in range(a1 + 1, num_agents):
                pair_t0 = time.perf_counter()
                count_before = len(collision_pairs)
                z_index = _collect_segment_collision_pairs(
                    collision_pairs,
                    self.paths[a1],
                    self.paths[a2],
                    ROBOT_DIAMETER,
                    prefix=(a1, a2),
                    start_z_index=z_index,
                )
                pair_detection_stats[(a1, a2)] = {
                    "robot_i": a1,
                    "robot_j": a2,
                    "pair_detect_time_s": float(time.perf_counter() - pair_t0),
                    "collision_tuple_count": int(len(collision_pairs) - count_before),
                }

        self._pair_detection_stats = pair_detection_stats
        return collision_pairs, z_index

    def _build_and_solve_from_collision_pairs(self, collision_pairs, max_z, verbose=False):
        """Build velocity + Big-M constraints from precomputed collision pairs and solve."""
        if len(self.v) == 1:
            self.v = [self.v[0] for _ in range(len(self.paths))]

        _print_compressed_z_pair_summary(collision_pairs, len(self.paths), max_z)
        if VIZ_MILP_Z_ASSIGNMENT:
            from social_path_planning.trajectory_viz import viz_milp_z_assignments

            viz_milp_z_assignments(
                self.paths,
                collision_pairs,
                occ_grid=self.occupancy_grid,
                output_dir=VIZ_MILP_Z_OUTPUT_DIR,
            )

        agent_times = [cp.Variable(len(path)) for path in self.paths]
        constraints = []
        for agent_time in agent_times:
            constraints += [agent_time[0] == 0]

        for agent_idx, path in enumerate(self.paths):
            _add_max_velocity_constraints(constraints, agent_times[agent_idx], path, self.v[agent_idx])

        z = cp.Variable(max_z, boolean=True) if max_z > 0 else None
        pair_constraint_build_time = defaultdict(float)
        pair_constraint_count = defaultdict(int)
        for entry in collision_pairs:
            leading, i, j, z_index, mode = _parse_collision_entry(entry)
            if z is None or len(leading) != 2:
                continue
            a1, a2 = int(leading[0]), int(leading[1])
            build_t0 = time.perf_counter()
            _add_ordering_constraints(
                constraints, agent_times[a1], agent_times[a2], i, j, z, z_index, mode
            )
            pair_key = (a1, a2)
            pair_constraint_build_time[pair_key] += float(time.perf_counter() - build_t0)
            pair_constraint_count[pair_key] += 2 if mode == "both" else 1

        detection_stats = getattr(self, "_pair_detection_stats", {})
        all_pair_keys = sorted(
            set(detection_stats.keys()) | set(pair_constraint_build_time.keys()),
            key=lambda pair: (pair[0], pair[1]),
        )
        self.constraint_timing_records = []
        for pair_key in all_pair_keys:
            detect_row = detection_stats.get(pair_key, {})
            self.constraint_timing_records.append(
                {
                    "robot_i": int(pair_key[0]),
                    "robot_j": int(pair_key[1]),
                    "pair_detect_time_s": float(detect_row.get("pair_detect_time_s", 0.0)),
                    "pair_constraint_build_time_s": float(pair_constraint_build_time.get(pair_key, 0.0)),
                    "collision_tuple_count": int(detect_row.get("collision_tuple_count", 0)),
                    "constraint_count": int(pair_constraint_count.get(pair_key, 0)),
                }
            )

        final_time_vars = cp.hstack([agent_time[-1] for agent_time in agent_times])
        objective = cp.Minimize(cp.norm(final_time_vars, p=self.norm))
        prob = cp.Problem(objective, constraints)
        print("Starting to solve multi-agent planning problem...")
        prob.solve(verbose=verbose, solver=cp.ECOS_BB)
        return _multi_agent_times_from_solver(prob, agent_times, "MultiAgentSimultaneousPlanner")

    def plan_from_collision_pairs(self, collision_pairs, max_z, verbose=False):
        """Solve timing MILP using pre-merged collision pairs (skips detection)."""
        if self.paths is None:
            raise ValueError("Paths not assigned. Please assign paths before planning.")
        return self._build_and_solve_from_collision_pairs(collision_pairs, max_z, verbose=verbose)

    def plan(self, verbose=False):
        if self.paths is None:
            raise ValueError("Paths not assigned. Please assign paths before planning.")

        collision_pairs, max_z = self.find_collision_pairs()
        return self.plan_from_collision_pairs(collision_pairs, max_z, verbose=verbose)

class MultiAgentCombinedPlanner(MultiAgentPlanner):

    def __init__(self, occupancy_grid: StochOccupancyGrid2D, other_agent_paths, other_agent_times, paths=None, v=None, norm=1):
        super().__init__(occupancy_grid, v)

        self.other_agent_paths = other_agent_paths  # List of other agent paths
        self.other_agent_times = other_agent_times  # List of other agent arrival times
        self.paths = paths  # List of paths to find control sequences for
        self.norm = norm

    def assign_path(self, paths):
        """
        Assign the paths for all agents.

        Args:
            paths (list of list of tuples): Paths for all agents.
                Each path is a list of (x, y) tuples.
                paths is a list of such paths.

        Returns:
            None
        """
        self.paths = paths

    def find_collision_pairs(self):
        z_index = 0
        num_agents = len(self.paths)

        sequential_pairs = []
        for other_idx, other_path in enumerate(self.other_agent_paths):
            for a, path in enumerate(self.paths):
                z_index = _collect_segment_collision_pairs(
                    sequential_pairs,
                    path,
                    other_path,
                    ROBOT_DIAMETER,
                    prefix=(other_idx, a),
                    start_z_index=z_index,
                )

        simultaneous_pairs = []
        for a1 in range(num_agents):
            for a2 in range(a1 + 1, num_agents):
                z_index = _collect_segment_collision_pairs(
                    simultaneous_pairs,
                    self.paths[a1],
                    self.paths[a2],
                    ROBOT_DIAMETER,
                    prefix=(a1, a2),
                    start_z_index=z_index,
                )

        return (sequential_pairs, simultaneous_pairs), z_index

    def plan(self, verbose=False):
        if self.paths is None:
            raise ValueError("Paths not assigned. Please assign paths before planning.")

        if len(self.v) == 1:
            self.v = [self.v[0] for _ in range(len(self.paths))]

        # Create CP variables for each agent's time to reach each waypoint
        agent_times = [cp.Variable(len(path)) for path in self.paths]
        constraints = []
        for agent_time in agent_times:
            constraints += [agent_time[0] == 0]  # All agents start at same time (t=0)

        for agent_idx, path in enumerate(self.paths):
            _add_max_velocity_constraints(constraints, agent_times[agent_idx], path, self.v[agent_idx])

        # Collision Avoiding Constraints using Big-M method
        collision_pairs, max_z = self.find_collision_pairs()  # Get all the collision pairs
        sequential_pairs, simultaneous_pairs = collision_pairs
        z = cp.Variable(max_z, boolean=True) if max_z > 0 else None
        for entry in sequential_pairs:
            leading, i, j, z_index, mode = _parse_collision_entry(entry)
            if z is None or len(leading) != 2:
                continue
            other_idx, a = int(leading[0]), int(leading[1])
            other_times = self.other_agent_times[other_idx]
            _add_ordering_constraints(
                constraints, agent_times[a], other_times, i, j, z, z_index, mode
            )
        for entry in simultaneous_pairs:
            leading, i, j, z_index, mode = _parse_collision_entry(entry)
            if z is None or len(leading) != 2:
                continue
            a1, a2 = int(leading[0]), int(leading[1])
            _add_ordering_constraints(
                constraints, agent_times[a1], agent_times[a2], i, j, z, z_index, mode
            )

        final_time_vars = cp.hstack([agent_time[-1] for agent_time in agent_times])
        objective = cp.Minimize(cp.norm(final_time_vars, p=self.norm))  # Minimize norm of final times
        prob = cp.Problem(objective, constraints)
        print("Starting to solve multi-agent planning problem...")
        prob.solve(verbose=verbose, solver=cp.ECOS_BB)
        return _multi_agent_times_from_solver(prob, agent_times, "MultiAgentCombinedPlanner")

def path_to_arc_length(path):
    """
    Convert a path [(x0, y0), (x1, y1), ...] to cumulative distance values.
    """
    if path is None or len(path) == 0:
        return []

    arc_lengths = [0.0]
    for i in range(1, len(path)):
        segment = np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        arc_lengths.append(arc_lengths[-1] + segment)
    return arc_lengths


# Plotting helpers now live in ``trajectory_viz``; keep the old names importable
# from here without importing matplotlib when this module loads.
_MOVED_TO_TRAJECTORY_VIZ = {
    "get_position_at_time": "get_position_at_time",
    "create_space_time_plot": "create_space_time_plot",
    "_draw_map_context": "_draw_map_context",
    "create_map_context_plot": "create_map_context_plot",
    "create_map_context_video": "create_map_context_video",
    "create_video": "create_video",
    "_viz_milp_z_assignments": "viz_milp_z_assignments",
}


def __getattr__(name):
    target = _MOVED_TO_TRAJECTORY_VIZ.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from social_path_planning import trajectory_viz

    return getattr(trajectory_viz, target)


if __name__ == "__main__":
    from social_path_planning.demo_multi_planning import main

    main()
