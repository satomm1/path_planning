"""
Event-based multi-agent path analysis: conflict intervals and interest waypoints.

Phase 1: detect conflict boundaries, refine polylines, visualize interest waypoints.
Phase 2: event-based MILP on interest waypoints with per-encounter mutex constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import cvxpy as cp
import numpy as np

from social_path_planning.mapf_comparison.motion import DEFAULT_MAX_VELOCITY_MPS
from social_path_planning.multi_planning import (
    DELTA,
    GEOM_MATCH_TOL,
    M,
    ROBOT_DIAMETER,
    _iter_conflicting_segment_pairs,
    _maximal_consecutive_runs,
    _multi_agent_times_from_solver,
    _scalar_times_from_solver,
    path_to_arc_length,
)

MAX_VELOCITY = DEFAULT_MAX_VELOCITY_MPS
INTERSECT_CONFLICT_LENGTH_MAX = 2.0


def _conflict_interval_length(interval):
    """Return planned-path arc-length spanned by a conflict interval."""
    return max(0.0, float(interval.s_exit) - float(interval.s_enter))


def _encounter_kind_from_index_corners(segment_pairs):
    """
    Classify an encounter from conflicting segment-index corners.

    Opposite when the earliest conflict index on one robot pairs with the
    latest index on the other, and vice versa (anti-diagonal in index space).
    """
    pair_set = {(int(i), int(j)) for i, j in segment_pairs}
    if len(pair_set) < 2:
        return "same"
    seg_i = {i for i, _ in pair_set}
    seg_j = {j for _, j in pair_set}
    i_start, i_end = min(seg_i), max(seg_i)
    j_start, j_end = min(seg_j), max(seg_j)
    if (i_start, j_end) in pair_set and (i_end, j_start) in pair_set:
        return "opposite"
    return "same"


def _classify_encounter_kind(segment_pairs, interval_a, interval_b):
    """
    Classify a pairwise encounter window.

    Short conflicts on both agents are treated as intersections. Otherwise fall
    back to index-corner same/opposite detection.
    """
    len_a = _conflict_interval_length(interval_a)
    len_b = _conflict_interval_length(interval_b)
    if (
        len_a < INTERSECT_CONFLICT_LENGTH_MAX
        and len_b < INTERSECT_CONFLICT_LENGTH_MAX
    ):
        return "intersect"
    return _encounter_kind_from_index_corners(segment_pairs)


def _encounter_uses_opposite_mutex(kind):
    """Return whether an encounter uses clear-before-enter mutex constraints."""
    return kind in ("opposite", "intersect")


@dataclass(frozen=True)
class ConflictInterval:
    agent: int
    s_enter: float
    s_exit: float
    seg_start: int
    seg_end: int


@dataclass(frozen=True)
class EncounterWindow:
    agent_a: int
    agent_b: int
    interval_a: ConflictInterval
    interval_b: ConflictInterval
    kind: str  # "opposite" | "same" | "intersect"


@dataclass(frozen=True)
class InterestWaypoint:
    agent: int
    s: float
    point: tuple[float, float]
    kind: str


@dataclass
class AgentEventPath:
    agent: int
    original_path: list[tuple[float, float]]
    refined_path: list[tuple[float, float]]
    interest_waypoints: list[InterestWaypoint]
    conflict_intervals: list[ConflictInterval]


@dataclass
class EventPathAnalysis:
    agents: list[AgentEventPath]
    encounters: list[EncounterWindow]


def point_at_arc_length(path, s):
    """Return ``(x, y)`` at arc-length ``s`` along ``path`` (clamped to ``[0, total]``)."""
    if not path:
        raise ValueError("path is empty")
    if len(path) == 1:
        return (float(path[0][0]), float(path[0][1]))

    cum = path_to_arc_length(path)
    total = float(cum[-1])
    s = float(np.clip(s, 0.0, total))
    if s <= 0.0:
        return (float(path[0][0]), float(path[0][1]))
    if s >= total:
        return (float(path[-1][0]), float(path[-1][1]))

    idx = int(np.searchsorted(cum, s, side="right") - 1)
    idx = max(0, min(idx, len(path) - 2))
    seg_len = float(cum[idx + 1] - cum[idx])
    if seg_len <= 1e-12:
        return (float(path[idx][0]), float(path[idx][1]))
    t = (s - float(cum[idx])) / seg_len
    p0 = np.array(path[idx], dtype=float)
    p1 = np.array(path[idx + 1], dtype=float)
    pt = p0 + t * (p1 - p0)
    return (float(pt[0]), float(pt[1]))


def refine_path_at_arc_lengths(path, s_values):
    """Insert points at arc-length positions so consecutive vertices span one segment each."""
    if not path:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]

    cum = path_to_arc_length(path)
    total = float(cum[-1])
    unique_s = sorted({float(np.clip(s, 0.0, total)) for s in s_values})
    refined = [point_at_arc_length(path, s) for s in unique_s]

    deduped = [refined[0]]
    for pt in refined[1:]:
        if np.linalg.norm(np.array(pt) - np.array(deduped[-1])) > GEOM_MATCH_TOL:
            deduped.append(pt)
    return deduped


def _merge_conflict_intervals(intervals, eps=GEOM_MATCH_TOL):
    """Merge overlapping or touching conflict intervals on one agent."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda row: (row.s_enter, row.s_exit))
    merged = [ordered[0]]
    for current in ordered[1:]:
        prev = merged[-1]
        if current.s_enter <= prev.s_exit + eps:
            merged[-1] = ConflictInterval(
                agent=prev.agent,
                s_enter=prev.s_enter,
                s_exit=max(prev.s_exit, current.s_exit),
                seg_start=min(prev.seg_start, current.seg_start),
                seg_end=max(prev.seg_end, current.seg_end),
            )
        else:
            merged.append(current)
    return merged


def _segment_run_to_interval(agent, path, seg_start, seg_end):
    cum = path_to_arc_length(path)
    return ConflictInterval(
        agent=int(agent),
        s_enter=float(cum[seg_start]),
        s_exit=float(cum[seg_end + 1]),
        seg_start=int(seg_start),
        seg_end=int(seg_end),
    )


def _intervals_from_segment_indices(agent, path, segment_indices):
    if not segment_indices:
        return []
    runs = _maximal_consecutive_runs(segment_indices)
    intervals = [_segment_run_to_interval(agent, path, lo, hi) for lo, hi in runs]
    return _merge_conflict_intervals(intervals)


def _build_interest_waypoints(agent, path, intervals, encounter_intervals=None):
    """Build interest waypoints from merged conflicts and per-pair encounter boundaries."""
    cum = path_to_arc_length(path)
    total = float(cum[-1]) if cum else 0.0
    enter_s = set()
    exit_s = set()
    for interval in intervals:
        enter_s.add(float(interval.s_enter))
        exit_s.add(float(interval.s_exit))
    for interval in encounter_intervals or []:
        enter_s.add(float(interval.s_enter))
        exit_s.add(float(interval.s_exit))

    s_values = {0.0, total, *enter_s, *exit_s}
    ordered_s = sorted(s_values)

    waypoints = []
    for s in ordered_s:
        if abs(s) <= GEOM_MATCH_TOL:
            kind = "start"
        elif abs(s - total) <= GEOM_MATCH_TOL:
            kind = "goal"
        else:
            is_enter = any(abs(float(s) - val) <= GEOM_MATCH_TOL for val in enter_s)
            is_exit = any(abs(float(s) - val) <= GEOM_MATCH_TOL for val in exit_s)
            if is_exit:
                kind = "conflict_exit"
            elif is_enter:
                kind = "conflict_enter"
            else:
                kind = "conflict_enter"
        waypoints.append(
            InterestWaypoint(
                agent=int(agent),
                s=float(s),
                point=point_at_arc_length(path, s),
                kind=kind,
            )
        )
    return waypoints


def _encounter_interval(agent, path, seg_indices):
    if not seg_indices:
        raise ValueError("encounter requires at least one conflicting segment")
    seg_start = min(seg_indices)
    seg_end = max(seg_indices)
    return _segment_run_to_interval(agent, path, seg_start, seg_end)


def _cluster_segment_pairs_into_encounters(segment_pairs):
    """
    Split conflicting segment pairs into disjoint encounter components.

    Pairs belong to the same encounter when they share a segment index on either
    robot or use consecutive segment indices on one robot (one contiguous
    conflict zone). Disjoint spatial conflict regions must not be merged into a
    single encounter window.
    """
    pairs = [(int(i), int(j)) for i, j in segment_pairs]
    if not pairs:
        return []
    if len(pairs) == 1:
        return [pairs]

    parent = list(range(len(pairs)))

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    # Two pairs are linked when their i (or j) indices are equal or differ by one. Rather
    # than testing every pair of pairs, link all pairs sharing an index value and then
    # link adjacent index values; this yields the same components in near-linear time.
    by_i: dict[int, list[int]] = {}
    by_j: dict[int, list[int]] = {}
    for idx, (i, j) in enumerate(pairs):
        by_i.setdefault(i, []).append(idx)
        by_j.setdefault(j, []).append(idx)
    for groups in (by_i, by_j):
        for value, members in groups.items():
            first = members[0]
            for other in members[1:]:
                union(first, other)
            neighbor = groups.get(value + 1)
            if neighbor is not None:
                union(first, neighbor[0])

    grouped: dict[int, list[tuple[int, int]]] = {}
    for idx, pair in enumerate(pairs):
        root = find(idx)
        grouped.setdefault(root, []).append(pair)
    return list(grouped.values())


def _segment_indices_from_pair_conflicts(pair_conflicts, num_agents):
    """Collect per-agent conflicting segment indices from a pair_conflicts map."""
    segment_indices_by_agent = [set() for _ in range(num_agents)]
    for (a1, a2), row in pair_conflicts.items():
        for i, _j in row["pairs"]:
            segment_indices_by_agent[int(a1)].add(int(i))
        for _i, j in row["pairs"]:
            segment_indices_by_agent[int(a2)].add(int(j))
    return segment_indices_by_agent


def _build_event_analysis(paths, pair_conflicts):
    """
    Build EventPathAnalysis from paths and precomputed conflicting segment pairs.

    ``pair_conflicts`` maps ``(a1, a2)`` to ``{"pairs": [(i, j), ...]}``.
    """
    paths = [list(path) for path in paths]
    num_agents = len(paths)
    segment_indices_by_agent = _segment_indices_from_pair_conflicts(pair_conflicts, num_agents)

    agents = []
    encounter_intervals_by_agent = [[] for _ in range(num_agents)]
    encounters = []
    for (a1, a2), row in sorted(pair_conflicts.items()):
        for component_pairs in _cluster_segment_pairs_into_encounters(row["pairs"]):
            seg_i = {int(i) for i, _j in component_pairs}
            seg_j = {int(j) for _i, j in component_pairs}
            interval_a = _encounter_interval(a1, paths[a1], seg_i)
            interval_b = _encounter_interval(a2, paths[a2], seg_j)
            kind = _classify_encounter_kind(component_pairs, interval_a, interval_b)
            encounters.append(
                EncounterWindow(
                    agent_a=a1,
                    agent_b=a2,
                    interval_a=interval_a,
                    interval_b=interval_b,
                    kind=kind,
                )
            )
            encounter_intervals_by_agent[a1].append(interval_a)
            encounter_intervals_by_agent[a2].append(interval_b)

    for agent, path in enumerate(paths):
        intervals = _intervals_from_segment_indices(agent, path, segment_indices_by_agent[agent])
        interest = _build_interest_waypoints(
            agent,
            path,
            intervals,
            encounter_intervals_by_agent[agent],
        )
        s_values = [wp.s for wp in interest]
        refined = refine_path_at_arc_lengths(path, s_values)
        agents.append(
            AgentEventPath(
                agent=agent,
                original_path=path,
                refined_path=refined,
                interest_waypoints=interest,
                conflict_intervals=intervals,
            )
        )

    return EventPathAnalysis(agents=agents, encounters=encounters)


def analyze_event_paths(paths, threshold=ROBOT_DIAMETER):
    """
    Analyze multi-agent paths and build interest waypoints at conflict boundaries.

    Returns:
        EventPathAnalysis with per-agent refined paths and pairwise encounter windows.
    """
    paths = [list(path) for path in paths]
    num_agents = len(paths)
    pair_conflicts = {}

    for a1 in range(num_agents):
        for a2 in range(a1 + 1, num_agents):
            path_a, path_b = paths[a1], paths[a2]
            if len(path_a) < 2 or len(path_b) < 2:
                continue
            segment_pairs = list(_iter_conflicting_segment_pairs(path_a, path_b, threshold))
            if not segment_pairs:
                continue
            pair_conflicts[(a1, a2)] = {
                "pairs": segment_pairs,
            }

    return _build_event_analysis(paths, pair_conflicts)


def analyze_event_paths_from_segment_reports(paths, reports, threshold=ROBOT_DIAMETER):
    """
    Build EventPathAnalysis from distributed segment-collision reports.

    ``reports`` is a list of dicts with keys ``a1``, ``a2``, ``segment_i``,
    ``segment_j`` (0-based agent indices into ``paths``, parallel segment lists).
    """
    del threshold  # reserved for API compatibility; pairs are pre-detected
    paths = [list(path) for path in paths]
    num_agents = len(paths)
    pair_conflicts = {}

    for row in sorted(reports, key=lambda item: (int(item["a1"]), int(item["a2"]))):
        a1 = int(row["a1"])
        a2 = int(row["a2"])
        if a1 < 0 or a2 < 0 or a1 >= num_agents or a2 >= num_agents:
            raise ValueError(f"report agent indices ({a1}, {a2}) out of range for {num_agents} paths")
        if a1 == a2:
            raise ValueError(f"report agent indices must differ, got ({a1}, {a2})")

        seg_i = [int(x) for x in (row.get("segment_i") or [])]
        seg_j = [int(x) for x in (row.get("segment_j") or [])]
        if len(seg_i) != len(seg_j):
            raise ValueError(f"segment_i/segment_j length mismatch for pair ({a1}, {a2})")

        path_a = paths[a1]
        path_b = paths[a2]
        segment_pairs = []
        for i, j in zip(seg_i, seg_j):
            if i < 0 or j < 0:
                raise ValueError(f"negative segment index in report for pair ({a1}, {a2})")
            if i >= len(path_a) - 1:
                raise ValueError(
                    f"segment_i index {i} out of range for agent {a1} path len {len(path_a)}"
                )
            if j >= len(path_b) - 1:
                raise ValueError(
                    f"segment_j index {j} out of range for agent {a2} path len {len(path_b)}"
                )
            segment_pairs.append((int(i), int(j)))

        if not segment_pairs:
            continue

        key = (min(a1, a2), max(a1, a2))
        existing = pair_conflicts.setdefault(key, {"pairs": []})
        pair_set = {(int(i), int(j)) for i, j in existing["pairs"]}
        for i, j in segment_pairs:
            if key[0] == a1:
                pair_set.add((int(i), int(j)))
            else:
                pair_set.add((int(j), int(i)))
        existing["pairs"] = sorted(pair_set)

    return _build_event_analysis(paths, pair_conflicts)


def _interest_segment_lengths(agent_path):
    """Return planned-path arc-length between consecutive interest waypoints."""
    waypoints = agent_path.interest_waypoints
    if len(waypoints) < 2:
        return []
    return [
        max(0.0, float(waypoints[k + 1].s) - float(waypoints[k].s))
        for k in range(len(waypoints) - 1)
    ]


def _index_for_arc_s_on_waypoints(interest_waypoints, s, *, endpoint="enter", eps=GEOM_MATCH_TOL):
    """
    Map original-path arc-length ``s`` to an interest-waypoint index.

    Waypoint ``s`` values are defined along the original planned path. Do not
    bracket using Euclidean cumulative length on ``refined_path``; that polyline
    can shortcut corners and mis-assign exit indices past the true boundary.
    """
    s = float(s)
    for idx, waypoint in enumerate(interest_waypoints):
        if abs(float(waypoint.s) - s) <= eps:
            return int(idx)

    s_vals = [float(waypoint.s) for waypoint in interest_waypoints]
    if endpoint == "enter":
        idx = int(np.searchsorted(s_vals, s, side="right") - 1)
    else:
        idx = int(np.searchsorted(s_vals, s, side="left"))
    return max(0, min(idx, len(interest_waypoints) - 1))


def _refined_indices_for_interval(agent_path, interval):
    """Map a conflict interval to enter/exit indices on the agent's interest waypoints."""
    waypoints = agent_path.interest_waypoints
    enter_idx = _index_for_arc_s_on_waypoints(waypoints, interval.s_enter, endpoint="enter")
    exit_idx = _index_for_arc_s_on_waypoints(waypoints, interval.s_exit, endpoint="exit")
    if exit_idx < enter_idx:
        exit_idx = enter_idx
    return enter_idx, exit_idx


def _compute_big_m_horizon(analysis, velocities):
    """Upper bound on schedule horizon for tightening Big-M."""
    horizon = 0.0
    for agent_path, velocity in zip(analysis.agents, velocities):
        velocity = max(float(velocity), 1e-6)
        for length in _interest_segment_lengths(agent_path):
            horizon += float(length) / velocity
    return max(float(M), horizon + float(DELTA) + 1.0)


def _normalized_encounter_position(s, interval):
    """Return progress in [0, 1] through an encounter interval."""
    span = float(interval.s_exit) - float(interval.s_enter)
    if span <= GEOM_MATCH_TOL:
        return 0.0
    return (float(s) - float(interval.s_enter)) / span


def _waypoint_indices_in_interval(agent_path, interval):
    """Interest-waypoint indices whose arc-length lies in the encounter interval."""
    indices = []
    for idx, waypoint in enumerate(agent_path.interest_waypoints):
        s = float(waypoint.s)
        if float(interval.s_enter) - GEOM_MATCH_TOL <= s <= float(interval.s_exit) + GEOM_MATCH_TOL:
            indices.append(int(idx))
    return indices


def _pair_same_direction_checkpoints(agent_a, agent_b, interval_a, interval_b):
    """
    Pair interest waypoints in the encounter band so both agents keep the same leader.

    Interior waypoints from other encounters can fall inside this band; each is paired
    to the closest checkpoint on the other agent by normalized progress through the
    encounter interval.
    """
    indices_a = _waypoint_indices_in_interval(agent_a, interval_a)
    indices_b = _waypoint_indices_in_interval(agent_b, interval_b)
    if not indices_a or not indices_b:
        return []

    waypoints_a = agent_a.interest_waypoints
    waypoints_b = agent_b.interest_waypoints
    pairs = set()

    for ia in indices_a:
        ua = _normalized_encounter_position(waypoints_a[ia].s, interval_a)
        ib = min(
            indices_b,
            key=lambda j: abs(
                _normalized_encounter_position(waypoints_b[j].s, interval_b) - ua
            ),
        )
        pairs.add((int(ia), int(ib)))

    for ib in indices_b:
        ub = _normalized_encounter_position(waypoints_b[ib].s, interval_b)
        ia = min(
            indices_a,
            key=lambda i: abs(
                _normalized_encounter_position(waypoints_a[i].s, interval_a) - ub
            ),
        )
        pairs.add((int(ia), int(ib)))

    return sorted(pairs)


def _conflict_exit_is_goal(agent_path, exit_idx):
    """True when the encounter exit waypoint is the robot's goal."""
    waypoints = agent_path.interest_waypoints
    if exit_idx < 0 or exit_idx >= len(waypoints):
        return False
    return waypoints[exit_idx].kind == "goal"


def _conflict_enter_is_start(agent_path, enter_idx):
    """True when the encounter enter waypoint is the robot's start."""
    waypoints = agent_path.interest_waypoints
    if enter_idx < 0 or enter_idx >= len(waypoints):
        return False
    return int(enter_idx) == 0 and waypoints[enter_idx].kind == "start"


def _encounter_mutex_uses_z(agent_a, agent_b, i_enter, i_exit, j_enter, j_exit):
    """
    Encounters need a binary only when neither fixed trailing-goal nor fixed
    leading-start ordering applies alone.

    Exactly one goal-at-exit robot must trail; exactly one start-at-enter robot
    must lead. Otherwise a binary chooses the encounter ordering.
    """
    a_goal = _conflict_exit_is_goal(agent_a, i_exit)
    b_goal = _conflict_exit_is_goal(agent_b, j_exit)
    a_start = _conflict_enter_is_start(agent_a, i_enter)
    b_start = _conflict_enter_is_start(agent_b, j_enter)
    goal_fixed = bool(a_goal ^ b_goal)
    start_fixed = bool(a_start ^ b_start)
    return not (goal_fixed or start_fixed)


def _encounter_fixed_leader_is_a(agent_a, agent_b, i_enter, i_exit, j_enter, j_exit):
    """
    Return True when agent A must lead through a fixed-order encounter.

    Start-at-enter takes precedence over goal-at-exit when both prescribe order.
    """
    a_start = _conflict_enter_is_start(agent_a, i_enter)
    b_start = _conflict_enter_is_start(agent_b, j_enter)
    if a_start ^ b_start:
        return bool(a_start)
    a_goal = _conflict_exit_is_goal(agent_a, i_exit)
    b_goal = _conflict_exit_is_goal(agent_b, j_exit)
    if a_goal ^ b_goal:
        return bool(b_goal)
    raise ValueError("fixed encounter ordering requires goal-exit or start-enter XOR")


def _time_at_index(times, idx):
    """Return a cvxpy expression or scalar time at ``idx`` for a schedule."""
    idx = int(idx)
    if isinstance(times, cp.Variable):
        return times[idx]
    return float(times[idx])


def _validate_fixed_event_schedule(agent_path, times):
    """Ensure a fixed schedule matches an agent's interest-waypoint count."""
    expected = len(agent_path.interest_waypoints)
    if len(times) != expected:
        raise ValueError(
            f"fixed event schedule length {len(times)} does not match "
            f"{expected} interest waypoints for agent {agent_path.agent}"
        )


def _broadcast_velocities(velocities, count):
    """Broadcast a scalar or one-element list of max velocities to ``count`` agents."""
    if not isinstance(velocities, (list, tuple)):
        return [float(velocities) for _ in range(count)]
    if len(velocities) == 1:
        return [float(velocities[0]) for _ in range(count)]
    return [float(v) for v in velocities]


def _append_kinematic_constraints(constraints, agent_path, agent_time, velocity):
    """Add start-at-zero and max-velocity constraints for one optimized agent."""
    if not isinstance(agent_time, cp.Variable):
        return
    velocity = max(float(velocity), 1e-6)
    constraints.append(agent_time[0] == 0)
    for k, seg_len in enumerate(_interest_segment_lengths(agent_path)):
        constraints.append(agent_time[k + 1] - agent_time[k] >= seg_len / velocity)


def _encounter_row(analysis, encounter):
    """
    Resolve one encounter's interest-waypoint indices and whether it needs a binary.

    Computed once per encounter and reused for counting binaries, building the
    constraints and printing the formulation.
    """
    agent_a = analysis.agents[encounter.agent_a]
    agent_b = analysis.agents[encounter.agent_b]
    i_enter, i_exit = _refined_indices_for_interval(agent_a, encounter.interval_a)
    j_enter, j_exit = _refined_indices_for_interval(agent_b, encounter.interval_b)
    return {
        "z_index": None,
        "uses_z": _encounter_mutex_uses_z(agent_a, agent_b, i_enter, i_exit, j_enter, j_exit),
        "encounter": encounter,
        "i_enter": i_enter,
        "i_exit": i_exit,
        "j_enter": j_enter,
        "j_exit": j_exit,
        "paired_indices": [],
        "lines": [],
    }


def _append_event_encounter_mutex(
    constraints,
    analysis,
    agent_schedules,
    encounter_rows,
    z,
    big_m,
    *,
    describe=False,
):
    """
    Append encounter mutex constraints for precomputed ``encounter_rows``.

    Binaries are assigned in row order to rows with ``uses_z``. ``agent_schedules[i]``
    is either a cvxpy Variable (optimized) or a fixed list of floats aligned with
    ``analysis.agents[i].interest_waypoints``. With ``describe=True`` each row's
    ``"lines"`` receives a human-readable listing of the rows that were added.
    Returns the number of mutex constraints added.
    """
    mutex_constraint_count = 0
    z_slot = 0
    for row in encounter_rows:
        if row["uses_z"]:
            row["z_index"] = z_slot
            z_slot += 1
        added, paired_indices = _add_encounter_mutex_constraints(
            constraints,
            analysis,
            agent_schedules,
            row,
            z,
            big_m,
            lines=row["lines"] if describe else None,
        )
        row["paired_indices"] = paired_indices
        mutex_constraint_count += added
    return mutex_constraint_count


def _sequential_event_encounters(analysis, ego_agent_idx):
    """Return encounters between one ego agent and all other analyzed agents."""
    return [
        encounter
        for encounter in analysis.encounters
        if ego_agent_idx in (encounter.agent_a, encounter.agent_b)
    ]


def _cross_event_encounters(analysis, left_agent_indices, right_agent_indices):
    """Return encounters with one endpoint in each index set."""
    left = set(left_agent_indices)
    right = set(right_agent_indices)
    return [
        encounter
        for encounter in analysis.encounters
        if (encounter.agent_a in left and encounter.agent_b in right)
        or (encounter.agent_a in right and encounter.agent_b in left)
    ]


def _within_event_encounters(analysis, agent_indices):
    """Return encounters where both agents lie in ``agent_indices``."""
    allowed = set(agent_indices)
    return [
        encounter
        for encounter in analysis.encounters
        if encounter.agent_a in allowed and encounter.agent_b in allowed
    ]


def _solve_event_problem(prob, agent_times, *, context, verbose=False):
    """Solve an event MILP and read per-agent optimized schedules.

    Returns ``(event_times, solver_runtime_s)`` where ``solver_runtime_s`` covers
    only cvxpy solve attempts and reading optimized schedules (not model build).
    """
    print("Starting to solve event multi-agent planning problem...")
    solver_chain = [
        name
        for name in (cp.MOSEK, cp.GLPK_MI, cp.ECOS_BB, cp.SCIPY)
        if name in cp.installed_solvers()
    ]
    if not solver_chain:
        raise RuntimeError("No cvxpy solver available for event MILP")

    t0 = time.perf_counter()
    last_error = None
    for solver in solver_chain:
        try:
            prob.solve(verbose=verbose, solver=solver)
            if prob.status is not None:
                break
        except Exception as exc:
            last_error = exc
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Event MILP solve failed without a solver status")

    if len(agent_times) == 1 and isinstance(agent_times[0], cp.Variable):
        event_times = _scalar_times_from_solver(prob, agent_times[0], context)
    else:
        event_times = _multi_agent_times_from_solver(prob, agent_times, context)
    return event_times, float(time.perf_counter() - t0)


def _add_opposite_encounter_mutex_constraints(
    constraints,
    times_a,
    times_b,
    i_enter,
    i_exit,
    j_enter,
    j_exit,
    z_var,
    z_index,
    big_m,
    *,
    agent_a,
    agent_b,
    uses_z,
    lines=None,
):
    """
    Opposite mutex: Big-M with z, or fixed ordering when one robot leads.

    Fixed order applies when exactly one robot starts at conflict enter (it leads)
    or exactly one robot ends at conflict exit at goal (it trails).
    """
    a, b = agent_a.agent, agent_b.agent
    delta = f"{float(DELTA):.4f}"
    if not uses_z:
        if _encounter_fixed_leader_is_a(
            agent_a, agent_b, i_enter, i_exit, j_enter, j_exit
        ):
            constraints.append(times_a[i_exit] <= _time_at_index(times_b, j_enter) - DELTA)
            constraints.append(_time_at_index(times_b, j_exit) >= times_a[i_exit] + DELTA)
            if lines is not None:
                lines.append(
                    f"    (1) t_{a}[{i_exit}] <= t_{b}[{j_enter}] - {delta}"
                    f"   # agent {a} leads (start-at-enter or other trails)"
                )
                lines.append(
                    f"    (2) t_{b}[{j_exit}] >= t_{a}[{i_exit}] + {delta}"
                    f"   # agent {b} follows agent {a}"
                )
        else:
            constraints.append(_time_at_index(times_b, j_exit) <= times_a[i_enter] - DELTA)
            constraints.append(times_a[i_exit] >= _time_at_index(times_b, j_exit) + DELTA)
            if lines is not None:
                lines.append(
                    f"    (1) t_{b}[{j_exit}] <= t_{a}[{i_enter}] - {delta}"
                    f"   # agent {b} leads (start-at-enter or other trails)"
                )
                lines.append(
                    f"    (2) t_{a}[{i_exit}] >= t_{b}[{j_exit}] + {delta}"
                    f"   # agent {a} follows agent {b}"
                )
        return 2

    constraints.append(
        times_a[i_exit] <= _time_at_index(times_b, j_enter) - DELTA + big_m * z_var[z_index]
    )
    constraints.append(
        _time_at_index(times_b, j_exit) <= times_a[i_enter] - DELTA + big_m * (1 - z_var[z_index])
    )
    if lines is not None:
        m = f"{float(big_m):.4f}"
        lines.append(
            f"    (1) t_{a}[{i_exit}] <= t_{b}[{j_enter}] - {delta}"
            f" + {m} * z_{z_index}"
            f"   # agent {a} clears before agent {b} enters"
        )
        lines.append(
            f"    (2) t_{b}[{j_exit}] <= t_{a}[{i_enter}] - {delta}"
            f" + {m} * (1 - z_{z_index})"
            f"   # agent {b} clears before agent {a} enters"
        )
    return 2


def _filter_fixed_same_direction_pairs(paired_indices, agent_a, agent_b, *, leader_is_a):
    """
    Drop paired checkpoints that contradict fixed leader ordering with t[0]=0.

    Requiring the leader to be ahead at its own start, or the trailer to lead at
    the other's start, is impossible when both schedules begin at zero.
    """
    filtered = []
    for ia, ib in paired_indices:
        if leader_is_a:
            if ia == 0 and agent_a.interest_waypoints[ia].kind == "start":
                continue
            waypoint = agent_b.interest_waypoints[ib]
            if waypoint.kind == "start" and ib == 0:
                continue
        else:
            if ib == 0 and agent_b.interest_waypoints[ib].kind == "start":
                continue
            waypoint = agent_a.interest_waypoints[ia]
            if waypoint.kind == "start" and ia == 0:
                continue
        filtered.append((int(ia), int(ib)))
    return filtered


def _same_direction_pairs_for_mutex(
    agent_a,
    agent_b,
    interval_a,
    interval_b,
    *,
    i_enter,
    j_enter,
    i_exit,
    j_exit,
    uses_z,
):
    """Return paired checkpoints used for same-direction mutex constraints."""
    pairs = _pair_same_direction_checkpoints(agent_a, agent_b, interval_a, interval_b)
    if uses_z:
        return pairs
    leader_is_a = _encounter_fixed_leader_is_a(
        agent_a, agent_b, i_enter, i_exit, j_enter, j_exit
    )
    return _filter_fixed_same_direction_pairs(
        pairs, agent_a, agent_b, leader_is_a=leader_is_a
    )


def _add_same_encounter_mutex_constraints(
    constraints,
    times_a,
    times_b,
    paired_indices,
    z_var,
    z_index,
    big_m,
    *,
    agent_a,
    agent_b,
    i_enter,
    j_enter,
    i_exit,
    j_exit,
    uses_z,
    lines=None,
):
    """
    Mutex for same-direction co-marching.

    With z: two Big-M rows per paired checkpoint (leader chosen by z).
    Without z: fixed leader from start-at-enter or trailing goal-at-exit.
    """
    a, b = agent_a.agent, agent_b.agent
    delta = f"{float(DELTA):.4f}"
    if not uses_z:
        leader_is_a = _encounter_fixed_leader_is_a(
            agent_a, agent_b, i_enter, i_exit, j_enter, j_exit
        )
        for pair_no, (ia, ib) in enumerate(paired_indices, start=1):
            if leader_is_a:
                constraints.append(times_a[ia] <= _time_at_index(times_b, ib) - DELTA)
            else:
                constraints.append(_time_at_index(times_b, ib) <= times_a[ia] - DELTA)
            if lines is not None:
                label_a = _waypoint_label(agent_a, ia)
                label_b = _waypoint_label(agent_b, ib)
                if leader_is_a:
                    lines.append(
                        f"    ({pair_no}) t_{a}[{ia}] ({label_a})"
                        f" <= t_{b}[{ib}] ({label_b})"
                        f" - {delta}"
                        f"   # agent {a} leads"
                    )
                else:
                    lines.append(
                        f"    ({pair_no}) t_{b}[{ib}] ({label_b})"
                        f" <= t_{a}[{ia}] ({label_a})"
                        f" - {delta}"
                        f"   # agent {b} leads"
                    )
        return len(paired_indices)

    z = z_var[z_index]
    for pair_no, (ia, ib) in enumerate(paired_indices, start=1):
        constraints.append(times_a[ia] <= _time_at_index(times_b, ib) - DELTA + big_m * z)
        constraints.append(_time_at_index(times_b, ib) <= times_a[ia] - DELTA + big_m * (1 - z))
        if lines is not None:
            label_a = _waypoint_label(agent_a, ia)
            label_b = _waypoint_label(agent_b, ib)
            m = f"{float(big_m):.4f}"
            lines.append(
                f"    ({pair_no}a) t_{a}[{ia}] ({label_a})"
                f" <= t_{b}[{ib}] ({label_b})"
                f" - {delta} + {m} * z_{z_index}"
                f"   # z=0: agent {a} ahead"
            )
            lines.append(
                f"    ({pair_no}b) t_{b}[{ib}] ({label_b})"
                f" <= t_{a}[{ia}] ({label_a})"
                f" - {delta} + {m} * (1 - z_{z_index})"
                f"   # z=1: agent {b} ahead"
            )
    return 2 * len(paired_indices)


def _add_encounter_mutex_constraints(
    constraints,
    analysis,
    agent_schedules,
    row,
    z_var,
    big_m,
    *,
    lines=None,
):
    """Append mutex constraints for one encounter row; returns ``(added, paired_indices)``."""
    encounter = row["encounter"]
    agent_a = analysis.agents[encounter.agent_a]
    agent_b = analysis.agents[encounter.agent_b]
    times_a = agent_schedules[encounter.agent_a]
    times_b = agent_schedules[encounter.agent_b]
    i_enter, i_exit = row["i_enter"], row["i_exit"]
    j_enter, j_exit = row["j_enter"], row["j_exit"]
    uses_z = row["uses_z"]
    z_index = row["z_index"]
    if _encounter_uses_opposite_mutex(encounter.kind):
        added = _add_opposite_encounter_mutex_constraints(
            constraints,
            times_a,
            times_b,
            i_enter,
            i_exit,
            j_enter,
            j_exit,
            z_var,
            z_index,
            big_m,
            agent_a=agent_a,
            agent_b=agent_b,
            uses_z=uses_z,
            lines=lines,
        )
        return added, []
    paired_indices = _same_direction_pairs_for_mutex(
        agent_a,
        agent_b,
        encounter.interval_a,
        encounter.interval_b,
        i_enter=i_enter,
        j_enter=j_enter,
        i_exit=i_exit,
        j_exit=j_exit,
        uses_z=uses_z,
    )
    added = _add_same_encounter_mutex_constraints(
        constraints,
        times_a,
        times_b,
        paired_indices,
        z_var,
        z_index,
        big_m,
        agent_a=agent_a,
        agent_b=agent_b,
        i_enter=i_enter,
        j_enter=j_enter,
        i_exit=i_exit,
        j_exit=j_exit,
        uses_z=uses_z,
        lines=lines,
    )
    return added, paired_indices


def _encounter_uses_z(analysis, encounter):
    """Return whether an encounter allocates a binary decision variable."""
    return _encounter_row(analysis, encounter)["uses_z"]


def _encounter_kind_counts(encounters):
    """Return counts of opposite, same, and intersect encounters."""
    opposite = sum(1 for enc in encounters if enc.kind == "opposite")
    same = sum(1 for enc in encounters if enc.kind == "same")
    intersect = sum(1 for enc in encounters if enc.kind == "intersect")
    return opposite, same, intersect


def _print_event_milp_summary(analysis, num_z, mutex_constraint_count):
    interest_total = sum(len(agent.interest_waypoints) for agent in analysis.agents)
    opposite_count, same_count, intersect_count = _encounter_kind_counts(analysis.encounters)
    print(
        "Event MILP: "
        f"{len(analysis.agents)} agents, {interest_total} interest waypoints, "
        f"{len(analysis.encounters)} encounters "
        f"({opposite_count} opposite, {same_count} same, {intersect_count} intersect), "
        f"{mutex_constraint_count} mutex constraints, {int(num_z)} binaries"
    )


def _waypoint_label(agent_path, idx):
    """Human-readable label for a refined-path / interest-waypoint index."""
    if 0 <= idx < len(agent_path.interest_waypoints):
        waypoint = agent_path.interest_waypoints[idx]
        return f"{waypoint.kind} (s={float(waypoint.s):.4f})"
    return f"idx={idx}"


def _print_event_milp_constraints(
    analysis,
    velocities,
    *,
    norm,
    big_m,
    encounter_rows,
):
    """Print a human-readable listing of the event MILP objective and constraints."""
    norm_label = "inf" if norm == np.inf else str(int(norm))
    final_terms = [f"t_{agent.agent}[{len(agent.refined_path) - 1}]" for agent in analysis.agents]
    print("\n=== Event MILP formulation ===")
    print(f"Objective: minimize ||[{', '.join(final_terms)}]||_{norm_label}")
    print(f"Big-M: {float(big_m):.4f}  (DELTA={float(DELTA):.4f})")

    for agent_path, velocity in zip(analysis.agents, velocities):
        agent = agent_path.agent
        velocity = max(float(velocity), 1e-6)
        num_wp = len(agent_path.interest_waypoints)
        print(f"\nAgent {agent}: {num_wp} waypoints, v_max={velocity:.4f} m/s")
        print(f"  t_{agent}[0] = 0")
        for k, seg_len in enumerate(_interest_segment_lengths(agent_path)):
            min_dt = float(seg_len) / velocity
            from_lbl = _waypoint_label(agent_path, k)
            to_lbl = _waypoint_label(agent_path, k + 1)
            print(
                f"  t_{agent}[{k + 1}] - t_{agent}[{k}] >= {min_dt:.4f}"
                f"   # path_len={float(seg_len):.4f} m, {from_lbl} -> {to_lbl}"
            )

    z_encounter_count = sum(1 for row in encounter_rows if row.get("uses_z"))
    if not encounter_rows:
        print("\nNo encounter mutex constraints (zero binaries).")
    else:
        print(
            f"\nEncounter mutex constraints "
            f"({z_encounter_count} binaries, {len(encounter_rows)} encounters):"
        )
        for row in encounter_rows:
            enc = row["encounter"]
            z_index = row.get("z_index")
            uses_z = row.get("uses_z", True)
            a, b = enc.agent_a, enc.agent_b
            agent_a = analysis.agents[a]
            agent_b = analysis.agents[b]
            i_enter, i_exit = row["i_enter"], row["i_exit"]
            j_enter, j_exit = row["j_enter"], row["j_exit"]
            if uses_z:
                print(f"\n  z_{z_index}: agents ({a},{b}), kind={enc.kind}")
            else:
                print(f"\n  fixed ordering: agents ({a},{b}), kind={enc.kind} (no binary)")
            print(
                f"    agent {a} interval: s=[{enc.interval_a.s_enter:.4f}, {enc.interval_a.s_exit:.4f}]"
                f" -> indices enter={i_enter} ({_waypoint_label(agent_a, i_enter)}),"
                f" exit={i_exit} ({_waypoint_label(agent_a, i_exit)})"
            )
            print(
                f"    agent {b} interval: s=[{enc.interval_b.s_enter:.4f}, {enc.interval_b.s_exit:.4f}]"
                f" -> indices enter={j_enter} ({_waypoint_label(agent_b, j_enter)}),"
                f" exit={j_exit} ({_waypoint_label(agent_b, j_exit)})"
            )
            for text in row.get("lines", []):
                print(text)
    print("=== End formulation ===\n")


def expand_event_times_to_original_path(agent_path, event_times):
    """
    Linearly interpolate event times from interest waypoints onto ``original_path``.

    ``event_times[k]`` aligns with ``agent_path.interest_waypoints[k].s``.
    """
    original = agent_path.original_path
    if not original:
        return []
    if len(original) == 1:
        return [float(event_times[0]) if event_times else 0.0]

    s_vals = [float(wp.s) for wp in agent_path.interest_waypoints]
    t_vals = [float(t) for t in event_times]
    if len(s_vals) != len(t_vals):
        raise ValueError("event_times length must match interest_waypoints")

    cum = path_to_arc_length(original)
    out = []
    for s in cum:
        s = float(s)
        if s <= s_vals[0]:
            out.append(t_vals[0])
            continue
        if s >= s_vals[-1]:
            out.append(t_vals[-1])
            continue
        idx = int(np.searchsorted(s_vals, s, side="right") - 1)
        idx = max(0, min(idx, len(s_vals) - 2))
        ds = s_vals[idx + 1] - s_vals[idx]
        if ds <= 1e-12:
            out.append(t_vals[idx])
        else:
            alpha = (s - s_vals[idx]) / ds
            out.append(t_vals[idx] + alpha * (t_vals[idx + 1] - t_vals[idx]))
    return out


class EventMultiAgentPlanner:
    """
    Shared event-MILP builder for the sequential, simultaneous and combined planners.

    The three scenarios use identical constraint construction; they differ only in
    which agents are optimized (cvxpy variables) versus fixed (given schedules), which
    encounters are constrained, and the objective.
    """

    _context = "EventMultiAgentPlanner"
    _missing_inputs_message = "Paths not assigned."

    def __init__(self, occupancy_grid, paths=None, v=None):
        self.occupancy_grid = occupancy_grid
        self.paths = paths
        self.v = v if v is not None else MAX_VELOCITY
        self.analysis = None
        self.event_times = None
        self.mutex_constraint_count = 0
        self.num_z = 0

    def assign_path(self, paths):
        self.paths = paths

    def assign_velocities(self, v):
        self.v = v

    def _inputs_missing(self):
        return self.paths is None

    def _analysis_paths(self):
        return self.paths

    def analyze(self, threshold=ROBOT_DIAMETER):
        if self._inputs_missing():
            raise ValueError(self._missing_inputs_message)
        self.analysis = analyze_event_paths(self._analysis_paths(), threshold=threshold)
        return self.analysis

    def build_problem(self, analysis=None, threshold=ROBOT_DIAMETER, print_constraints=False):
        if analysis is None:
            analysis = self.analyze(threshold=threshold)
        else:
            self.analysis = analysis
        return self._build_for_analysis(analysis, print_constraints)

    def _build_for_analysis(self, analysis, print_constraints):
        raise NotImplementedError

    def _build_event_problem(
        self,
        analysis,
        *,
        velocities,
        agent_schedules,
        optimized_indices,
        kinematic_velocities,
        encounters,
        objective,
        print_norm,
        print_constraints,
    ):
        """
        Assemble the event MILP.

        ``agent_schedules[i]`` is a cvxpy Variable for optimized agents and a fixed
        float list otherwise. ``kinematic_velocities`` pairs each optimized index with
        its max velocity. ``objective`` maps the optimized variables to a cvxpy
        objective. Returns ``(prob, optimized_vars, num_z)``.
        """
        constraints = []
        for agent_idx, velocity in kinematic_velocities:
            _append_kinematic_constraints(
                constraints, analysis.agents[agent_idx], agent_schedules[agent_idx], velocity
            )

        encounter_rows = [_encounter_row(analysis, encounter) for encounter in encounters]
        num_z = sum(1 for row in encounter_rows if row["uses_z"])
        z = cp.Variable(num_z, boolean=True) if num_z > 0 else None
        big_m = _compute_big_m_horizon(analysis, velocities)
        mutex_constraint_count = _append_event_encounter_mutex(
            constraints,
            analysis,
            agent_schedules,
            encounter_rows,
            z,
            big_m,
            describe=print_constraints,
        )

        optimized_vars = [agent_schedules[idx] for idx in optimized_indices]
        prob = cp.Problem(objective(optimized_vars), constraints)

        self.num_z = num_z
        self.mutex_constraint_count = mutex_constraint_count
        _print_event_milp_summary(analysis, num_z, mutex_constraint_count)
        if print_constraints:
            _print_event_milp_constraints(
                analysis,
                velocities,
                norm=print_norm,
                big_m=big_m,
                encounter_rows=encounter_rows,
            )
        return prob, optimized_vars, num_z

    def _minimize_norm_of_final_times(self, optimized_vars):
        final_time_vars = cp.hstack([agent_time[-1] for agent_time in optimized_vars])
        return cp.Minimize(cp.norm(final_time_vars, p=self.norm))

    def _solve(self, prob, agent_times, verbose):
        self.event_times, self.solver_runtime_s = _solve_event_problem(
            prob,
            agent_times,
            context=self._context,
            verbose=verbose,
        )
        return self.event_times

    def plan(self, verbose=False, threshold=ROBOT_DIAMETER, print_constraints=False):
        if self._inputs_missing():
            raise ValueError(self._missing_inputs_message)
        prob, agent_times, _num_z = self.build_problem(
            threshold=threshold,
            print_constraints=print_constraints,
        )
        return self._solve(prob, agent_times, verbose)

    def plan_from_analysis(self, analysis, verbose=False, print_constraints=False):
        """Solve event MILP from a precomputed :class:`EventPathAnalysis`."""
        if self._inputs_missing():
            raise ValueError(self._missing_inputs_message)
        prob, agent_times, _num_z = self.build_problem(
            analysis=analysis,
            print_constraints=print_constraints,
        )
        return self._solve(prob, agent_times, verbose)


class EventMultiAgentSimultaneousPlanner(EventMultiAgentPlanner):
    """All agents are optimized; objective is the ``norm`` of their completion times."""

    _context = "EventMultiAgentSimultaneousPlanner"

    def __init__(self, occupancy_grid, paths=None, norm=1, v=None):
        super().__init__(occupancy_grid, paths=paths, v=v)
        self.norm = norm

    def _build_for_analysis(self, analysis, print_constraints):
        velocities = _broadcast_velocities(self.v, len(analysis.agents))
        agent_schedules = [
            cp.Variable(len(agent.refined_path)) for agent in analysis.agents
        ]
        indices = list(range(len(analysis.agents)))
        return self._build_event_problem(
            analysis,
            velocities=velocities,
            agent_schedules=agent_schedules,
            optimized_indices=indices,
            kinematic_velocities=list(zip(indices, velocities)),
            encounters=analysis.encounters,
            objective=self._minimize_norm_of_final_times,
            print_norm=self.norm,
            print_constraints=print_constraints,
        )


class EventMultiAgentSequentialPlanner(EventMultiAgentPlanner):
    """One ego agent (index 0) is optimized around fixed schedules of the others."""

    _context = "EventMultiAgentSequentialPlanner"
    _missing_inputs_message = "Path not assigned."

    def __init__(self, occupancy_grid, other_agent_paths, other_agent_times, path=None, v=None):
        super().__init__(occupancy_grid, v=v)
        self.path = path
        self.other_agent_paths = [list(path) for path in other_agent_paths]
        self.other_agent_times = [list(times) for times in other_agent_times]

    def assign_path(self, path):
        self.path = list(path)

    def _inputs_missing(self):
        return self.path is None

    def _analysis_paths(self):
        return [self.path] + self.other_agent_paths

    def _build_for_analysis(self, analysis, print_constraints):
        ego_idx = 0
        velocities = _broadcast_velocities(self.v, len(analysis.agents))
        agent_schedules = [None] * len(analysis.agents)
        ego_path = analysis.agents[ego_idx]
        agent_schedules[ego_idx] = cp.Variable(len(ego_path.refined_path))

        for other_idx, agent_path in enumerate(analysis.agents[1:], start=0):
            fixed_times = self.other_agent_times[other_idx]
            _validate_fixed_event_schedule(agent_path, fixed_times)
            agent_schedules[ego_idx + 1 + other_idx] = [float(t) for t in fixed_times]

        return self._build_event_problem(
            analysis,
            velocities=velocities,
            agent_schedules=agent_schedules,
            optimized_indices=[ego_idx],
            kinematic_velocities=[(ego_idx, velocities[ego_idx])],
            encounters=_sequential_event_encounters(analysis, ego_idx),
            objective=lambda optimized: cp.Minimize(optimized[0][-1]),
            print_norm=1,
            print_constraints=print_constraints,
        )


class EventMultiAgentCombinedPlanner(EventMultiAgentPlanner):
    """New agents (``paths``) are optimized around fixed schedules of ``other_agent_paths``."""

    _context = "EventMultiAgentCombinedPlanner"

    def __init__(
        self,
        occupancy_grid,
        other_agent_paths,
        other_agent_times,
        paths=None,
        v=None,
        norm=1,
    ):
        super().__init__(occupancy_grid, paths=paths, v=v)
        self.other_agent_paths = [list(path) for path in other_agent_paths]
        self.other_agent_times = [list(times) for times in other_agent_times]
        self.norm = norm

    def _analysis_paths(self):
        return self.other_agent_paths + list(self.paths)

    def _build_for_analysis(self, analysis, print_constraints):
        num_others = len(self.other_agent_paths)
        num_opt = len(self.paths)
        if num_opt < 1:
            raise ValueError("Combined planner requires at least one path to optimize.")

        opt_velocities = _broadcast_velocities(self.v, num_opt)
        velocities = [float(MAX_VELOCITY)] * num_others + opt_velocities

        agent_schedules = [None] * len(analysis.agents)
        for other_idx in range(num_others):
            agent_path = analysis.agents[other_idx]
            fixed_times = self.other_agent_times[other_idx]
            _validate_fixed_event_schedule(agent_path, fixed_times)
            agent_schedules[other_idx] = [float(t) for t in fixed_times]

        opt_indices = list(range(num_others, num_others + num_opt))
        for agent_idx in opt_indices:
            agent_schedules[agent_idx] = cp.Variable(len(analysis.agents[agent_idx].refined_path))

        other_indices = list(range(num_others))
        encounters = _cross_event_encounters(analysis, other_indices, opt_indices) + _within_event_encounters(
            analysis, opt_indices
        )
        return self._build_event_problem(
            analysis,
            velocities=velocities,
            agent_schedules=agent_schedules,
            optimized_indices=opt_indices,
            kinematic_velocities=[(agent_idx, velocities[agent_idx]) for agent_idx in opt_indices],
            encounters=encounters,
            objective=self._minimize_norm_of_final_times,
            print_norm=self.norm,
            print_constraints=print_constraints,
        )


def build_constant_velocity_schedules(analysis, event_times_list):
    """
    Build per-agent (path, arrival_times) for constant-velocity playback.

    Times are expanded onto ``original_path`` with speed uniform in arc-length
    between consecutive interest waypoints, so the robot follows the planned
    polyline rather than straight chords between event vertices.
    """
    paths = []
    times = []
    for agent_path, event_times in zip(analysis.agents, event_times_list):
        paths.append(list(agent_path.original_path))
        times.append(expand_event_times_to_original_path(agent_path, event_times))
    return paths, times


# Plotting helpers now live in ``trajectory_viz`` and the grid2 demo in
# ``demo_event_planning``; keep the old names importable from here.
_MOVED_NAMES = {
    "VIZ_EVENT_OUTPUT_DIR": "social_path_planning.trajectory_viz",
    "_AGENT_COLORS": "social_path_planning.trajectory_viz",
    "_WAYPOINT_MARKERS": "social_path_planning.trajectory_viz",
    "_plot_conflict_segments": "social_path_planning.trajectory_viz",
    "_plot_interest_waypoint": "social_path_planning.trajectory_viz",
    "viz_event_waypoints": "social_path_planning.trajectory_viz",
    "_save_encounter_panel": "social_path_planning.trajectory_viz",
    "create_event_video": "social_path_planning.trajectory_viz",
    "create_space_time_plot": "social_path_planning.trajectory_viz",
    "create_map_context_plot": "social_path_planning.trajectory_viz",
    "create_map_context_video": "social_path_planning.trajectory_viz",
    "_resolve_fixed_event_times": "social_path_planning.demo_event_planning",
    "_social_graph_from_heatmap": "social_path_planning.demo_event_planning",
    "_load_or_plan_path": "social_path_planning.demo_event_planning",
}


def __getattr__(name):
    module_name = _MOVED_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


if __name__ == "__main__":
    from social_path_planning.demo_event_planning import main

    main()
