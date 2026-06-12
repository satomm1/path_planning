"""
Event-based multi-agent path analysis: conflict intervals and interest waypoints.

Phase 1: detect conflict boundaries, refine polylines, visualize interest waypoints.
Phase 2: event-based MILP on interest waypoints with per-encounter mutex constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
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
    create_map_context_plot,
    create_space_time_plot,
    create_video,
    path_to_arc_length,
)

MAX_VELOCITY = DEFAULT_MAX_VELOCITY_MPS


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


VIZ_EVENT_WAYPOINTS = False
VIZ_EVENT_OUTPUT_DIR = "results/event_planning_viz"

_AGENT_COLORS = ["steelblue", "darkorange", "seagreen", "purple", "brown", "teal"]
_WAYPOINT_MARKERS = {
    "start": ("o", 10),
    "conflict_enter": ("s", 9),
    "conflict_exit": ("^", 9),
    "goal": ("*", 12),
}


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
    kind: str  # "opposite" | "same"


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


def analyze_event_paths(paths, threshold=ROBOT_DIAMETER):
    """
    Analyze multi-agent paths and build interest waypoints at conflict boundaries.

    Returns:
        EventPathAnalysis with per-agent refined paths and pairwise encounter windows.
    """
    paths = [list(path) for path in paths]
    num_agents = len(paths)
    segment_indices_by_agent = [set() for _ in range(num_agents)]
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
                "kind": _encounter_kind_from_index_corners(segment_pairs),
            }
            for i, _j in segment_pairs:
                segment_indices_by_agent[a1].add(int(i))
            for _i, j in segment_pairs:
                segment_indices_by_agent[a2].add(int(j))

    agents = []
    encounter_intervals_by_agent = [[] for _ in range(num_agents)]
    encounters = []
    for (a1, a2), row in sorted(pair_conflicts.items()):
        seg_i = {int(i) for i, _j in row["pairs"]}
        seg_j = {int(j) for _i, j in row["pairs"]}
        kind = row["kind"]
        interval_a = _encounter_interval(a1, paths[a1], seg_i)
        interval_b = _encounter_interval(a2, paths[a2], seg_j)
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


def _encounter_goal_exit_uses_z(agent_a, agent_b, i_exit, j_exit):
    """
    Encounters need a binary only when neither or both agents end at goal.

    If exactly one agent's conflict exit is its goal, that robot must trail the
    other through the encounter; ordering is fixed and no z is required.
    """
    a_goal = _conflict_exit_is_goal(agent_a, i_exit)
    b_goal = _conflict_exit_is_goal(agent_b, j_exit)
    return not (a_goal ^ b_goal)


def _opposite_encounter_uses_z(agent_a, agent_b, i_exit, j_exit):
    """Backward-compatible alias for :func:`_encounter_goal_exit_uses_z`."""
    return _encounter_goal_exit_uses_z(agent_a, agent_b, i_exit, j_exit)


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
):
    """
    Opposite mutex: Big-M with z, or fixed ordering when one robot ends at goal.

    When agent A's conflict exit is its goal, B must clear before A enters and
    A must reach goal only after B has cleared. Symmetric when B ends at goal.
    """
    if not uses_z:
        a_goal = _conflict_exit_is_goal(agent_a, i_exit)
        if a_goal:
            constraints.append(times_b[j_exit] <= times_a[i_enter] - DELTA)
            constraints.append(times_a[i_exit] >= times_b[j_exit] + DELTA)
        else:
            constraints.append(times_a[i_exit] <= times_b[j_enter] - DELTA)
            constraints.append(times_b[j_exit] >= times_a[i_exit] + DELTA)
        return 2

    constraints.append(times_a[i_exit] <= times_b[j_enter] - DELTA + big_m * z_var[z_index])
    constraints.append(
        times_b[j_exit] <= times_a[i_enter] - DELTA - big_m * (1 - z_var[z_index])
    )
    return 2


def _filter_fixed_same_direction_pairs(paired_indices, agent_a, agent_b, *, a_goal):
    """
    Drop paired checkpoints that contradict fixed trailing-goal ordering with t[0]=0.

    Requiring the leader to be ahead of the other agent's path start is impossible
    when both schedules begin at zero.
    """
    filtered = []
    for ia, ib in paired_indices:
        if a_goal:
            waypoint = agent_b.interest_waypoints[ib]
            if waypoint.kind == "start" and ib == 0:
                continue
        else:
            waypoint = agent_a.interest_waypoints[ia]
            if waypoint.kind == "start" and ia == 0:
                continue
        filtered.append((int(ia), int(ib)))
    return filtered


def _same_direction_pairs_for_mutex(agent_a, agent_b, interval_a, interval_b, *, uses_z):
    """Return paired checkpoints used for same-direction mutex constraints."""
    pairs = _pair_same_direction_checkpoints(agent_a, agent_b, interval_a, interval_b)
    if uses_z:
        return pairs
    i_enter, i_exit = _refined_indices_for_interval(agent_a, interval_a)
    a_goal = _conflict_exit_is_goal(agent_a, i_exit)
    return _filter_fixed_same_direction_pairs(pairs, agent_a, agent_b, a_goal=a_goal)


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
    i_exit,
    j_exit,
    uses_z,
):
    """
    Mutex for same-direction co-marching.

    With z: two Big-M rows per paired checkpoint (leader chosen by z).
    Without z: exactly one agent ends at goal; it trails at every paired checkpoint.
    """
    if not uses_z:
        a_goal = _conflict_exit_is_goal(agent_a, i_exit)
        for ia, ib in paired_indices:
            if a_goal:
                constraints.append(times_b[ib] <= times_a[ia] - DELTA)
            else:
                constraints.append(times_a[ia] <= times_b[ib] - DELTA)
        return len(paired_indices)

    z = z_var[z_index]
    for ia, ib in paired_indices:
        constraints.append(times_a[ia] <= times_b[ib] - DELTA + big_m * z)
        constraints.append(times_b[ib] <= times_a[ia] - DELTA + big_m * (1 - z))
    return 2 * len(paired_indices)


def _add_encounter_mutex_constraints(
    constraints,
    times_a,
    times_b,
    z_var,
    z_index,
    big_m,
    encounter,
    agent_a,
    agent_b,
):
    """Append mutex constraints for one encounter window."""
    i_enter, i_exit = _refined_indices_for_interval(agent_a, encounter.interval_a)
    j_enter, j_exit = _refined_indices_for_interval(agent_b, encounter.interval_b)
    if encounter.kind == "opposite":
        uses_z = _encounter_goal_exit_uses_z(agent_a, agent_b, i_exit, j_exit)
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
        )
        return added, [], uses_z
    uses_z = _encounter_goal_exit_uses_z(agent_a, agent_b, i_exit, j_exit)
    paired_indices = _same_direction_pairs_for_mutex(
        agent_a,
        agent_b,
        encounter.interval_a,
        encounter.interval_b,
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
        i_exit=i_exit,
        j_exit=j_exit,
        uses_z=uses_z,
    )
    return added, paired_indices, uses_z


def _mutex_constraint_count_for_encounters(analysis):
    """Return total mutex rows for all encounter windows."""
    total = 0
    for encounter in analysis.encounters:
        agent_a = analysis.agents[encounter.agent_a]
        agent_b = analysis.agents[encounter.agent_b]
        i_enter, i_exit = _refined_indices_for_interval(agent_a, encounter.interval_a)
        j_enter, j_exit = _refined_indices_for_interval(agent_b, encounter.interval_b)
        uses_z = _encounter_goal_exit_uses_z(agent_a, agent_b, i_exit, j_exit)
        if encounter.kind == "opposite":
            total += 2
            continue
        pairs = _same_direction_pairs_for_mutex(
            agent_a,
            agent_b,
            encounter.interval_a,
            encounter.interval_b,
            uses_z=uses_z,
        )
        total += (2 if uses_z else 1) * len(pairs)
    return total


def _encounter_uses_z(analysis, encounter):
    """Return whether an encounter allocates a binary decision variable."""
    agent_a = analysis.agents[encounter.agent_a]
    agent_b = analysis.agents[encounter.agent_b]
    i_enter, i_exit = _refined_indices_for_interval(agent_a, encounter.interval_a)
    j_enter, j_exit = _refined_indices_for_interval(agent_b, encounter.interval_b)
    return _encounter_goal_exit_uses_z(agent_a, agent_b, i_exit, j_exit)


def _print_event_milp_summary(analysis, num_z, mutex_constraint_count):
    interest_total = sum(len(agent.interest_waypoints) for agent in analysis.agents)
    opposite_count = sum(1 for enc in analysis.encounters if enc.kind == "opposite")
    same_count = len(analysis.encounters) - opposite_count
    print(
        "Event MILP: "
        f"{len(analysis.agents)} agents, {interest_total} interest waypoints, "
        f"{len(analysis.encounters)} encounters ({opposite_count} opposite, {same_count} same), "
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
            if enc.kind == "opposite":
                if uses_z:
                    print(
                        f"    (1) t_{a}[{i_exit}] <= t_{b}[{j_enter}] - {float(DELTA):.4f}"
                        f" + {float(big_m):.4f} * z_{z_index}"
                        f"   # agent {a} clears before agent {b} enters"
                    )
                    print(
                        f"    (2) t_{b}[{j_exit}] <= t_{a}[{i_enter}] - {float(DELTA):.4f}"
                        f" - {float(big_m):.4f} * (1 - z_{z_index})"
                        f"   # agent {b} clears before agent {a} enters"
                    )
                elif _conflict_exit_is_goal(agent_a, i_exit):
                    print(
                        f"    (1) t_{b}[{j_exit}] <= t_{a}[{i_enter}] - {float(DELTA):.4f}"
                        f"   # agent {b} clears before agent {a} enters"
                    )
                    print(
                        f"    (2) t_{a}[{i_exit}] >= t_{b}[{j_exit}] + {float(DELTA):.4f}"
                        f"   # agent {a} reaches goal after agent {b} cleared"
                    )
                else:
                    print(
                        f"    (1) t_{a}[{i_exit}] <= t_{b}[{j_enter}] - {float(DELTA):.4f}"
                        f"   # agent {a} clears before agent {b} enters"
                    )
                    print(
                        f"    (2) t_{b}[{j_exit}] >= t_{a}[{i_exit}] + {float(DELTA):.4f}"
                        f"   # agent {b} reaches goal after agent {a} cleared"
                    )
            else:
                paired_indices = row.get("paired_indices", [])
                uses_z = row.get("uses_z", True)
                for pair_no, (ia, ib) in enumerate(paired_indices, start=1):
                    if uses_z:
                        print(
                            f"    ({pair_no}a) t_{a}[{ia}] ({_waypoint_label(agent_a, ia)})"
                            f" <= t_{b}[{ib}] ({_waypoint_label(agent_b, ib)})"
                            f" - {float(DELTA):.4f} + {float(big_m):.4f} * z_{z_index}"
                            f"   # z=0: agent {a} ahead"
                        )
                        print(
                            f"    ({pair_no}b) t_{b}[{ib}] ({_waypoint_label(agent_b, ib)})"
                            f" <= t_{a}[{ia}] ({_waypoint_label(agent_a, ia)})"
                            f" - {float(DELTA):.4f} + {float(big_m):.4f} * (1 - z_{z_index})"
                            f"   # z=1: agent {b} ahead"
                        )
                    elif _conflict_exit_is_goal(agent_a, i_exit):
                        print(
                            f"    ({pair_no}) t_{b}[{ib}] ({_waypoint_label(agent_b, ib)})"
                            f" <= t_{a}[{ia}] ({_waypoint_label(agent_a, ia)})"
                            f" - {float(DELTA):.4f}"
                            f"   # agent {b} leads, agent {a} reaches goal last"
                        )
                    else:
                        print(
                            f"    ({pair_no}) t_{a}[{ia}] ({_waypoint_label(agent_a, ia)})"
                            f" <= t_{b}[{ib}] ({_waypoint_label(agent_b, ib)})"
                            f" - {float(DELTA):.4f}"
                            f"   # agent {a} leads, agent {b} reaches goal last"
                        )
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
    def __init__(self, occupancy_grid, paths=None, v=None):
        self.occupancy_grid = occupancy_grid
        self.paths = paths
        self.v = v if v is not None else MAX_VELOCITY
        self.analysis = None
        self.event_times = None

    def assign_path(self, paths):
        self.paths = paths

    def assign_velocities(self, v):
        self.v = v


class EventMultiAgentSimultaneousPlanner(EventMultiAgentPlanner):
    def __init__(self, occupancy_grid, paths=None, norm=1, v=None):
        super().__init__(occupancy_grid, paths=paths, v=v)
        self.norm = norm
        self.mutex_constraint_count = 0
        self.num_z = 0

    def analyze(self, threshold=ROBOT_DIAMETER):
        if self.paths is None:
            raise ValueError("Paths not assigned.")
        self.analysis = analyze_event_paths(self.paths, threshold=threshold)
        return self.analysis

    def build_problem(self, analysis=None, threshold=ROBOT_DIAMETER, print_constraints=False):
        if analysis is None:
            analysis = self.analyze(threshold=threshold)
        else:
            self.analysis = analysis

        if len(self.v) == 1:
            velocities = [self.v[0] for _ in range(len(analysis.agents))]
        else:
            velocities = list(self.v)

        agent_times = [
            cp.Variable(len(agent.refined_path)) for agent in analysis.agents
        ]
        constraints = []
        for agent_time in agent_times:
            constraints.append(agent_time[0] == 0)

        for agent_idx, agent_path in enumerate(analysis.agents):
            velocity = max(float(velocities[agent_idx]), 1e-6)
            agent_time = agent_times[agent_idx]
            for k, seg_len in enumerate(_interest_segment_lengths(agent_path)):
                constraints.append(agent_time[k + 1] - agent_time[k] >= seg_len / velocity)

        big_m = _compute_big_m_horizon(analysis, velocities)
        encounter_specs = []
        for encounter in analysis.encounters:
            agent_a = analysis.agents[encounter.agent_a]
            agent_b = analysis.agents[encounter.agent_b]
            i_enter, i_exit = _refined_indices_for_interval(agent_a, encounter.interval_a)
            j_enter, j_exit = _refined_indices_for_interval(agent_b, encounter.interval_b)
            encounter_specs.append(
                {
                    "encounter": encounter,
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "i_enter": i_enter,
                    "i_exit": i_exit,
                    "j_enter": j_enter,
                    "j_exit": j_exit,
                    "uses_z": _encounter_uses_z(analysis, encounter),
                }
            )

        num_z = sum(1 for spec in encounter_specs if spec["uses_z"])
        z = cp.Variable(num_z, boolean=True) if num_z > 0 else None
        mutex_constraint_count = 0
        encounter_rows = []
        z_slot = 0

        for spec in encounter_specs:
            encounter = spec["encounter"]
            agent_a = spec["agent_a"]
            agent_b = spec["agent_b"]
            uses_z = spec["uses_z"]
            z_index = z_slot if uses_z else None
            added, paired_indices, _uses_z = _add_encounter_mutex_constraints(
                constraints,
                agent_times[encounter.agent_a],
                agent_times[encounter.agent_b],
                z,
                z_index,
                big_m,
                encounter,
                agent_a,
                agent_b,
            )
            if uses_z:
                z_slot += 1
            encounter_rows.append(
                {
                    "z_index": z_index,
                    "uses_z": uses_z,
                    "encounter": encounter,
                    "i_enter": spec["i_enter"],
                    "i_exit": spec["i_exit"],
                    "j_enter": spec["j_enter"],
                    "j_exit": spec["j_exit"],
                    "paired_indices": paired_indices,
                }
            )
            mutex_constraint_count += added

        final_time_vars = cp.hstack([agent_time[-1] for agent_time in agent_times])
        objective = cp.Minimize(cp.norm(final_time_vars, p=self.norm))
        prob = cp.Problem(objective, constraints)

        self.num_z = num_z
        self.mutex_constraint_count = mutex_constraint_count
        _print_event_milp_summary(analysis, num_z, mutex_constraint_count)
        if print_constraints:
            _print_event_milp_constraints(
                analysis,
                velocities,
                norm=self.norm,
                big_m=big_m,
                encounter_rows=encounter_rows,
            )
        return prob, agent_times, num_z

    def _solve_problem(self, prob, agent_times, verbose=False):
        print("Starting to solve event multi-agent planning problem...")
        solver_chain = [
            name
            for name in (cp.GLPK_MI, cp.HIGHS, cp.SCIPY)
            if name in cp.installed_solvers()
        ]
        if not solver_chain:
            raise RuntimeError("No cvxpy solver available for event MILP")

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

        return _multi_agent_times_from_solver(
            prob, agent_times, "EventMultiAgentSimultaneousPlanner"
        )

    def plan(self, verbose=False, threshold=ROBOT_DIAMETER, print_constraints=False):
        if self.paths is None:
            raise ValueError("Paths not assigned.")
        prob, agent_times, _num_z = self.build_problem(
            threshold=threshold,
            print_constraints=print_constraints,
        )
        self.event_times = self._solve_problem(prob, agent_times, verbose=verbose)
        return self.event_times


def _plot_conflict_segments(ax, path, intervals, color="crimson"):
    for interval in intervals:
        for seg in range(interval.seg_start, interval.seg_end + 1):
            x0, y0 = path[seg]
            x1, y1 = path[seg + 1]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=color,
                linewidth=5.0,
                alpha=0.9,
                solid_capstyle="round",
                zorder=8,
            )


def _plot_interest_waypoint(ax, waypoint, color):
    marker, size = _WAYPOINT_MARKERS.get(waypoint.kind, ("o", 8))
    x, y = waypoint.point
    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=size**2,
        color=color,
        edgecolors="black",
        linewidths=0.6,
        zorder=10,
        label=waypoint.kind if waypoint.kind not in getattr(ax, "_event_labels", set()) else None,
    )
    labels = getattr(ax, "_event_labels", set())
    labels.add(waypoint.kind)
    ax._event_labels = labels


def viz_event_waypoints(
    analysis,
    *,
    occ_grid=None,
    output_path=None,
    also_write_pair_panels=True,
):
    """Save a multi-agent figure highlighting conflict segments and interest waypoints."""
    output_path = Path(output_path or (VIZ_EVENT_OUTPUT_DIR + "/waypoints.png"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    if occ_grid is not None:
        occ_grid.plot_grid(ax=ax)
    ax._event_labels = set()

    for agent_path in analysis.agents:
        color = _AGENT_COLORS[agent_path.agent % len(_AGENT_COLORS)]
        path = agent_path.original_path
        if len(path) >= 2:
            ax.plot(
                [p[0] for p in path],
                [p[1] for p in path],
                color=color,
                linewidth=2.0,
                alpha=0.55,
                label=f"agent {agent_path.agent}",
                zorder=5,
            )
        _plot_conflict_segments(ax, path, agent_path.conflict_intervals)
        for waypoint in agent_path.interest_waypoints:
            _plot_interest_waypoint(ax, waypoint, color)

    opposite_count = sum(1 for enc in analysis.encounters if enc.kind == "opposite")
    same_count = len(analysis.encounters) - opposite_count
    interest_total = sum(len(agent.interest_waypoints) for agent in analysis.agents)
    ax.set_title(
        "Event interest waypoints: "
        f"{len(analysis.agents)} agents, {interest_total} waypoints, "
        f"{len(analysis.encounters)} encounters "
        f"({opposite_count} opposite, {same_count} same)"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved event waypoint visualization: {output_path.resolve()}")

    if also_write_pair_panels and analysis.encounters:
        for encounter in analysis.encounters:
            pair_path = output_path.parent / f"encounter_{encounter.agent_a}_{encounter.agent_b}.png"
            _save_encounter_panel(analysis, encounter, occ_grid=occ_grid, output_path=pair_path)

    return output_path


def _save_encounter_panel(analysis, encounter, *, occ_grid=None, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    if occ_grid is not None:
        occ_grid.plot_grid(ax=ax)

    for agent_idx in (encounter.agent_a, encounter.agent_b):
        agent_path = analysis.agents[agent_idx]
        color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
        path = agent_path.original_path
        ax.plot(
            [p[0] for p in path],
            [p[1] for p in path],
            color=color,
            linewidth=2.0,
            alpha=0.55,
            label=f"agent {agent_idx}",
            zorder=5,
        )
        interval = (
            encounter.interval_a if agent_idx == encounter.agent_a else encounter.interval_b
        )
        enter = point_at_arc_length(path, interval.s_enter)
        exit_pt = point_at_arc_length(path, interval.s_exit)
        ax.plot(
            [enter[0], exit_pt[0]],
            [enter[1], exit_pt[1]],
            color=color,
            linewidth=7.0,
            alpha=0.25,
            solid_capstyle="round",
            zorder=6,
        )
        _plot_conflict_segments(ax, path, [interval], color="crimson")
        for waypoint in agent_path.interest_waypoints:
            _plot_interest_waypoint(ax, waypoint, color)

    ax.set_title(
        f"Encounter ({encounter.agent_a},{encounter.agent_b}) kind={encounter.kind}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved encounter visualization: {Path(output_path).resolve()}")


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


def create_event_video(
    analysis,
    event_times_list,
    *,
    output_file="video_event.gif",
    occ_grid=None,
):
    """Render a multi-agent animation from event MILP schedules."""
    paths, times = build_constant_velocity_schedules(analysis, event_times_list)
    create_video(paths, times, output_file=output_file, occ_grid=occ_grid)


def _load_or_plan_path(occ_grid, map_size, map_resolution, pickle_name, x_init, x_goal):
    """Load a cached A* path or plan and pickle it (grid2 demo helper)."""
    import pickle

    from social_path_planning.a_star import AStar
    from social_path_planning.utils import snap_to_grid

    try:
        with open(pickle_name, "rb") as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        problem = AStar(
            [0, 0],
            snap_to_grid(map_size, map_resolution),
            snap_to_grid(x_init, map_resolution),
            snap_to_grid(x_goal, map_resolution),
            occ_grid,
            resolution=map_resolution,
        )
        if not problem.solve():
            raise RuntimeError(f"A* failed for {pickle_name}")
        path = problem.path
        with open(pickle_name, "wb") as handle:
            pickle.dump(path, handle)
        return path


if __name__ == "__main__":
    from social_path_planning.grid_loader import load_grid_scenario
    from social_path_planning.occupancy_grid import StochOccupancyGrid2D

    scenario_name = "sample2_default"  # grid2 / four-quadrant benchmark map
    output_dir = Path("results/event_planning_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    occ_grid = StochOccupancyGrid2D(
        map_resolution,
        round(map_size[0] / map_resolution),
        round(map_size[1] / map_resolution),
        0,
        0,
        10,
        occ.T,
    )

    path1 = _load_or_plan_path(
        occ_grid, map_size, map_resolution, "path1.pkl", [2, 25], [97, 65]
    )
    path2 = _load_or_plan_path(
        occ_grid, map_size, map_resolution, "path2.pkl", [46, 80], [46, 20]
    )
    path3 = _load_or_plan_path(
        occ_grid, map_size, map_resolution, "path3.pkl", [75, 48], [65, 45.5]
    )
    path4 = _load_or_plan_path(
        occ_grid, map_size, map_resolution, "path4.pkl", [50, 20], [99, 60]
    )

    stride = 1
    stage_paths = [path1[::stride], path2[::stride], path3[::stride], path4[::stride]]

    planner = EventMultiAgentSimultaneousPlanner(occ_grid, paths=stage_paths, norm=1)
    planner.assign_velocities(
        [MAX_VELOCITY, MAX_VELOCITY / 1.2, MAX_VELOCITY / 1.5, MAX_VELOCITY / 1.65]
    )

    print("Running event MILP on grid2 (sample2_default)...")
    try:
        event_times = planner.plan(verbose=True, print_constraints=True)
        for event_time in event_times:
            print(event_time)
    except Exception as exc:
        print(f"Event MILP failed: {type(exc).__name__}: {exc}")
        raise

    analysis = planner.analysis
    anim_paths, anim_times = build_constant_velocity_schedules(analysis, event_times)

    viz_event_waypoints(
        analysis,
        occ_grid=occ_grid,
        output_path=output_dir / "grid2_event_waypoints.png",
        also_write_pair_panels=True,
    )
    create_space_time_plot(
        anim_paths,
        anim_times,
        output_file=str(output_dir / "grid2_space_time_event.png"),
        title="Event MILP Space-Time Plot (constant velocity segments)",
    )
    snapshot_time = 0.35 * max(t_seq[-1] for t_seq in anim_times)
    create_map_context_plot(
        anim_paths,
        occ_grid=occ_grid,
        times=anim_times,
        snapshot_time=snapshot_time,
        output_file=str(output_dir / "grid2_map_context_event.png"),
        title="Event MILP Paths on Grid2",
    )
    create_event_video(
        analysis,
        event_times,
        output_file=str(output_dir / "grid2_video_event.gif"),
        occ_grid=occ_grid,
    )
    print(
        "Event MILP demo complete: "
        f"{planner.num_z} binaries, {planner.mutex_constraint_count} mutex constraints, "
        f"outputs in {output_dir.resolve()}"
    )
