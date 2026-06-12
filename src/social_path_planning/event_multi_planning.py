"""
Event-based multi-agent path analysis: conflict intervals and interest waypoints.

Phase 1: detect conflict boundaries, refine polylines, visualize interest waypoints.
MILP timing optimization is deferred to a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from social_path_planning.multi_planning import (
    GEOM_MATCH_TOL,
    ROBOT_DIAMETER,
    _iter_conflicting_segment_pairs,
    _maximal_consecutive_runs,
    _tag_opposite_segment_pairs,
    path_to_arc_length,
)

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


def _build_interest_waypoints(agent, path, intervals):
    cum = path_to_arc_length(path)
    total = float(cum[-1]) if cum else 0.0
    s_values = {0.0, total}
    for interval in intervals:
        s_values.add(float(interval.s_enter))
        s_values.add(float(interval.s_exit))

    ordered_s = sorted(s_values)
    exit_s = {float(iv.s_exit) for iv in intervals}

    waypoints = []
    for s in ordered_s:
        if abs(s) <= GEOM_MATCH_TOL:
            kind = "start"
        elif abs(s - total) <= GEOM_MATCH_TOL:
            kind = "goal"
        elif any(abs(float(s) - val) <= GEOM_MATCH_TOL for val in exit_s):
            kind = "conflict_exit"
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
            opposite_pairs = _tag_opposite_segment_pairs(path_a, path_b, segment_pairs)
            pair_conflicts[(a1, a2)] = {
                "pairs": segment_pairs,
                "opposite": opposite_pairs,
            }
            for i, _j in segment_pairs:
                segment_indices_by_agent[a1].add(int(i))
            for _i, j in segment_pairs:
                segment_indices_by_agent[a2].add(int(j))

    agents = []
    for agent, path in enumerate(paths):
        intervals = _intervals_from_segment_indices(agent, path, segment_indices_by_agent[agent])
        interest = _build_interest_waypoints(agent, path, intervals)
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

    encounters = []
    for (a1, a2), row in sorted(pair_conflicts.items()):
        seg_i = {int(i) for i, _j in row["pairs"]}
        seg_j = {int(j) for _i, j in row["pairs"]}
        kind = "opposite" if row["opposite"] else "same"
        encounters.append(
            EncounterWindow(
                agent_a=a1,
                agent_b=a2,
                interval_a=_encounter_interval(a1, paths[a1], seg_i),
                interval_b=_encounter_interval(a2, paths[a2], seg_j),
                kind=kind,
            )
        )

    return EventPathAnalysis(agents=agents, encounters=encounters)


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
