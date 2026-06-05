"""Unified travel-time and path-length metrics from timed world polylines."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from social_path_planning.mapf_adapter import path_length_meters

_EPS = 1e-9


def dedupe_world_path(path: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    prev = None
    for p in path:
        pt = (float(p[0]), float(p[1]))
        if pt != prev:
            out.append(pt)
            prev = pt
    return out


def compute_solution_metrics(
    world_paths: Sequence[Sequence[Tuple[float, float]]],
    time_lists: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """
    Completion times and geometric path length for executed trajectories.

    ``time_lists[i][-1]`` is agent i completion time; path length uses deduped
    world waypoints (waits do not add length).
    """
    per_agent_length: List[float] = []
    per_agent_completion: List[float] = []
    per_agent_max_speed: List[float] = []

    for path, times in zip(world_paths, time_lists):
        if not path or not times or len(path) != len(times):
            per_agent_length.append(0.0)
            per_agent_completion.append(0.0)
            per_agent_max_speed.append(0.0)
            continue

        deduped = dedupe_world_path(path)
        per_agent_length.append(path_length_meters(deduped))
        per_agent_completion.append(float(times[-1]))

        max_speed = 0.0
        for i in range(len(path) - 1):
            dt = float(times[i + 1] - times[i])
            if dt <= _EPS:
                continue
            p0 = np.array(path[i], dtype=float)
            p1 = np.array(path[i + 1], dtype=float)
            dist = float(np.linalg.norm(p1 - p0))
            if dist > _EPS:
                max_speed = max(max_speed, dist / dt)
        per_agent_max_speed.append(max_speed)

    total_length = float(sum(per_agent_length)) if per_agent_length else 0.0
    completions = [t for t in per_agent_completion if t > 0 or len(world_paths) == 1]

    soc_seconds = float(sum(per_agent_completion)) if per_agent_completion else None
    makespan_seconds = float(max(per_agent_completion)) if per_agent_completion else None

    return {
        "soc_seconds": soc_seconds,
        "makespan_seconds": makespan_seconds,
        "per_agent_path_length_m": per_agent_length,
        "total_path_length_m": total_length,
        "per_agent_completion_s": per_agent_completion,
        "per_agent_max_speed_mps": per_agent_max_speed,
    }
