"""Prioritized planning with path-length priority ordering."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from social_path_planning.mapf_adapter import Coord, MapfGridConfig, MapfPath, path_length_meters

CoordList = List[Coord]


def priority_order_by_path_length(routes: Sequence[dict]) -> List[int]:
    """
    Longer geometric path length = higher priority (planned first).
    Tie-break by route_id (larger route_id first when lengths are equal).
    """
    indexed = []
    for idx, route in enumerate(routes):
        length = path_length_meters(route.get("path") or [])
        route_id = int(route.get("route_id", idx))
        indexed.append((length, route_id, idx))
    indexed.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in indexed]


def _reservations_from_paths(committed: Sequence[MapfPath]) -> Dict[int, Set[Coord]]:
    """Space-time reservations from already planned agents."""
    dynamic: Dict[int, Set[Coord]] = {}
    for path in committed:
        if path is None or len(path) == 0:
            continue
        for t, point in enumerate(path):
            cell = (int(point[0]), int(point[1]))
            dynamic.setdefault(t, set()).add(cell)
    return dynamic


def run_prioritized_planning(
    grid_config: MapfGridConfig,
    starts: CoordList,
    goals: CoordList,
    routes: Sequence[dict],
    low_level_max_iter: int = 500,
    debug: bool = False,
) -> Tuple[Optional[List[MapfPath]], List[int], str]:
    """
    Plan agents in path-length priority order using space-time A*.

    Returns:
        (paths in original agent index order, priority order used, status)
    """
    if len(starts) != len(goals) or len(starts) != len(routes):
        raise ValueError("starts, goals, and routes must have the same length")

    try:
        from stastar.planner import Planner as STPlanner
    except ImportError as exc:
        raise ImportError("space-time-astar (stastar) is required: pip install cbs-mapf") from exc

    order = priority_order_by_path_length(routes)
    st_planner = STPlanner(
        grid_config.grid_size,
        grid_config.robot_radius,
        grid_config.static_obstacles,
    )

    committed_by_order: List[MapfPath] = []
    paths_by_agent: Dict[int, MapfPath] = {}

    for agent_idx in order:
        reservations = _reservations_from_paths(committed_by_order)
        path = st_planner.plan(
            starts[agent_idx],
            goals[agent_idx],
            dynamic_obstacles=reservations,
            semi_dynamic_obstacles=None,
            max_iter=int(low_level_max_iter),
            debug=debug,
        )
        if path is None or len(path) == 0:
            return None, order, "no_solution"
        path_arr = np.asarray(path, dtype=np.int32)
        committed_by_order.append(path_arr)
        paths_by_agent[agent_idx] = path_arr

    ordered_paths = [paths_by_agent[i] for i in range(len(starts))]
    return ordered_paths, order, "ok"
