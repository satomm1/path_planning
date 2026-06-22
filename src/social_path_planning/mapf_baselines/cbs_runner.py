"""Conflict-Based Search via cbs-mapf."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from social_path_planning.mapf_adapter import Coord, MapfGridConfig, MapfPath

CoordList = List[Coord]


def identity_assign(starts: CoordList, goals: CoordList):
    """Keep agent i paired with goals[i] (no Hungarian reassignment)."""
    from cbs_mapf.agent import Agent

    return [Agent(starts[i], goals[i]) for i in range(len(starts))]


def run_cbs(
    grid_config: MapfGridConfig,
    starts: CoordList,
    goals: CoordList,
    max_iter: int = 200,
    low_level_max_iter: int = 500,
    max_process: int = 1,
    debug: bool = False,
) -> Tuple[Optional[List[MapfPath]], str]:
    """
    Run CBS on integer grid coordinates.

    Returns:
        (list of per-agent paths, or None on failure), status message
    """
    if len(starts) != len(goals):
        raise ValueError("starts and goals must have the same length")
    if len(starts) == 0:
        return [], "empty"

    try:
        from cbs_mapf.planner import Planner
    except ImportError as exc:
        raise ImportError("cbs-mapf is required: pip install cbs-mapf") from exc

    planner = Planner(
        grid_size=grid_config.grid_size,
        robot_radius=grid_config.robot_radius,
        static_obstacles=grid_config.static_obstacles,
    )

    # Multiprocessing on Windows can be flaky; default max_process=1 for benchmarks.
    result = planner.plan(
        starts=list(starts),
        goals=list(goals),
        assign=identity_assign,
        max_iter=int(max_iter),
        low_level_max_iter=int(low_level_max_iter),
        max_process=int(max_process),
        debug=debug,
    )

    if result is None or (isinstance(result, np.ndarray) and result.size == 0):
        return None, "no_solution"

    paths: List[MapfPath] = []
    for agent_path in result:
        if agent_path is None or len(agent_path) == 0:
            return None, "empty_agent_path"
        paths.append(np.asarray(agent_path, dtype=np.int32))

    return paths, "ok"
