"""Convert MAPF grid-time paths to world schedules at a shared max velocity."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from social_path_planning.mapf_adapter import MapfPath, cell_to_world
from social_path_planning.mapf_comparison.motion import MotionConfig

_EPS = 1e-9


def schedule_mapf_paths(
    occ_grid,
    grid_paths: Sequence[MapfPath],
    downsample: int,
    motion: MotionConfig,
) -> Tuple[List[List[Tuple[float, float]]], List[List[float]]]:
    """
    Build world waypoints and cumulative times for STA* paths (including waits).

    Move segment: Δt = dist_m / max_velocity_mps.
    Wait (zero dist): Δt = motion.wait_dt(resolution, downsample).
    """
    resolution = float(occ_grid.resolution)
    ds = max(1, int(downsample))
    wait_dt = motion.wait_dt(resolution, ds)
    v_max = max(float(motion.max_velocity_mps), _EPS)

    world_paths: List[List[Tuple[float, float]]] = []
    time_lists: List[List[float]] = []

    for path in grid_paths:
        if path is None or len(path) == 0:
            world_paths.append([])
            time_lists.append([])
            continue

        wp = [cell_to_world(occ_grid, (int(p[0]), int(p[1])), ds) for p in path]
        times = [0.0]
        for i in range(len(wp) - 1):
            p0 = np.array(wp[i], dtype=float)
            p1 = np.array(wp[i + 1], dtype=float)
            dist = float(np.linalg.norm(p1 - p0))
            if dist > _EPS:
                dt = dist / v_max
            else:
                dt = wait_dt
            times.append(times[-1] + dt)

        world_paths.append(wp)
        time_lists.append(times)

    return world_paths, time_lists
