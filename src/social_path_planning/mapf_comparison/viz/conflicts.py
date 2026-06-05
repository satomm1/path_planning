"""Conflict detection for visualization."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from social_path_planning.mapf_adapter import MapfPath
from social_path_planning.mapf_comparison.grid_traversability import ROBOT_DIAMETER_M
from social_path_planning.mapf_comparison.viz.interpolation import get_position_at_time


def pairwise_conflict_at_time_world(
    world_paths: Sequence[Sequence[Tuple[float, float]]],
    time_lists: Sequence[Sequence[float]],
    t: float,
    threshold_m: float,
) -> List[Tuple[int, int]]:
    positions = []
    for path, times in zip(world_paths, time_lists):
        if not path or not times:
            return []
        positions.append(get_position_at_time(t, path, times))
    pairs = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pi = np.array(positions[i], dtype=float)
            pj = np.array(positions[j], dtype=float)
            if float(np.linalg.norm(pi - pj)) <= threshold_m:
                pairs.append((i, j))
    return pairs


def pairwise_conflict_at_time(
    paths: Sequence[MapfPath],
    t: int,
    robot_radius: int,
) -> List[Tuple[int, int]]:
    positions = []
    for path in paths:
        if path is None or len(path) == 0:
            return []
        idx = min(t, len(path) - 1)
        positions.append(np.asarray(path[idx], dtype=float))
    pairs = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if int(np.linalg.norm(positions[i] - positions[j], 2)) <= 2 * robot_radius:
                pairs.append((i, j))
    return pairs


DEFAULT_CONFLICT_THRESHOLD_M = ROBOT_DIAMETER_M
