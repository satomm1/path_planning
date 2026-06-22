"""MAPF grid config, search budgets, coords, and conflict checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from social_path_planning.mapf_comparison.grid_traversability import (
    COARSE_BLOCK_POLICIES,
    DEFAULT_COARSE_BLOCK_POLICY,
    ROBOT_DIAMETER_M,
    CoarseBlockPolicy,
    TraversabilityModel,
    mapf_cell_to_world,
    normalize_coarse_block_policy,
    world_to_mapf_cell,
)

__all__ = [
    "COARSE_BLOCK_POLICIES",
    "DEFAULT_COARSE_BLOCK_POLICY",
    "DEFAULT_ROBOT_DIAMETER_M",
    "MapfGridConfig",
    "MapfPath",
    "TraversabilityModel",
    "build_mapf_grid_config",
    "build_static_obstacles",
    "cell_to_world",
    "make_traversability_model",
    "mapf_path_to_world_polyline",
    "normalize_coarse_block_policy",
    "resolve_mapf_search_budget",
    "starts_goals_from_routes",
    "validate_mapf_solution",
    "world_to_cell",
]

# Back-compat aliases
DEFAULT_ROBOT_DIAMETER_M = ROBOT_DIAMETER_M

Coord = Tuple[int, int]
MapfPath = np.ndarray  # shape (T, 2) integer positions per timestep

_GRID_CONFIG_CACHE: dict[tuple, MapfGridConfig] = {}


@dataclass(frozen=True)
class MapfGridConfig:
    """Parameters for cbs-mapf / space-time A* on a discretized occupancy grid."""

    grid_size: int
    robot_radius: int
    static_obstacles: List[Coord]
    width_cells: int
    height_cells: int
    crop_bounds: Tuple[int, int, int, int] | None = None
    downsample: int = 1
    coarse_block_policy: CoarseBlockPolicy = DEFAULT_COARSE_BLOCK_POLICY


def mapf_robot_radius_cells() -> int:
    """
    Radius passed to cbs-mapf / stastar.

    Use 0 so static clearance is "cell center not in obstacle list".
    """
    return 0


def validation_robot_radius_cells(
    resolution: float, robot_diameter: float = ROBOT_DIAMETER_M
) -> int:
    """Radius in coarse cells for post-hoc agent-agent conflict checks."""
    return max(1, int(math.ceil(robot_diameter / (2.0 * resolution))))


def manhattan_cell_distance(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


REFERENCE_MAPF_DOWNSAMPLE = 4


def downsample_search_scale(downsample: int, reference_ds: int = REFERENCE_MAPF_DOWNSAMPLE) -> float:
    ds = max(1, int(downsample))
    ref = max(1, int(reference_ds))
    return float(ref / ds) ** 1.5


def effective_crop_padding_cells(
    padding_cells: int,
    downsample: int,
    reference_ds: int = REFERENCE_MAPF_DOWNSAMPLE,
) -> int:
    ds = max(1, int(downsample))
    ref = max(1, int(reference_ds))
    return max(int(padding_cells), int(round(int(padding_cells) * ref / ds)))


def suggest_low_level_max_iter(
    starts: Sequence[Coord],
    goals: Sequence[Coord],
    num_agents: int = 1,
    scale: float = 8.0,
    floor: int = 2000,
    ceiling: int = 100000,
    downsample: int = 1,
) -> int:
    if not starts or not goals:
        return floor
    longest = max(manhattan_cell_distance(s, g) for s, g in zip(starts, goals))
    ds = max(1, int(downsample))
    effort = downsample_search_scale(ds)
    estimate = int(scale * longest * max(1, num_agents) * effort)
    if ds >= 4:
        floor = max(floor, 8000)
    elif ds >= 2:
        floor = max(floor, 10000)
    else:
        floor = max(floor, 15000)
    return int(min(ceiling, max(floor, estimate)))


def suggest_cbs_max_iter(
    num_agents: int = 2,
    downsample: int = 1,
    base: int = 300,
    ceiling: int = 5000,
) -> int:
    ds = max(1, int(downsample))
    scaled_ceiling = int(min(20000, ceiling * max(1.0, num_agents / 4.0)))
    estimate = int(base * downsample_search_scale(ds) * max(1, num_agents) ** 0.5)
    floor = max(base, int(400 * downsample_search_scale(ds)))
    return int(min(scaled_ceiling, max(floor, estimate)))


def resolve_mapf_search_budget(
    starts: Sequence[Coord],
    goals: Sequence[Coord],
    num_agents: int,
    downsample: int,
    crop_padding_cells: int = 40,
    pp_low_level_max_iter: int = 0,
    cbs_low_level_max_iter: int = 0,
    cbs_max_iter: int = 0,
) -> dict:
    ds = max(1, int(downsample))
    low_level = int(
        min(200000, suggest_low_level_max_iter(starts, goals, num_agents=num_agents, downsample=ds))
    )
    if pp_low_level_max_iter > 0:
        low_level_pp = int(pp_low_level_max_iter)
    else:
        low_level_pp = low_level
    if cbs_low_level_max_iter > 0:
        low_level_cbs = int(cbs_low_level_max_iter)
    else:
        low_level_cbs = low_level
    if cbs_max_iter > 0:
        cbs_hi = int(cbs_max_iter)
    else:
        cbs_hi = suggest_cbs_max_iter(num_agents=num_agents, downsample=ds)
    return {
        "downsample": ds,
        "crop_padding_cells": effective_crop_padding_cells(crop_padding_cells, ds),
        "pp_low_level_max_iter": low_level_pp,
        "cbs_low_level_max_iter": low_level_cbs,
        "cbs_max_iter": cbs_hi,
    }


def default_mapf_downsample(occ_grid) -> int:
    side = max(int(occ_grid.width), int(occ_grid.height))
    if side >= 400:
        return 4
    if side >= 200:
        return 2
    return 1


def world_to_cell(occ_grid, xy: Sequence[float], downsample: int = 1) -> Coord:
    return world_to_mapf_cell(occ_grid, xy, downsample)


def cell_to_world(occ_grid, cell: Coord, downsample: int = 1) -> Tuple[float, float]:
    return mapf_cell_to_world(occ_grid, cell, downsample)


def path_length_meters(path: Sequence[Sequence[float]]) -> float:
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        p0 = np.array(path[i], dtype=float)
        p1 = np.array(path[i + 1], dtype=float)
        total += float(np.linalg.norm(p1 - p0))
    return total


def starts_goals_from_routes(
    routes: Sequence[dict],
    occ_grid,
    downsample: int = 1,
) -> Tuple[List[Coord], List[Coord]]:
    model = TraversabilityModel(occ_grid, downsample=downsample)
    starts = [model.world_to_cell(route["x_init"]) for route in routes]
    goals = [model.world_to_cell(route["x_goal"]) for route in routes]
    return starts, goals


def compute_crop_bounds(
    occ_grid,
    coords: Sequence[Coord],
    padding_cells: int = 40,
    downsample: int = 1,
) -> Tuple[int, int, int, int]:
    ds = max(1, int(downsample))
    cols = [int(c[0]) for c in coords]
    rows = [int(c[1]) for c in coords]
    pad = int(padding_cells)
    col_min = max(0, min(cols) - pad)
    col_max = min((occ_grid.width - 1) // ds, max(cols) + pad)
    row_min = max(0, min(rows) - pad)
    row_max = min((occ_grid.height - 1) // ds, max(rows) + pad)
    fine_col_min = col_min * ds
    fine_col_max = min(occ_grid.width - 1, col_max * ds + (ds - 1))
    fine_row_min = row_min * ds
    fine_row_max = min(occ_grid.height - 1, row_max * ds + (ds - 1))
    return fine_col_min, fine_col_max, fine_row_min, fine_row_max


def build_static_obstacles(
    occ_grid,
    crop_bounds: Tuple[int, int, int, int] | None = None,
    downsample: int = 1,
    coarse_block_policy: str = DEFAULT_COARSE_BLOCK_POLICY,
) -> List[Coord]:
    model = TraversabilityModel(
        occ_grid, downsample=downsample, coarse_block_policy=coarse_block_policy
    )
    return model.static_obstacle_cells(crop_bounds)


def build_mapf_grid_config(
    occ_grid,
    grid_size: int = 1,
    starts: Sequence[Coord] | None = None,
    goals: Sequence[Coord] | None = None,
    crop: bool = True,
    crop_padding_cells: int = 40,
    downsample: int | None = None,
    coarse_block_policy: str = DEFAULT_COARSE_BLOCK_POLICY,
) -> MapfGridConfig:
    ds = max(1, int(downsample if downsample is not None else default_mapf_downsample(occ_grid)))
    policy = normalize_coarse_block_policy(coarse_block_policy)
    crop_bounds = None
    if crop and starts and goals:
        crop_bounds = compute_crop_bounds(
            occ_grid, list(starts) + list(goals), crop_padding_cells, downsample=ds
        )
    cache_key = (id(occ_grid), ds, policy, crop_bounds)
    cached = _GRID_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model = TraversabilityModel(occ_grid, downsample=ds, coarse_block_policy=policy)
    static_obstacles = model.static_obstacle_cells(crop_bounds)
    config = MapfGridConfig(
        grid_size=int(grid_size),
        robot_radius=mapf_robot_radius_cells(),
        static_obstacles=static_obstacles,
        width_cells=int(occ_grid.width),
        height_cells=int(occ_grid.height),
        crop_bounds=crop_bounds,
        downsample=ds,
        coarse_block_policy=policy,
    )
    _GRID_CONFIG_CACHE[cache_key] = config
    return config


def make_traversability_model(
    occ_grid,
    downsample: int = 1,
    coarse_block_policy: str = DEFAULT_COARSE_BLOCK_POLICY,
) -> TraversabilityModel:
    return TraversabilityModel(
        occ_grid, downsample=downsample, coarse_block_policy=coarse_block_policy
    )


def mapf_path_to_world_polyline(
    occ_grid,
    path: MapfPath,
    downsample: int = 1,
) -> List[Tuple[float, float]]:
    if path is None or len(path) == 0:
        return []
    out: List[Tuple[float, float]] = []
    prev = None
    for point in path:
        cell = (int(point[0]), int(point[1]))
        if cell == prev:
            continue
        out.append(cell_to_world(occ_grid, cell, downsample))
        prev = cell
    return out


def count_vertex_conflicts(
    paths: Sequence[MapfPath],
    robot_radius: int,
    *,
    cell_size_m: float | None = None,
    robot_diameter_m: float = ROBOT_DIAMETER_M,
) -> int:
    if not paths:
        return 0
    lengths = [len(p) for p in paths if p is not None and len(p) > 0]
    if not lengths:
        return 0
    max_t = max(lengths)
    conflicts = 0
    for t in range(max_t):
        positions = []
        for path in paths:
            if path is None or len(path) == 0:
                continue
            idx = min(t, len(path) - 1)
            positions.append(np.asarray(path[idx], dtype=float))
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist_cells = float(np.linalg.norm(positions[i] - positions[j], 2))
                if cell_size_m is not None:
                    if dist_cells * cell_size_m <= robot_diameter_m:
                        conflicts += 1
                elif int(dist_cells) <= 2 * robot_radius:
                    conflicts += 1
    return conflicts


def validate_mapf_solution(
    paths: Sequence[MapfPath],
    robot_radius: int,
    *,
    cell_size_m: float | None = None,
    robot_diameter_m: float = ROBOT_DIAMETER_M,
) -> Tuple[bool, int]:
    conflict_count = count_vertex_conflicts(
        paths,
        robot_radius,
        cell_size_m=cell_size_m,
        robot_diameter_m=robot_diameter_m,
    )
    return conflict_count == 0, conflict_count
