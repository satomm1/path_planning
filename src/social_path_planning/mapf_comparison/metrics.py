"""Metrics for MAPF and MILP coordination benchmarks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from social_path_planning.mapf_adapter import (
    MapfPath,
    cell_to_world,
    path_length_meters,
    validate_mapf_solution,
    validation_robot_radius_cells,
)
from social_path_planning.mapf_comparison.grid_traversability import (
    DEFAULT_COARSE_BLOCK_POLICY,
    TraversabilityModel,
)
from social_path_planning.mapf_comparison.motion import MotionConfig, cell_size_m
from social_path_planning.mapf_comparison.solution_metrics import compute_solution_metrics


def mapf_path_timesteps(path: MapfPath) -> int:
    """Cost in grid timesteps (wait actions count as steps in STA*)."""
    if path is None or len(path) == 0:
        return 0
    return int(len(path) - 1)


def _minimal_occ_grid(resolution: float):
    """Tiny grid for unit tests when no occupancy grid is passed."""

    class _Grid:
        pass

    g = _Grid()
    g.resolution = float(resolution)
    g.origin_x = 0.0
    g.origin_y = 0.0
    g.width = 10000
    g.height = 10000
    return g


def aggregate_mapf_metrics(
    paths: Sequence[MapfPath],
    robot_radius: int,
    resolution: float,
    runtime_s: float,
    status: str,
    validation_radius: int | None = None,
    downsample: int = 1,
    occ_grid=None,
    motion: MotionConfig | None = None,
    coarse_block_policy: str = DEFAULT_COARSE_BLOCK_POLICY,
) -> Dict[str, Any]:
    valid_paths = paths is not None and all(p is not None and len(p) > 0 for p in paths)
    success = status == "ok" and valid_paths
    ds = max(1, int(downsample))
    check_radius = validation_radius if validation_radius is not None else robot_radius
    if validation_radius is None and resolution > 0:
        coarse_res = float(resolution) * ds
        check_radius = validation_robot_radius_cells(coarse_res)
    cell_sz = cell_size_m(resolution, ds) if resolution > 0 else None
    is_valid, conflict_count = (False, -1)
    static_collision_count = 0
    static_valid = True
    if valid_paths:
        is_valid, conflict_count = validate_mapf_solution(
            paths,
            check_radius,
            cell_size_m=cell_sz,
        )
        if occ_grid is not None and resolution > 0:
            trav = TraversabilityModel(
                occ_grid, downsample=ds, coarse_block_policy=coarse_block_policy
            )
            for path in paths:
                static_collision_count += trav.count_mapf_path_static_collisions(path)
            static_valid = static_collision_count == 0
        success = success and is_valid and static_valid

    per_agent_steps = [mapf_path_timesteps(p) for p in paths] if valid_paths else []
    soc_steps = int(sum(per_agent_steps)) if per_agent_steps else None
    makespan_steps = int(max(per_agent_steps)) if per_agent_steps else None

    motion_cfg = motion if motion is not None else MotionConfig()
    grid_for_metrics = occ_grid if occ_grid is not None else _minimal_occ_grid(resolution)

    soc_seconds = float(soc_steps) if soc_steps is not None else None
    makespan_seconds = float(makespan_steps) if makespan_steps is not None else None
    total_path_length_m = None
    per_agent_path_length_m: List[float] = []
    per_agent_completion_s = (
        [float(t) for t in per_agent_steps] if per_agent_steps else []
    )

    if valid_paths and resolution > 0:
        world_paths = [
            [cell_to_world(grid_for_metrics, (int(p[0]), int(p[1])), ds) for p in path]
            for path in paths
        ]
        per_agent_path_length_m = [path_length_meters(wp) for wp in world_paths]
        total_path_length_m = float(sum(per_agent_path_length_m))

    return {
        "success": bool(success),
        "status": status,
        "conflict_count": int(conflict_count),
        "static_collision_count": int(static_collision_count),
        "static_valid": bool(static_valid),
        "dynamic_valid": bool(is_valid) if valid_paths else False,
        "soc_timesteps": soc_steps,
        "makespan_timesteps": makespan_steps,
        "soc_seconds": soc_seconds,
        "makespan_seconds": makespan_seconds,
        "total_path_length_m": total_path_length_m,
        "per_agent_path_length_m": per_agent_path_length_m,
        "per_agent_completion_s": per_agent_completion_s,
        "per_agent_timesteps": per_agent_steps,
        "solver_runtime_s": float(runtime_s),
        "runtime_s": float(runtime_s),
        "max_velocity_mps": float(motion_cfg.max_velocity_mps),
        "cell_size_m": float(cell_sz) if cell_sz is not None else None,
        "downsample": int(ds),
    }


def aggregate_milp_metrics(
    agent_times: Optional[List[List[float]]],
    runtime_s: float,
    status: str,
    norm: int = 1,
    world_paths: Sequence[Sequence] | None = None,
    motion: MotionConfig | None = None,
    *,
    solver_runtime_s: float | None = None,
) -> Dict[str, Any]:
    success = status == "ok" and agent_times is not None and len(agent_times) > 0
    finals = []
    if success:
        for times in agent_times:
            if times is None or len(times) == 0:
                success = False
                break
            finals.append(float(times[-1]))

    soc = float(sum(finals)) if success else None
    makespan = float(max(finals)) if success else None

    motion_cfg = motion if motion is not None else MotionConfig()
    total_path_length_m = None
    per_agent_path_length_m: List[float] = []

    if success and world_paths is not None:
        time_lists = [list(t) for t in agent_times]
        sol = compute_solution_metrics(world_paths, time_lists)
        soc = sol["soc_seconds"]
        makespan = sol["makespan_seconds"]
        total_path_length_m = sol["total_path_length_m"]
        per_agent_path_length_m = sol["per_agent_path_length_m"]

    norm_value: int | str
    if norm == np.inf or norm == float("inf"):
        norm_value = "inf"
    else:
        norm_value = int(norm)

    return {
        "success": bool(success),
        "status": status,
        "norm": norm_value,
        "soc_seconds": soc,
        "makespan_seconds": makespan,
        "per_agent_completion_s": finals if success else [],
        "total_path_length_m": total_path_length_m,
        "per_agent_path_length_m": per_agent_path_length_m,
        "solver_runtime_s": float(
            runtime_s if solver_runtime_s is None else solver_runtime_s
        ),
        "runtime_s": float(runtime_s),
        "max_velocity_mps": float(motion_cfg.max_velocity_mps),
    }


def path_bank_soc_meters(routes: Sequence[dict]) -> List[float]:
    """Solo A* path lengths from the path bank (for priority ordering context)."""
    return [path_length_meters(r.get("path") or []) for r in routes]
