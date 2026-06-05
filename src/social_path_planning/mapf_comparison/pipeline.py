"""Shared MAPF / MILP solver orchestration for benchmark and visualization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from social_path_planning.mapf_adapter import (
    MapfGridConfig,
    MapfPath,
    build_mapf_grid_config,
    cell_to_world,
    default_mapf_downsample,
    mapf_path_to_world_polyline,
    path_length_meters,
    resolve_mapf_search_budget,
    starts_goals_from_routes,
    validation_robot_radius_cells,
)
from social_path_planning.mapf_baselines.cbs_runner import run_cbs
from social_path_planning.mapf_baselines.prioritized_planning import (
    priority_order_by_path_length,
    run_prioritized_planning,
)
from social_path_planning.mapf_comparison.grid_traversability import (
    DEFAULT_COARSE_BLOCK_POLICY,
    ROBOT_DIAMETER_M,
)
from social_path_planning.mapf_comparison.metrics import (
    aggregate_mapf_metrics,
    aggregate_milp_metrics,
)
from social_path_planning.mapf_comparison.motion import MotionConfig
from social_path_planning.mapf_comparison.schedule import schedule_mapf_paths

Coord = Tuple[int, int]


@dataclass
class MapfRunConfig:
    downsample: int | None = None
    crop_padding_cells: int = 40
    coarse_block_policy: str = DEFAULT_COARSE_BLOCK_POLICY
    pp_low_level_max_iter: int = 0
    cbs_low_level_max_iter: int = 0
    cbs_max_iter: int = 0
    cbs_max_process: int = 1
    motion: MotionConfig = field(default_factory=MotionConfig)
    run_cbs: bool = True
    run_pp: bool = True
    crop: bool = True


def resolve_downsample(occ_grid, cfg: MapfRunConfig, demo_open: bool = False) -> int:
    if demo_open:
        return 1
    if cfg.downsample is not None:
        return max(1, int(cfg.downsample))
    return default_mapf_downsample(occ_grid)


def prepare_mapf_problem(
    occ_grid,
    routes: Sequence[dict],
    cfg: MapfRunConfig,
    *,
    demo_open: bool = False,
) -> Tuple[MapfGridConfig, List[Coord], List[Coord], dict]:
    ds = resolve_downsample(occ_grid, cfg, demo_open=demo_open)
    starts, goals = starts_goals_from_routes(routes, occ_grid, downsample=ds)
    budget = resolve_mapf_search_budget(
        starts,
        goals,
        num_agents=len(routes),
        downsample=ds,
        crop_padding_cells=cfg.crop_padding_cells,
        pp_low_level_max_iter=cfg.pp_low_level_max_iter,
        cbs_low_level_max_iter=cfg.cbs_low_level_max_iter,
        cbs_max_iter=cfg.cbs_max_iter,
    )
    grid_config = build_mapf_grid_config(
        occ_grid,
        starts=starts,
        goals=goals,
        crop=cfg.crop and not demo_open,
        crop_padding_cells=budget["crop_padding_cells"],
        downsample=ds,
        coarse_block_policy=cfg.coarse_block_policy,
    )
    return grid_config, starts, goals, budget


def _build_solver_entry(
    occ_grid,
    grid_config: MapfGridConfig,
    grid_paths: Sequence[MapfPath] | None,
    metrics: dict,
    motion: MotionConfig,
    *,
    starts_world: Sequence[Tuple[float, float]],
    goals_world: Sequence[Tuple[float, float]],
    extras: dict | None = None,
) -> dict:
    ds = grid_config.downsample
    wp, tl = (
        schedule_mapf_paths(occ_grid, grid_paths or [], ds, motion)
        if grid_paths
        else ([], [])
    )
    polylines = (
        [mapf_path_to_world_polyline(occ_grid, p, ds) for p in grid_paths]
        if grid_paths
        else []
    )
    entry = {
        "grid_paths": list(grid_paths) if grid_paths else None,
        "world_paths": wp,
        "time_lists": tl,
        "world_polylines": polylines,
        "metrics": metrics,
        "starts_world": list(starts_world),
        "goals_world": list(goals_world),
        "robot_radius_cells": grid_config.robot_radius,
        "conflict_threshold_m": float(ROBOT_DIAMETER_M),
        "success": bool(metrics.get("success")),
    }
    if extras:
        entry.update(extras)
    return entry


def run_mapf_solvers(
    occ_grid,
    routes: Sequence[dict],
    grid_config: MapfGridConfig,
    starts: Sequence[Coord],
    goals: Sequence[Coord],
    budget: dict,
    cfg: MapfRunConfig,
    *,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run CBS and/or PP; return dict keyed by method name."""
    motion = cfg.motion
    ds = grid_config.downsample
    starts_world = [cell_to_world(occ_grid, s, ds) for s in starts]
    goals_world = [cell_to_world(occ_grid, g, ds) for g in goals]
    results: Dict[str, dict] = {}
    map_resolution = float(occ_grid.resolution)

    if cfg.run_cbs:
        if verbose:
            print("\nSolving CBS...")
        t0 = time.perf_counter()
        cbs_paths, cbs_status = run_cbs(
            grid_config,
            list(starts),
            list(goals),
            max_iter=budget["cbs_max_iter"],
            low_level_max_iter=budget["cbs_low_level_max_iter"],
            max_process=cfg.cbs_max_process,
        )
        cbs_runtime = float(time.perf_counter() - t0)
        cbs_metrics = aggregate_mapf_metrics(
            cbs_paths or [],
            grid_config.robot_radius,
            map_resolution,
            cbs_runtime,
            cbs_status,
            downsample=ds,
            occ_grid=occ_grid,
            motion=motion,
            coarse_block_policy=grid_config.coarse_block_policy,
        )
        if verbose:
            print(
                f"CBS done: status={cbs_status} success={cbs_metrics['success']} "
                f"conflicts={cbs_metrics.get('conflict_count')} "
                f"static={cbs_metrics.get('static_valid')} ({cbs_runtime:.2f}s)"
            )
            if cbs_status == "no_solution":
                print(
                    "  Hint: try --pp-low-level-max-iter 0 (auto) or "
                    "raise --cbs-max-iter / --crop-padding"
                )
        results["cbs"] = _build_solver_entry(
            occ_grid,
            grid_config,
            cbs_paths,
            cbs_metrics,
            motion,
            starts_world=starts_world,
            goals_world=goals_world,
        )

    if cfg.run_pp:
        if verbose:
            print("\nSolving PP (prioritized planning)...")
        t0 = time.perf_counter()
        pp_paths, pp_order, pp_status = run_prioritized_planning(
            grid_config,
            list(starts),
            list(goals),
            routes,
            low_level_max_iter=budget["pp_low_level_max_iter"],
        )
        pp_runtime = float(time.perf_counter() - t0)
        pp_metrics = aggregate_mapf_metrics(
            pp_paths or [],
            grid_config.robot_radius,
            map_resolution,
            pp_runtime,
            pp_status,
            downsample=ds,
            occ_grid=occ_grid,
            motion=motion,
            coarse_block_policy=grid_config.coarse_block_policy,
        )
        if verbose:
            print(
                f"PP done: status={pp_status} success={pp_metrics['success']} "
                f"conflicts={pp_metrics.get('conflict_count')} "
                f"static={pp_metrics.get('static_valid')} "
                f"priority_order={pp_order} ({pp_runtime:.2f}s)"
            )
            if pp_status == "no_solution":
                print(
                    "  Hint: try --pp-low-level-max-iter 0 (auto) or increase --crop-padding"
                )
        results["pp"] = _build_solver_entry(
            occ_grid,
            grid_config,
            pp_paths,
            pp_metrics,
            motion,
            starts_world=starts_world,
            goals_world=goals_world,
            extras={
                "priority_order": pp_order,
                "priority_lengths": [path_length_meters(r.get("path") or []) for r in routes],
            },
        )

    return results


def run_milp_solver(
    occ_grid,
    routes: Sequence[dict],
    *,
    norm: int = 1,
    motion: MotionConfig | None = None,
    verbose: bool = True,
) -> dict:
    """Time-optimal coordination on fixed path-bank polylines."""
    motion_cfg = motion if motion is not None else MotionConfig()
    stage_paths = [list(route["path"]) for route in routes]
    starts_world = [tuple(route["x_init"]) for route in routes]
    goals_world = [tuple(route["x_goal"]) for route in routes]

    try:
        from social_path_planning.multi_planning import MultiAgentSimultaneousPlanner
    except ImportError as exc:
        return {
            "grid_paths": None,
            "world_paths": stage_paths,
            "time_lists": [[] for _ in routes],
            "world_polylines": stage_paths,
            "metrics": aggregate_milp_metrics(None, 0.0, f"import_error:{exc}"),
            "starts_world": starts_world,
            "goals_world": goals_world,
            "robot_radius_cells": validation_robot_radius_cells(occ_grid.resolution),
            "conflict_threshold_m": float(ROBOT_DIAMETER_M),
            "success": False,
            "norm": int(norm),
        }

    if verbose:
        print(f"Solving MILP simultaneous planner (norm={norm})...")
    t0 = time.perf_counter()
    status = "ok"
    agent_times = None
    try:
        planner = MultiAgentSimultaneousPlanner(occ_grid, paths=stage_paths, norm=norm)
        planner.assign_velocities([motion_cfg.max_velocity_mps] * len(stage_paths))
        agent_times = planner.plan(verbose=False)
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
    runtime_s = float(time.perf_counter() - t0)
    metrics = aggregate_milp_metrics(
        agent_times,
        runtime_s,
        status,
        norm=norm,
        world_paths=stage_paths,
        motion=motion_cfg,
    )
    if verbose:
        print(
            f"MILP done: status={metrics.get('status')} success={metrics.get('success')} "
            f"({runtime_s:.2f}s)"
        )

    time_lists = [list(t) for t in agent_times] if agent_times else [[] for _ in routes]
    return {
        "grid_paths": None,
        "world_paths": stage_paths,
        "time_lists": time_lists,
        "world_polylines": stage_paths,
        "metrics": metrics,
        "starts_world": starts_world,
        "goals_world": goals_world,
        "robot_radius_cells": validation_robot_radius_cells(occ_grid.resolution),
        "conflict_threshold_m": float(ROBOT_DIAMETER_M),
        "success": bool(metrics["success"]),
        "norm": int(norm),
        "max_velocity_mps": float(motion_cfg.max_velocity_mps),
    }


def print_grid_summary(
    occ_grid,
    grid_config: MapfGridConfig,
    budget: dict,
    routes: Sequence[dict],
    motion: MotionConfig,
) -> None:
    print(
        f"MAPF grid: {occ_grid.width}x{occ_grid.height} cells, "
        f"obstacles={len(grid_config.static_obstacles)}, "
        f"downsample={grid_config.downsample}, "
        f"coarse_block_policy={grid_config.coarse_block_policy}, "
        f"robot_radius={grid_config.robot_radius}, "
        f"crop={grid_config.crop_bounds}, "
        f"crop_padding_cells={budget['crop_padding_cells']}, "
        f"cbs_max_iter={budget['cbs_max_iter']}, "
        f"cbs_low_level_max_iter={budget['cbs_low_level_max_iter']}, "
        f"pp_low_level_max_iter={budget['pp_low_level_max_iter']}, "
        f"max_velocity_mps={motion.max_velocity_mps}, "
        f"stastar_connectivity=8_octile"
    )
    print(f"Priority order (longer path first): {priority_order_by_path_length(routes)}")
