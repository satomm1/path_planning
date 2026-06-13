"""Shared MAPF / MILP solver orchestration for benchmark and visualization."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


def _format_solver_error_status(exc: BaseException) -> str:
    msg = str(exc).strip().replace("\n", " ")
    if msg:
        return f"error:{type(exc).__name__}:{msg}"
    return f"error:{type(exc).__name__}"


def _print_solver_exception(
    solver_label: str,
    exc: BaseException,
    *,
    context: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    print_traceback: bool = True,
) -> None:
    lines = [f"\n--- {solver_label} failed ---"]
    if context:
        ctx = ", ".join(f"{key}={value}" for key, value in context.items())
        lines.append(f"Context: {ctx}")
    if diagnostics:
        diag = ", ".join(f"{key}={value}" for key, value in diagnostics.items())
        lines.append(f"Diagnostics: {diag}")
    lines.append(f"{type(exc).__name__}: {exc}")
    print("\n".join(lines), flush=True)
    if print_traceback:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


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
        print("Starting CBS...", flush=True)
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
        print("Starting PP...", flush=True)
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
    stride: int = 1,
    verbose: bool = True,
) -> dict:
    """Time-optimal coordination on fixed path-bank polylines."""
    motion_cfg = motion if motion is not None else MotionConfig()
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    try:
        from social_path_planning.multi_planning import (
            MultiAgentSimultaneousPlanner,
            subsample_path_by_stride,
        )
    except ImportError as exc:
        stage_paths = [list(route["path"]) for route in routes]
        starts_world = [tuple(route["x_init"]) for route in routes]
        goals_world = [tuple(route["x_goal"]) for route in routes]
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
            "norm": "inf" if norm == np.inf else int(norm),
            "milp_stride": stride,
        }

    stage_paths = [
        subsample_path_by_stride(list(route["path"]), stride) for route in routes
    ]
    starts_world = [tuple(route["x_init"]) for route in routes]
    goals_world = [tuple(route["x_goal"]) for route in routes]

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
        "norm": metrics.get("norm"),
        "max_velocity_mps": float(motion_cfg.max_velocity_mps),
        "milp_stride": stride,
    }


def run_event_milp_solver(
    occ_grid,
    routes: Sequence[dict],
    *,
    norm: int = 1,
    motion: MotionConfig | None = None,
    stride: int = 1,
    verbose: bool = True,
    print_errors: bool = True,
    print_constraints: bool = False,
    print_constraints_on_error: bool = True,
    error_context: Mapping[str, Any] | None = None,
) -> dict:
    """Time-optimal coordination on event interest waypoints (reduced MILP)."""
    motion_cfg = motion if motion is not None else MotionConfig()
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    try:
        from social_path_planning.event_multi_planning import (
            EventMultiAgentSimultaneousPlanner,
            expand_event_times_to_original_path,
        )
        from social_path_planning.multi_planning import subsample_path_by_stride
    except ImportError as exc:
        if print_errors:
            _print_solver_exception(
                "Event MILP import",
                exc,
                context=error_context,
                print_traceback=False,
            )
        stage_paths = [list(route["path"]) for route in routes]
        starts_world = [tuple(route["x_init"]) for route in routes]
        goals_world = [tuple(route["x_goal"]) for route in routes]
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
            "norm": "inf" if norm == np.inf else int(norm),
            "milp_stride": stride,
            "planner_mode": "event",
        }

    stage_paths = [
        subsample_path_by_stride(list(route["path"]), stride) for route in routes
    ]
    starts_world = [tuple(route["x_init"]) for route in routes]
    goals_world = [tuple(route["x_goal"]) for route in routes]

    if verbose:
        print(f"Solving event MILP simultaneous planner (norm={norm})...")
    t0 = time.perf_counter()
    status = "ok"
    agent_times = None
    analysis = None
    planner = None
    solver_runtime_s = 0.0
    solver_exc: BaseException | None = None
    try:
        planner = EventMultiAgentSimultaneousPlanner(occ_grid, paths=stage_paths, norm=norm)
        planner.assign_velocities([motion_cfg.max_velocity_mps] * len(stage_paths))
        agent_times = planner.plan(verbose=False, print_constraints=print_constraints)
        analysis = planner.analysis
        solver_runtime_s = float(getattr(planner, "solver_runtime_s", 0.0))
        time_lists = [
            expand_event_times_to_original_path(agent_path, event_times)
            for agent_path, event_times in zip(analysis.agents, agent_times)
        ]
    except Exception as exc:
        solver_exc = exc
        status = _format_solver_error_status(exc)
        time_lists = [[] for _ in routes]
        if print_errors:
            diagnostics = {
                "norm": norm,
                "num_agents": len(stage_paths),
                "milp_stride": stride,
            }
            if planner is not None:
                diagnostics["binary_z_count"] = getattr(planner, "num_z", None)
                planner_analysis = getattr(planner, "analysis", None)
                if planner_analysis is not None:
                    diagnostics["encounter_count"] = len(planner_analysis.encounters)
                    diagnostics["interest_waypoint_count"] = sum(
                        len(agent.interest_waypoints) for agent in planner_analysis.agents
                    )
            _print_solver_exception(
                "Event MILP",
                exc,
                context=error_context,
                diagnostics=diagnostics,
            )
            if print_constraints_on_error and not print_constraints and planner is not None:
                try:
                    planner.build_problem(print_constraints=True)
                except Exception as print_exc:
                    print(
                        f"Could not print event MILP constraints after failure: "
                        f"{type(print_exc).__name__}: {print_exc}",
                        flush=True,
                    )
    runtime_s = float(time.perf_counter() - t0)

    interest_waypoint_count = (
        sum(len(agent.interest_waypoints) for agent in analysis.agents)
        if analysis is not None
        else 0
    )
    encounter_count = len(analysis.encounters) if analysis is not None else 0
    binary_z_count = int(planner.num_z) if planner is not None else 0

    metrics = aggregate_milp_metrics(
        time_lists,
        runtime_s,
        status,
        norm=norm,
        world_paths=stage_paths,
        motion=motion_cfg,
        solver_runtime_s=solver_runtime_s,
    )
    metrics["interest_waypoint_count"] = int(interest_waypoint_count)
    metrics["encounter_count"] = int(encounter_count)
    metrics["binary_z_count"] = binary_z_count
    if solver_exc is not None:
        metrics["error_type"] = type(solver_exc).__name__
        metrics["error_message"] = str(solver_exc)
    if verbose:
        print(
            f"Event MILP done: status={metrics.get('status')} success={metrics.get('success')} "
            f"encounters={encounter_count} z={binary_z_count} "
            f"(solver={solver_runtime_s:.2f}s total={runtime_s:.2f}s)"
        )
    elif print_errors and solver_exc is not None:
        print(
            f"Event MILP trial recorded as failure: status={status} "
            f"encounters={encounter_count} z={binary_z_count} "
            f"(solver={solver_runtime_s:.2f}s total={runtime_s:.2f}s)",
            flush=True,
        )

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
        "norm": metrics.get("norm"),
        "max_velocity_mps": float(motion_cfg.max_velocity_mps),
        "milp_stride": stride,
        "planner_mode": "event",
        "interest_waypoint_count": int(interest_waypoint_count),
        "encounter_count": int(encounter_count),
        "binary_z_count": binary_z_count,
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
