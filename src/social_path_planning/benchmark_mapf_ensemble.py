"""
Multi-trial MAPF ensemble benchmark: pooled routes, random agent subsets, averaged metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.mapf_comparison import MapfRunConfig, MotionConfig
from social_path_planning.mapf_comparison.grid_traversability import ROBOT_DIAMETER_M
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.utils import snap_to_grid

_MODULE_DIR = Path(__file__).resolve().parent
from social_path_planning.mapf_comparison.checkpoint import (
    append_trial_rows,
    compact_trial_rows,
    load_completed_keys,
    load_methods_to_retry,
    load_trial_rows,
    write_csv as checkpoint_write_csv,
)
from social_path_planning.mapf_comparison.ensemble import (
    aggregate_ensemble_metrics,
    ensure_modified_path_pool,
    load_cached_milp_routes,
    print_ensemble_summary,
    run_ensemble_trial,
    sample_route_subset,
)


def _wall_distance_cache_path(scenario_name: str) -> Path:
    return _MODULE_DIR / "environments" / f"{scenario_name}_wall_dist.npz"


def build_occ_grid(scenario_name: str):
    """
    Build the occupancy grid for ensemble runs.

    Uses a precomputed ``{scenario}_wall_dist.npz`` when present; does not build one
    if the file is missing (modified A* falls back to raycast wall queries).
    """
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    map_dim = [round(map_size[i] / map_resolution) for i in range(2)]
    cache_path = _wall_distance_cache_path(scenario_name)
    wall_cache = cache_path if cache_path.is_file() else None
    if wall_cache is None:
        print(
            f"No wall-distance cache at {cache_path}; "
            "modified A* will use runtime raycast (slower)."
        )
    occ_grid = StochOccupancyGrid2D(
        map_resolution,
        map_dim[0],
        map_dim[1],
        0,
        0,
        10,
        occ.T,
        robot_d=ROBOT_DIAMETER_M,
        wall_distance_cache_path=wall_cache,
        auto_build_wall_distance_cache=False,
    )
    statespace_hi = snap_to_grid(map_size, map_resolution)
    return occ_grid, map_size, map_resolution, statespace_hi


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _trial_fieldnames():
    return [
        "trial_id",
        "trial_seed",
        "sampled_route_ids",
        "num_agents",
        "method",
        "success",
        "status",
        "solver_runtime_s",
        "runtime_s",
        "soc_seconds",
        "makespan_seconds",
        "soc_timesteps",
        "makespan_timesteps",
        "total_path_length_m",
        "conflict_count",
        "priority_order",
        "interest_waypoint_count",
        "encounter_count",
        "binary_z_count",
        "error_type",
        "error_message",
    ]


def _summary_fieldnames():
    return [
        "num_agents",
        "method",
        "num_trials",
        "success_count",
        "failure_count",
        "timeout_count",
        "success_rate",
        "avg_solver_runtime_s",
        "std_solver_runtime_s",
        "median_solver_runtime_s",
        "avg_makespan_seconds",
        "std_makespan_seconds",
        "avg_soc_seconds",
        "std_soc_seconds",
        "avg_makespan_timesteps",
        "std_makespan_timesteps",
        "avg_soc_timesteps",
        "std_soc_timesteps",
        "avg_total_path_length_m",
        "std_total_path_length_m",
    ]


def _event_viz_output_dir(output_prefix: str, trial_id: int) -> Path:
    return (
        Path(output_prefix).parent
        / "event_planning_viz"
        / Path(output_prefix).name
        / f"trial{trial_id:04d}"
    )


def _viz_event_waypoints_for_trial(
    *,
    trial_id: int,
    trial_seed: int,
    occ_grid,
    pool,
    num_agents: int,
    path_bank_path: Path,
    milp_astar_mode: str,
    milp_stride: int,
    output_prefix: str,
    social_graph=None,
    heatmap_prefix=None,
    heatmap_file=None,
    sparse_graph_threshold: float | None = None,
    sparse_min_component_size: int | None = None,
):
    """Analyze and plot event interest waypoints for one trial's sampled MILP paths."""
    from social_path_planning.event_multi_planning import analyze_event_paths, viz_event_waypoints
    from social_path_planning.multi_planning import subsample_path_by_stride

    rng = np.random.default_rng(int(trial_seed))
    stage_records, route_ids = sample_route_subset(pool, num_agents, rng)
    milp_routes = load_cached_milp_routes(
        stage_records,
        milp_astar_mode,
        path_bank_path,
        social_graph=social_graph,
        heatmap_prefix=heatmap_prefix,
        heatmap_file=heatmap_file,
        sparse_graph_threshold=sparse_graph_threshold,
        sparse_min_component_size=sparse_min_component_size,
    )
    paths = [
        subsample_path_by_stride(list(route["path"]), milp_stride) for route in milp_routes
    ]
    analysis = analyze_event_paths(paths)
    out_dir = _event_viz_output_dir(output_prefix, trial_id)
    viz_event_waypoints(
        analysis,
        occ_grid=occ_grid,
        output_path=out_dir / "waypoints.png",
    )
    print(
        f"Event waypoint viz: trial {trial_id} (seed={trial_seed}) "
        f"route_ids={route_ids} -> {out_dir.resolve()}"
    )


def _write_manifest(
    manifest_json: Path,
    *,
    run_id: str,
    config: dict,
    pool_meta: dict,
    path_bank_path: Path,
    pool_astar_build_s: float,
    trial_loop_s: float,
    summary_rows: list,
    trial_rows_count: int,
) -> None:
    try:
        import cbs_mapf

        cbs_version = getattr(cbs_mapf, "__version__", "unknown")
    except ImportError:
        cbs_version = None

    manifest = {
        "run_id": run_id,
        "config": config,
        "path_bank": pool_meta,
        "path_bank_sha256": _file_sha256(path_bank_path) if path_bank_path.exists() else None,
        "pool_astar_build_s": pool_astar_build_s,
        "trial_loop_s": trial_loop_s,
        "trial_rows_count": int(trial_rows_count),
        "summary": summary_rows,
        "libraries": {"cbs_mapf": cbs_version},
        "comparison_notes": {
            "cbs_pp": "Discrete space-time MAPF (library-default STA*, unit cost per timestep).",
            "event_milp_soc": "Continuous-time coordination with max-velocity kinematic constraints.",
            "cross_method_cost": (
                "Use soc_timesteps/makespan_timesteps for CBS/PP and "
                "soc_seconds/makespan_seconds for event MILP."
            ),
        },
        "timing_notes": {
            "pool_astar_build_s": (
                "One-time modified/vanilla A* path generation during pool prep; "
                "excluded from event MILP trial averages."
            ),
            "avg_solver_runtime_s": (
                "Mean over successful trials only. "
                "Event MILP: cvxpy solve only (excludes constraint build). "
                "CBS/PP: full planner wall time."
            ),
            "runtime_s": (
                "Per-trial total wall time. Event MILP includes analysis, "
                "constraint construction, solve, and time expansion."
            ),
        },
        "performance_notes": {
            "mapf_grid_cache": "static_obstacles cached per (occ_grid, downsample, crop_bounds, policy)",
            "cbs_max_iter_ceiling": "Scales with num_agents (base ceiling 5000, up to 20000)",
        },
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_mapf_ensemble(
    scenario_name: str,
    pool_size: int,
    num_agents: int,
    num_trials: int,
    benchmark_seed: int,
    trial_seed: int,
    max_attempts: int,
    output_prefix: str,
    path_bank_path=None,
    cbs_max_iter: int = 0,
    cbs_low_level_max_iter: int = 500,
    cbs_max_process: int = 1,
    pp_low_level_max_iter: int = 0,
    max_velocity_mps: float | None = None,
    mapf_downsample: int | None = None,
    crop_padding_cells: int = 40,
    coarse_block_policy: str = "fine_center",
    milp_astar_mode: str = "modified",
    heatmap_prefix: str | None = None,
    heatmap_file: str | None = None,
    sparse_graph_threshold: float = 2.0,
    sparse_min_component_size: int = 15,
    refresh_social_bank: bool = False,
    pregen_only: bool = False,
    milp_stride: int = 1,
    viz_event_waypoints: bool = False,
    viz_event_every: int = 1,
    print_event_milp_constraints: bool = False,
    print_event_milp_constraints_every: int = 1,
    print_event_milp_constraints_on_error: bool = True,
    num_agents_list: list[int] | None = None,
    resume: bool = False,
    resume_partial: bool = False,
    retry_failures: bool = False,
    reaggregate_only: bool = False,
    fail_soft: bool = True,
    cbs_timeout_s: float = 0.0,
    pp_timeout_s: float = 0.0,
):
    agents_list = sorted(set(num_agents_list or [num_agents]))
    if not agents_list:
        raise ValueError("num_agents_list must contain at least one value")
    max_agents = max(agents_list)
    if pool_size < max_agents:
        raise ValueError(
            f"pool_size ({pool_size}) must be >= max num_agents ({max_agents})"
        )
    if num_trials < 1 and not pregen_only and not reaggregate_only:
        raise ValueError("num_trials must be >= 1")

    output_prefix_path = Path(output_prefix)
    trials_csv = Path(f"{output_prefix}_trials.csv")
    summary_csv = Path(f"{output_prefix}_summary.csv")
    manifest_json = Path(f"{output_prefix}_manifest.json")

    if reaggregate_only:
        trial_rows = compact_trial_rows(trials_csv, _trial_fieldnames())
        if not trial_rows:
            raise FileNotFoundError(f"No trial rows found at {trials_csv}")
        summary_rows = aggregate_ensemble_metrics(trial_rows)
        checkpoint_write_csv(summary_csv, summary_rows, _summary_fieldnames())
        print(f"Reaggregated {len(trial_rows)} trial rows -> {summary_csv}")
        print_ensemble_summary(summary_rows, pool_astar_build_s=0.0)
        return trial_rows, summary_rows

    motion = (
        MotionConfig(max_velocity_mps=max_velocity_mps)
        if max_velocity_mps is not None
        else MotionConfig()
    )

    wall_cache_path = _wall_distance_cache_path(scenario_name)
    occ_grid, _, map_resolution, statespace_hi = build_occ_grid(scenario_name)
    if path_bank_path is None:
        path_bank_path = output_prefix_path.parent / f"{output_prefix_path.name}_path_pool.json"
    path_bank_path = Path(path_bank_path)

    social_graph = None
    if milp_astar_mode == "modified" and (heatmap_prefix or heatmap_file):
        from social_path_planning.compare_astar import build_sparse_graph_from_heatmap

        social_graph = build_sparse_graph_from_heatmap(
            occ_grid,
            heatmap_prefix=heatmap_prefix,
            heatmap_path=heatmap_file,
            sparse_graph_threshold=sparse_graph_threshold,
            sparse_min_component_size=sparse_min_component_size,
        )
        print(
            f"MILP geometry: modified A* on heatmap sparse graph "
            f"(threshold={sparse_graph_threshold}, "
            f"min_component_size={sparse_min_component_size})"
        )
    elif milp_astar_mode == "modified":
        print("MILP geometry: modified A* (rightness penalty)")
    else:
        print(f"MILP geometry: path-bank polylines ({milp_astar_mode})")

    print(f"Building route pool (size={pool_size}, seed={benchmark_seed})...")
    pool, pool_meta = ensure_modified_path_pool(
        occ_grid,
        statespace_hi,
        map_resolution,
        scenario_name,
        pool_size,
        benchmark_seed,
        max_attempts,
        path_bank_path,
        milp_astar_mode=milp_astar_mode,
        social_graph=social_graph,
        heatmap_prefix=heatmap_prefix,
        heatmap_file=heatmap_file,
        sparse_graph_threshold=sparse_graph_threshold,
        sparse_min_component_size=sparse_min_component_size,
        refresh=refresh_social_bank,
    )
    pool_astar_build_s = float(pool_meta.get("pool_astar_build_s", 0.0))
    print(f"Pool ready: {len(pool)} routes, A* build time {pool_astar_build_s:.2f}s")

    if pregen_only:
        print(f"Pregen-only: cached pool at {path_bank_path.resolve()}")
        return [], []

    mapf_cfg = MapfRunConfig(
        downsample=mapf_downsample,
        crop_padding_cells=crop_padding_cells,
        coarse_block_policy=coarse_block_policy,
        pp_low_level_max_iter=pp_low_level_max_iter,
        cbs_low_level_max_iter=cbs_low_level_max_iter,
        cbs_max_iter=cbs_max_iter,
        cbs_max_process=cbs_max_process,
        motion=motion,
        run_cbs=True,
        run_pp=True,
        cbs_timeout_s=float(cbs_timeout_s),
        pp_timeout_s=float(pp_timeout_s),
    )

    agents_tag = "-".join(str(n) for n in agents_list)
    run_id = (
        f"ensemble_{scenario_name}_pool{pool_size}_agents{agents_tag}_"
        f"trials{num_trials}_seed{benchmark_seed}_trial{trial_seed}"
    )

    retry_map: dict[tuple[int, int], set[str]] = {}
    completed: set[tuple[int, int]] = set()
    if retry_failures:
        if not trials_csv.is_file():
            raise FileNotFoundError(
                f"--retry-failures requires an existing trials CSV at {trials_csv}"
            )
        retry_map = load_methods_to_retry(trials_csv)
        print(f"Retry failures: {len(retry_map)} trial(s) have methods to rerun")
    elif resume:
        completed = load_completed_keys(
            trials_csv,
            require_all_methods=not resume_partial,
            require_success=not resume_partial,
        )
        if completed:
            print(f"Resume: skipping {len(completed)} successful trial(s)")

    new_rows: list[dict] = []
    t_loop_start = time.perf_counter()

    for num_agents in agents_list:
        print(f"\n=== Agent count N={num_agents} ===")
        for trial_id in range(num_trials):
            key = (int(num_agents), int(trial_id))
            methods_to_run = None
            if retry_failures:
                methods_to_run = retry_map.get(key)
                if not methods_to_run:
                    continue
            elif resume and key in completed:
                continue
            this_trial_seed = int(trial_seed + trial_id)
            if (trial_id + 1) % 10 == 0 or trial_id == 0:
                print(
                    f"N={num_agents} trial {trial_id + 1}/{num_trials} "
                    f"(seed={this_trial_seed})"
                )
            if viz_event_waypoints and trial_id % int(viz_event_every) == 0:
                _viz_event_waypoints_for_trial(
                    trial_id=trial_id,
                    trial_seed=this_trial_seed,
                    occ_grid=occ_grid,
                    pool=pool,
                    num_agents=num_agents,
                    path_bank_path=path_bank_path,
                    milp_astar_mode=milp_astar_mode,
                    milp_stride=milp_stride,
                    output_prefix=output_prefix,
                    social_graph=social_graph,
                    heatmap_prefix=heatmap_prefix,
                    heatmap_file=heatmap_file,
                    sparse_graph_threshold=sparse_graph_threshold,
                    sparse_min_component_size=sparse_min_component_size,
                )
            print_constraints_this_trial = (
                print_event_milp_constraints
                and trial_id % int(print_event_milp_constraints_every) == 0
            )
            rows = run_ensemble_trial(
                trial_id,
                this_trial_seed,
                pool,
                num_agents,
                occ_grid,
                mapf_cfg,
                motion,
                path_bank_path,
                milp_astar_mode=milp_astar_mode,
                milp_stride=milp_stride,
                print_event_milp_constraints=print_constraints_this_trial,
                print_event_milp_constraints_on_error=print_event_milp_constraints_on_error,
                fail_soft=fail_soft,
                sparse_graph_threshold=sparse_graph_threshold,
                sparse_min_component_size=sparse_min_component_size,
                methods_to_run=methods_to_run,
                social_graph=social_graph,
                heatmap_prefix=heatmap_prefix,
                heatmap_file=heatmap_file,
            )
            new_rows.extend(rows)
            append_trial_rows(trials_csv, rows, _trial_fieldnames())

    trial_loop_s = float(time.perf_counter() - t_loop_start)
    trial_rows = compact_trial_rows(
        trials_csv,
        _trial_fieldnames(),
        extra_rows=new_rows if new_rows else None,
    )

    summary_rows = aggregate_ensemble_metrics(trial_rows)
    checkpoint_write_csv(summary_csv, summary_rows, _summary_fieldnames())

    config = {
        "scenario_name": scenario_name,
        "pool_size": int(pool_size),
        "num_agents_list": agents_list,
        "num_trials": int(num_trials),
        "benchmark_seed": int(benchmark_seed),
        "trial_seed": int(trial_seed),
        "max_attempts": int(max_attempts),
        "map_resolution": float(map_resolution),
        "motion": motion.to_manifest_dict(),
        "milp_astar_mode": milp_astar_mode,
        "milp_stride": int(milp_stride),
        "heatmap_prefix": heatmap_prefix,
        "heatmap_file": heatmap_file,
        "sparse_graph_threshold": float(sparse_graph_threshold),
        "sparse_min_component_size": int(sparse_min_component_size),
        "mapf_downsample": mapf_downsample,
        "crop_padding_cells": int(crop_padding_cells),
        "coarse_block_policy": coarse_block_policy,
        "cbs_max_iter": int(cbs_max_iter),
        "cbs_low_level_max_iter": int(cbs_low_level_max_iter),
        "pp_low_level_max_iter": int(pp_low_level_max_iter),
        "cbs_max_process": int(cbs_max_process),
        "cbs_timeout_s": float(cbs_timeout_s),
        "pp_timeout_s": float(pp_timeout_s),
        "fail_soft": bool(fail_soft),
        "resume": bool(resume),
        "resume_partial": bool(resume_partial),
        "retry_failures": bool(retry_failures),
        "wall_distance_cache_path": str(wall_cache_path.resolve()),
        "wall_distance_cache_used": bool(wall_cache_path.is_file()),
    }
    _write_manifest(
        manifest_json,
        run_id=run_id,
        config=config,
        pool_meta=pool_meta,
        path_bank_path=path_bank_path,
        pool_astar_build_s=pool_astar_build_s,
        trial_loop_s=trial_loop_s,
        summary_rows=summary_rows,
        trial_rows_count=len(trial_rows),
    )

    print(f"\nSaved {trials_csv}")
    print(f"Saved {summary_csv}")
    print(f"Saved {manifest_json}")
    print_ensemble_summary(summary_rows, pool_astar_build_s)

    return trial_rows, summary_rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Multi-trial MAPF ensemble benchmark with pooled routes and averaged metrics."
    )
    parser.add_argument("--scenario", default="sample2_default")
    parser.add_argument("--pool-size", type=int, default=32, help="Routes to pregenerate in pool")
    parser.add_argument("--num-agents", type=int, default=4, help="Agents sampled per trial (single-N runs)")
    parser.add_argument(
        "--num-agents-list",
        default=None,
        help="Comma-separated agent counts for sweep (e.g. 3,4,5,6,7,8,9,10)",
    )
    parser.add_argument("--num-trials", type=int, default=50, help="Number of random trials")
    parser.add_argument("--benchmark-seed", type=int, default=42, help="Seed for pool generation")
    parser.add_argument("--trial-seed", type=int, default=0, help="Base seed for trial sampling")
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument(
        "--output-prefix",
        default="results/mapf_ensemble/sample2",
        help="Output path prefix (without extension)",
    )
    parser.add_argument("--path-bank-path", default=None)
    parser.add_argument("--pregen-only", action="store_true", help="Build/cache pool then exit")
    parser.add_argument("--cbs-max-iter", type=int, default=0, help="0 = auto")
    parser.add_argument("--cbs-low-level-max-iter", type=int, default=500)
    parser.add_argument("--cbs-max-process", type=int, default=1)
    parser.add_argument("--pp-low-level-max-iter", type=int, default=0, help="0 = auto")
    parser.add_argument("--cbs-timeout-s", type=float, default=0.0, help="CBS wall-clock limit (0=off)")
    parser.add_argument("--pp-timeout-s", type=float, default=0.0, help="PP wall-clock limit (0=off)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trials where every method already succeeded (requires trials CSV)",
    )
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="With --resume, skip trials that already have any method row (ignore success)",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Rerun only methods that failed or are missing; keeps successful rows",
    )
    parser.add_argument(
        "--reaggregate-only",
        action="store_true",
        help="Rebuild summary CSV from existing trials CSV without running solvers",
    )
    parser.add_argument(
        "--no-fail-soft",
        action="store_true",
        help="Abort on first trial exception instead of recording failure rows",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=None,
        help="Max speed cap (m/s) for MAPF schedules and MILP; default 0.7",
    )
    parser.add_argument("--mapf-downsample", type=int, default=None)
    parser.add_argument("--crop-padding", type=int, default=40)
    parser.add_argument(
        "--mapf-coarse-block-policy",
        default="fine_center",
        choices=["any", "all", "majority", "center", "fine_center"],
    )
    milp_group = parser.add_argument_group("MILP geometry")
    milp_group.add_argument(
        "--milp-astar-mode",
        default="modified",
        choices=["vanilla", "modified"],
    )
    milp_group.add_argument("--milp-astar-vanilla", action="store_true")
    milp_group.add_argument("--heatmap-prefix", default=None)
    milp_group.add_argument("--heatmap-file", default=None)
    milp_group.add_argument(
        "--sparse-graph-threshold",
        type=float,
        default=2.0,
        help="Min directional heat for a sparse-graph edge when using --heatmap-prefix/file (default: 2.0).",
    )
    milp_group.add_argument(
        "--sparse-min-component-size",
        type=int,
        default=15,
        help="Prune heatmap graph components smaller than this many nodes (default: 15).",
    )
    milp_group.add_argument(
        "--refresh-social-bank",
        action="store_true",
        help="Re-run modified A* and overwrite cached paths in the path pool.",
    )
    milp_group.add_argument(
        "--milp-stride",
        type=int,
        default=1,
        help="Subsample MILP path waypoints every N vertices (1=full path, default).",
    )
    milp_group.add_argument(
        "--viz-event-waypoints",
        action="store_true",
        help="Save event interest-waypoint PNGs for each trial's MILP paths.",
    )
    milp_group.add_argument(
        "--viz-event-every",
        type=int,
        default=1,
        help="With --viz-event-waypoints, visualize every N trials (default 1 = all).",
    )
    milp_group.add_argument(
        "--print-event-milp-constraints",
        action="store_true",
        help="Print full event MILP objective and mutex constraints during trials.",
    )
    milp_group.add_argument(
        "--print-event-milp-constraints-every",
        type=int,
        default=1,
        help="With --print-event-milp-constraints, print every N trials (default 1).",
    )
    milp_group.add_argument(
        "--no-print-event-milp-constraints-on-error",
        action="store_true",
        help="Do not dump event MILP constraints when a trial fails.",
    )
    cli = parser.parse_args(argv)
    if cli.milp_astar_vanilla:
        cli.milp_astar_mode = "vanilla"
    if cli.heatmap_prefix and cli.heatmap_file:
        parser.error("Use only one of --heatmap-prefix or --heatmap-file.")
    if cli.milp_stride < 1:
        parser.error("--milp-stride must be >= 1.")
    if cli.viz_event_every < 1:
        parser.error("--viz-event-every must be >= 1.")
    if cli.print_event_milp_constraints_every < 1:
        parser.error("--print-event-milp-constraints-every must be >= 1.")

    if cli.retry_failures and cli.resume_partial:
        parser.error("Use only one of --retry-failures or --resume-partial.")
    if cli.retry_failures and cli.resume:
        parser.error("Use --retry-failures alone (it replaces --resume for failed rows).")

    num_agents_list = (
        _parse_int_list(cli.num_agents_list) if cli.num_agents_list else None
    )

    run_mapf_ensemble(
        scenario_name=cli.scenario,
        pool_size=cli.pool_size,
        num_agents=cli.num_agents,
        num_agents_list=num_agents_list,
        num_trials=cli.num_trials,
        benchmark_seed=cli.benchmark_seed,
        trial_seed=cli.trial_seed,
        max_attempts=cli.max_attempts,
        output_prefix=cli.output_prefix,
        path_bank_path=cli.path_bank_path,
        cbs_max_iter=cli.cbs_max_iter,
        cbs_low_level_max_iter=cli.cbs_low_level_max_iter,
        cbs_max_process=cli.cbs_max_process,
        pp_low_level_max_iter=cli.pp_low_level_max_iter,
        max_velocity_mps=cli.max_velocity,
        mapf_downsample=cli.mapf_downsample,
        crop_padding_cells=cli.crop_padding,
        coarse_block_policy=cli.mapf_coarse_block_policy,
        milp_astar_mode=cli.milp_astar_mode,
        heatmap_prefix=cli.heatmap_prefix,
        heatmap_file=cli.heatmap_file,
        sparse_graph_threshold=cli.sparse_graph_threshold,
        sparse_min_component_size=cli.sparse_min_component_size,
        refresh_social_bank=cli.refresh_social_bank,
        pregen_only=cli.pregen_only,
        milp_stride=cli.milp_stride,
        viz_event_waypoints=cli.viz_event_waypoints,
        viz_event_every=cli.viz_event_every,
        print_event_milp_constraints=cli.print_event_milp_constraints,
        print_event_milp_constraints_every=cli.print_event_milp_constraints_every,
        print_event_milp_constraints_on_error=not cli.no_print_event_milp_constraints_on_error,
        resume=cli.resume,
        resume_partial=cli.resume_partial,
        retry_failures=cli.retry_failures,
        reaggregate_only=cli.reaggregate_only,
        fail_soft=not cli.no_fail_soft,
        cbs_timeout_s=cli.cbs_timeout_s,
        pp_timeout_s=cli.pp_timeout_s,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
