"""
Multi-trial MAPF ensemble benchmark: pooled routes, random agent subsets, averaged metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.mapf_comparison import MapfRunConfig, MotionConfig
from social_path_planning.mapf_comparison.grid_traversability import ROBOT_DIAMETER_M
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.utils import snap_to_grid

_MODULE_DIR = Path(__file__).resolve().parent
from social_path_planning.mapf_comparison.ensemble import (
    aggregate_ensemble_metrics,
    ensure_modified_path_pool,
    print_ensemble_summary,
    run_ensemble_trial,
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
        "total_path_length_m",
        "conflict_count",
        "priority_order",
    ]


def _summary_fieldnames():
    return [
        "method",
        "num_trials",
        "success_count",
        "failure_count",
        "success_rate",
        "avg_solver_runtime_s",
        "std_solver_runtime_s",
        "avg_makespan_seconds",
        "std_makespan_seconds",
    ]


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    refresh_social_bank: bool = False,
    pregen_only: bool = False,
):
    if pool_size < num_agents:
        raise ValueError(f"pool_size ({pool_size}) must be >= num_agents ({num_agents})")
    if num_trials < 1 and not pregen_only:
        raise ValueError("num_trials must be >= 1")

    motion = (
        MotionConfig(max_velocity_mps=max_velocity_mps)
        if max_velocity_mps is not None
        else MotionConfig()
    )

    wall_cache_path = _wall_distance_cache_path(scenario_name)
    occ_grid, _, map_resolution, statespace_hi = build_occ_grid(scenario_name)
    output_prefix_path = Path(output_prefix)
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
        )
        print("MILP geometry: modified A* on heatmap sparse graph")
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
    )

    run_id = (
        f"ensemble_{scenario_name}_pool{pool_size}_n{num_agents}_"
        f"trials{num_trials}_seed{benchmark_seed}_trial{trial_seed}"
    )

    trial_rows = []
    t_loop_start = time.perf_counter()
    for trial_id in range(num_trials):
        this_trial_seed = int(trial_seed + trial_id)
        if (trial_id + 1) % 10 == 0 or trial_id == 0:
            print(f"Trial {trial_id + 1}/{num_trials} (seed={this_trial_seed})")
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
            social_graph=social_graph,
            heatmap_prefix=heatmap_prefix,
            heatmap_file=heatmap_file,
        )
        trial_rows.extend(rows)
    trial_loop_s = float(time.perf_counter() - t_loop_start)

    summary_rows = aggregate_ensemble_metrics(trial_rows)

    trials_csv = Path(f"{output_prefix}_trials.csv")
    summary_csv = Path(f"{output_prefix}_summary.csv")
    manifest_json = Path(f"{output_prefix}_manifest.json")

    _write_csv(trials_csv, trial_rows, _trial_fieldnames())
    _write_csv(summary_csv, summary_rows, _summary_fieldnames())

    try:
        import cbs_mapf

        cbs_version = getattr(cbs_mapf, "__version__", "unknown")
    except ImportError:
        cbs_version = None

    manifest = {
        "run_id": run_id,
        "config": {
            "scenario_name": scenario_name,
            "pool_size": int(pool_size),
            "num_agents": int(num_agents),
            "num_trials": int(num_trials),
            "benchmark_seed": int(benchmark_seed),
            "trial_seed": int(trial_seed),
            "max_attempts": int(max_attempts),
            "map_resolution": float(map_resolution),
            "motion": motion.to_manifest_dict(),
            "milp_astar_mode": milp_astar_mode,
            "heatmap_prefix": heatmap_prefix,
            "heatmap_file": heatmap_file,
            "mapf_downsample": mapf_downsample,
            "crop_padding_cells": int(crop_padding_cells),
            "coarse_block_policy": coarse_block_policy,
            "cbs_max_iter": int(cbs_max_iter),
            "cbs_low_level_max_iter": int(cbs_low_level_max_iter),
            "pp_low_level_max_iter": int(pp_low_level_max_iter),
            "wall_distance_cache_path": str(wall_cache_path.resolve()),
            "wall_distance_cache_used": bool(wall_cache_path.is_file()),
        },
        "path_bank": pool_meta,
        "path_bank_sha256": _file_sha256(path_bank_path) if path_bank_path.exists() else None,
        "pool_astar_build_s": pool_astar_build_s,
        "trial_loop_s": trial_loop_s,
        "summary": summary_rows,
        "libraries": {"cbs_mapf": cbs_version},
        "timing_notes": {
            "pool_astar_build_s": (
                "One-time modified/vanilla A* path generation during pool prep; "
                "excluded from avg_milp_solver_runtime_s in trial summary."
            ),
            "avg_solver_runtime_s": "Mean over successful trials only (MILP = cvxpy coordination only).",
            "avg_makespan_seconds": "Mean over successful trials only.",
        },
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

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
    parser.add_argument("--num-agents", type=int, default=4, help="Agents sampled per trial")
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
        "--refresh-social-bank",
        action="store_true",
        help="Re-run modified A* and overwrite cached paths in the path pool.",
    )
    cli = parser.parse_args(argv)
    if cli.milp_astar_vanilla:
        cli.milp_astar_mode = "vanilla"
    if cli.heatmap_prefix and cli.heatmap_file:
        parser.error("Use only one of --heatmap-prefix or --heatmap-file.")

    run_mapf_ensemble(
        scenario_name=cli.scenario,
        pool_size=cli.pool_size,
        num_agents=cli.num_agents,
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
        refresh_social_bank=cli.refresh_social_bank,
        pregen_only=cli.pregen_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
