"""
Benchmark MILP simultaneous coordination vs grid MAPF baselines (CBS, prioritized planning).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from social_path_planning.benchmark_sparse_snapshots import (
    build_occ_grid,
    get_or_compute_astar_paths,
    load_or_generate_path_bank,
)
from social_path_planning.mapf_comparison import MotionConfig
from social_path_planning.mapf_comparison.metrics import aggregate_milp_metrics
from social_path_planning.mapf_comparison.pipeline import MapfRunConfig, prepare_mapf_problem, run_mapf_solvers


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_milp(occ_grid, stage_paths, norm: int, motion: MotionConfig):
    from social_path_planning.multi_planning import MultiAgentSimultaneousPlanner

    planner = MultiAgentSimultaneousPlanner(occ_grid, paths=stage_paths, norm=norm)
    planner.assign_velocities([motion.max_velocity_mps] * len(stage_paths))
    t0 = time.perf_counter()
    status = "ok"
    agent_times = None
    try:
        agent_times = planner.plan()
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
    runtime_s = float(time.perf_counter() - t0)
    metrics = aggregate_milp_metrics(
        agent_times,
        runtime_s,
        status,
        norm=norm,
        world_paths=stage_paths,
        motion=motion,
    )
    return metrics, agent_times


def _serialize_mapf_paths(paths):
    if paths is None:
        return None
    return [[[int(p[0]), int(p[1])] for p in path] for path in paths]


def run_mapf_comparison(
    scenario_name: str,
    benchmark_seed: int,
    max_attempts: int,
    output_prefix: str,
    stage_sizes=None,
    path_bank_path=None,
    cbs_max_iter: int = 0,
    cbs_low_level_max_iter: int = 500,
    cbs_max_process: int = 1,
    pp_low_level_max_iter: int = 0,
    save_example_maps: bool = True,
    example_stage_size: int = 4,
    max_velocity_mps: float | None = None,
    mapf_downsample: int | None = None,
    crop_padding_cells: int = 40,
    coarse_block_policy: str = "fine_center",
    milp_astar_mode: str = "modified",
    heatmap_prefix: str | None = None,
    heatmap_file: str | None = None,
    refresh_social_bank: bool = False,
):
    motion = (
        MotionConfig(max_velocity_mps=max_velocity_mps)
        if max_velocity_mps is not None
        else MotionConfig()
    )
    stage_sizes = stage_sizes or [2, 4, 8, 16]
    stage_sizes = [int(s) for s in stage_sizes]
    if sorted(stage_sizes) != stage_sizes:
        raise ValueError("stage_sizes must be sorted ascending")
    if stage_sizes[0] < 2:
        raise ValueError("smallest stage must be >= 2")

    max_stage = stage_sizes[-1]
    occ_grid, _, map_resolution, statespace_hi = build_occ_grid(scenario_name)
    output_prefix_path = Path(output_prefix)
    if path_bank_path is None:
        path_bank_path = output_prefix_path.parent / f"{output_prefix_path.name}_path_bank.json"

    path_bank_path = Path(path_bank_path)
    path_bank, path_bank_meta = load_or_generate_path_bank(
        occ_grid=occ_grid,
        statespace_hi=statespace_hi,
        map_resolution=map_resolution,
        scenario_name=scenario_name,
        num_robots=max_stage,
        benchmark_seed=benchmark_seed,
        max_attempts=max_attempts,
        path_bank_path=path_bank_path,
    )

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

    run_id = f"mapf_seed_{benchmark_seed}_stages_{'-'.join(str(s) for s in stage_sizes)}"

    detailed_rows = []
    summary_rows = []
    example_payload = None

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

    for stage_size in stage_sizes:
        print(f"Stage size {stage_size}")
        stage_records = path_bank[:stage_size]
        bank_paths = [record["path"] for record in stage_records]
        milp_routes = get_or_compute_astar_paths(
            occ_grid,
            statespace_hi,
            map_resolution,
            stage_records,
            milp_astar_mode,
            path_bank_path,
            social_graph=social_graph,
            heatmap_prefix=heatmap_prefix,
            heatmap_file=heatmap_file,
            refresh=refresh_social_bank,
        )
        milp_paths = [route["path"] for route in milp_routes]
        grid_config, starts, goals, budget = prepare_mapf_problem(
            occ_grid, stage_records, mapf_cfg
        )
        solver_results = run_mapf_solvers(
            occ_grid,
            stage_records,
            grid_config,
            starts,
            goals,
            budget,
            mapf_cfg,
            verbose=False,
        )

        if "cbs" in solver_results:
            _append_rows(
                detailed_rows,
                run_id,
                stage_size,
                benchmark_seed,
                "cbs",
                solver_results["cbs"]["metrics"],
                stage_size,
            )
        if "pp" in solver_results:
            pp_entry = solver_results["pp"]
            _append_rows(
                detailed_rows,
                run_id,
                stage_size,
                benchmark_seed,
                "pp_path_length",
                pp_entry["metrics"],
                stage_size,
                extra={"priority_order": pp_entry.get("priority_order")},
            )

        cbs_paths = (solver_results.get("cbs") or {}).get("grid_paths")
        pp_paths = (solver_results.get("pp") or {}).get("grid_paths")

        # MILP sum of costs (geometry from milp_astar_mode; timing only)
        milp_soc_metrics, milp_times = _run_milp(occ_grid, milp_paths, norm=1, motion=motion)
        _append_rows(
            detailed_rows,
            run_id,
            stage_size,
            benchmark_seed,
            "milp_soc",
            milp_soc_metrics,
            stage_size,
        )

        # MILP makespan objective
        milp_ms_metrics, _ = _run_milp(occ_grid, milp_paths, norm=np.inf, motion=motion)
        _append_rows(
            detailed_rows,
            run_id,
            stage_size,
            benchmark_seed,
            "milp_makespan",
            milp_ms_metrics,
            stage_size,
        )

        if save_example_maps and stage_size == example_stage_size and example_payload is None:
            example_payload = {
                "stage_size": stage_size,
                "fixed_paths": bank_paths,
                "milp_paths": milp_paths,
                "milp_times": milp_times,
                "cbs_paths_world": (solver_results.get("cbs") or {}).get("world_polylines") or [],
                "pp_paths_world": (solver_results.get("pp") or {}).get("world_polylines") or [],
            }

        for method in ("cbs", "pp_path_length", "milp_soc", "milp_makespan"):
            row = next(r for r in detailed_rows if r["stage_size"] == stage_size and r["method"] == method)
            summary_rows.append(_summary_from_detail(row))

    output_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_csv = Path(f"{output_prefix}_detailed.csv")
    summary_csv = Path(f"{output_prefix}_summary.csv")
    manifest_json = Path(f"{output_prefix}_manifest.json")

    _write_csv(detailed_csv, detailed_rows, _detail_fieldnames())
    _write_csv(summary_csv, summary_rows, _summary_fieldnames())

    try:
        import cbs_mapf

        cbs_version = getattr(cbs_mapf, "__version__", "unknown")
    except ImportError:
        cbs_version = None

    manifest = {
        "config": {
            "scenario_name": scenario_name,
            "benchmark_seed": int(benchmark_seed),
            "max_attempts": int(max_attempts),
            "stage_sizes": stage_sizes,
            "cbs_max_iter": int(cbs_max_iter),
            "cbs_low_level_max_iter": int(cbs_low_level_max_iter),
            "pp_low_level_max_iter": int(pp_low_level_max_iter),
            "map_resolution": float(map_resolution),
            "robot_radius_cells": int(grid_config.robot_radius),
            "motion": motion.to_manifest_dict(),
            "milp_astar_mode": milp_astar_mode,
            "heatmap_prefix": heatmap_prefix,
            "heatmap_file": heatmap_file,
        },
        "path_bank": path_bank_meta,
        "path_bank_sha256": _file_sha256(Path(path_bank_path)) if Path(path_bank_path).exists() else None,
        "libraries": {"cbs_mapf": cbs_version},
        "run_id": run_id,
        "summary": summary_rows,
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    from social_path_planning.mapf_comparison.plot import plot_metrics_bars, plot_example_map

    plot_metrics_bars(summary_rows, Path(f"{output_prefix}_metrics.png"))
    if example_payload is not None:
        example_dir = Path(f"{output_prefix}_example_maps")
        plot_example_map(
            occ_grid,
            example_payload["fixed_paths"],
            example_payload.get("cbs_paths_world") or [],
            example_payload.get("pp_paths_world") or [],
            example_dir / f"stage_{example_payload['stage_size']}_overlay.png",
        )

    print(f"Saved {detailed_csv}")
    print(f"Saved {summary_csv}")
    print(f"Saved {manifest_json}")
    return detailed_rows, summary_rows


def _append_rows(detailed_rows, run_id, stage_size, seed, method, metrics, num_agents, extra=None):
    row = {
        "run_id": run_id,
        "stage_size": int(stage_size),
        "num_agents": int(num_agents),
        "method": method,
        "success": metrics.get("success"),
        "status": metrics.get("status"),
        "solver_runtime_s": metrics.get("solver_runtime_s", metrics.get("runtime_s")),
        "runtime_s": metrics.get("runtime_s"),
        "soc_seconds": metrics.get("soc_seconds"),
        "makespan_seconds": metrics.get("makespan_seconds"),
        "total_path_length_m": metrics.get("total_path_length_m"),
        "max_velocity_mps": metrics.get("max_velocity_mps"),
        "seed": int(seed),
        "soc_timesteps": metrics.get("soc_timesteps"),
        "makespan_timesteps": metrics.get("makespan_timesteps"),
        "conflict_count": metrics.get("conflict_count"),
        "cell_size_m": metrics.get("cell_size_m"),
        "downsample": metrics.get("downsample"),
    }
    if extra:
        row.update(extra)
    detailed_rows.append(row)


def _summary_from_detail(row):
    return {
        "run_id": row["run_id"],
        "stage_size": row["stage_size"],
        "method": row["method"],
        "success": row["success"],
        "solver_runtime_s": row.get("solver_runtime_s", row.get("runtime_s")),
        "runtime_s": row.get("runtime_s"),
        "soc_seconds": row.get("soc_seconds"),
        "makespan_seconds": row.get("makespan_seconds"),
        "total_path_length_m": row.get("total_path_length_m"),
        "max_velocity_mps": row.get("max_velocity_mps"),
        "soc_timesteps": row.get("soc_timesteps"),
        "makespan_timesteps": row.get("makespan_timesteps"),
        "conflict_count": row.get("conflict_count"),
        "seed": row["seed"],
    }


def _detail_fieldnames():
    return [
        "run_id",
        "stage_size",
        "num_agents",
        "method",
        "success",
        "status",
        "solver_runtime_s",
        "runtime_s",
        "soc_seconds",
        "makespan_seconds",
        "total_path_length_m",
        "max_velocity_mps",
        "soc_timesteps",
        "makespan_timesteps",
        "conflict_count",
        "cell_size_m",
        "downsample",
        "seed",
        "priority_order",
    ]


def _summary_fieldnames():
    return [
        "run_id",
        "stage_size",
        "method",
        "success",
        "solver_runtime_s",
        "runtime_s",
        "soc_seconds",
        "makespan_seconds",
        "total_path_length_m",
        "max_velocity_mps",
        "soc_timesteps",
        "makespan_timesteps",
        "conflict_count",
        "seed",
    ]


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare MILP timing vs CBS and prioritized MAPF.")
    parser.add_argument("--scenario", default="sample2_default")
    parser.add_argument("--benchmark-seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument(
        "--output-prefix",
        default="results/mapf_comparison/sample2_mapf",
        help="Output path prefix (without extension)",
    )
    parser.add_argument("--path-bank-path", default=None)
    parser.add_argument("--stage-sizes", default="2,4,8,16")
    parser.add_argument("--cbs-max-iter", type=int, default=0, help="0 = auto")
    parser.add_argument("--cbs-low-level-max-iter", type=int, default=500)
    parser.add_argument("--cbs-max-process", type=int, default=1)
    parser.add_argument("--pp-low-level-max-iter", type=int, default=0, help="0 = auto")
    parser.add_argument("--no-example-maps", action="store_true")
    parser.add_argument("--example-stage-size", type=int, default=4)
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=None,
        help="Max speed cap (m/s) for MAPF schedules and MILP; default 0.7",
    )
    parser.add_argument(
        "--mapf-downsample",
        type=int,
        default=None,
        help="MAPF grid downsample factor (default: auto by map size)",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=40,
        help="Coarse-cell padding around agents when cropping MAPF workspace",
    )
    parser.add_argument(
        "--mapf-coarse-block-policy",
        default="fine_center",
        choices=["any", "all", "majority", "center", "fine_center"],
        help="How fine occupancy merges into coarse cells (fine_center matches A*; any=strictest)",
    )
    milp_group = parser.add_argument_group("MILP geometry")
    milp_group.add_argument(
        "--milp-astar-mode",
        default="modified",
        choices=["vanilla", "modified"],
        help="A* mode for MILP fixed polylines (default: modified social paths).",
    )
    milp_group.add_argument(
        "--milp-astar-vanilla",
        action="store_true",
        help="Shortcut for --milp-astar-mode vanilla (path-bank polylines).",
    )
    milp_group.add_argument("--heatmap-prefix", default=None)
    milp_group.add_argument("--heatmap-file", default=None)
    milp_group.add_argument(
        "--refresh-social-bank",
        action="store_true",
        help="Re-run social/modified A* and overwrite cached paths in the path bank.",
    )
    cli = parser.parse_args(argv)
    if cli.milp_astar_vanilla:
        cli.milp_astar_mode = "vanilla"
    if cli.heatmap_prefix and cli.heatmap_file:
        parser.error("Use only one of --heatmap-prefix or --heatmap-file.")

    stage_sizes = [int(s.strip()) for s in cli.stage_sizes.split(",") if s.strip()]
    run_mapf_comparison(
        scenario_name=cli.scenario,
        benchmark_seed=cli.benchmark_seed,
        max_attempts=cli.max_attempts,
        output_prefix=cli.output_prefix,
        stage_sizes=stage_sizes,
        path_bank_path=cli.path_bank_path,
        cbs_max_iter=cli.cbs_max_iter,
        cbs_low_level_max_iter=cli.cbs_low_level_max_iter,
        cbs_max_process=cli.cbs_max_process,
        pp_low_level_max_iter=cli.pp_low_level_max_iter,
        save_example_maps=not cli.no_example_maps,
        example_stage_size=cli.example_stage_size,
        max_velocity_mps=cli.max_velocity,
        mapf_downsample=cli.mapf_downsample,
        crop_padding_cells=cli.crop_padding,
        coarse_block_policy=cli.mapf_coarse_block_policy,
        milp_astar_mode=cli.milp_astar_mode,
        heatmap_prefix=cli.heatmap_prefix,
        heatmap_file=cli.heatmap_file,
        refresh_social_bank=cli.refresh_social_bank,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
