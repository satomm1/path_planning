"""
Run CBS / PP / MILP / Event MILP simultaneous coordination and produce animations for visual verification.

Example:
  python -m social_path_planning.visualize_mapf --scenario sample2_default --num-agents 4 --output-dir results/mapf_comparison/verify
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from social_path_planning.benchmark_sparse_snapshots import (
    build_occ_grid,
    get_or_compute_astar_paths,
    load_or_generate_path_bank,
)
from social_path_planning.mapf_comparison import MotionConfig
from social_path_planning.mapf_comparison.pipeline import (
    MapfRunConfig,
    prepare_mapf_problem,
    print_grid_summary,
    run_event_milp_solver,
    run_mapf_solvers,
    run_milp_solver,
)
from social_path_planning.mapf_comparison.viz import (
    build_demo_open_scenario,
    create_mapf_animation,
    create_multi_panel_animation,
    create_side_by_side_animation,
    plot_routes_overlay,
    plot_side_by_side_snapshots,
    write_verification_report,
)


def run_visualization(
    scenario_name: str,
    num_agents: int,
    output_dir: Path,
    benchmark_seed: int = 42,
    max_attempts: int = 500,
    path_bank_path: Optional[Path] = None,
    cbs_max_iter: int = 0,
    cbs_low_level_max_iter: int = 0,
    cbs_max_process: int = 1,
    pp_low_level_max_iter: int = 0,
    mapf_downsample: int | None = None,
    coarse_block_policy: str = "fine_center",
    make_video: bool = True,
    make_snapshots: bool = True,
    make_static: bool = True,
    show: bool = False,
    demo_open: bool = False,
    crop: bool = True,
    crop_padding_cells: int = 40,
    run_milp: bool = True,
    run_event_milp: bool = True,
    milp_norm: int = 1,
    milp_stride: int = 1,
    milp_astar_mode: str = "modified",
    heatmap_prefix: Optional[str] = None,
    heatmap_file: Optional[str] = None,
    refresh_social_bank: bool = False,
    max_velocity_mps: float | None = None,
    run_cbs: bool = True,
) -> Dict[str, dict]:
    motion = (
        MotionConfig(max_velocity_mps=max_velocity_mps)
        if max_velocity_mps is not None
        else MotionConfig()
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if demo_open:
        occ_grid, routes = build_demo_open_scenario(num_agents)
        fixed_paths = [r["path"] for r in routes]
        scenario_name = "demo_open"
        map_resolution = float(occ_grid.resolution)
        statespace_hi = (
            occ_grid.origin_x + occ_grid.width * map_resolution,
            occ_grid.origin_y + occ_grid.height * map_resolution,
        )
    else:
        occ_grid, _, map_resolution, statespace_hi = build_occ_grid(scenario_name)
        if path_bank_path is None:
            path_bank_path = Path(
                output_dir / f"path_bank_{scenario_name}_seed{benchmark_seed}.json"
            )
        else:
            path_bank_path = Path(path_bank_path)
        path_bank, _meta = load_or_generate_path_bank(
            occ_grid=occ_grid,
            statespace_hi=statespace_hi,
            map_resolution=map_resolution,
            scenario_name=scenario_name,
            num_robots=num_agents,
            benchmark_seed=benchmark_seed,
            max_attempts=max_attempts,
            path_bank_path=path_bank_path,
        )
        routes = path_bank[:num_agents]
        fixed_paths = [r["path"] for r in routes]

    cfg = MapfRunConfig(
        downsample=mapf_downsample,
        crop_padding_cells=crop_padding_cells,
        coarse_block_policy=coarse_block_policy,
        pp_low_level_max_iter=pp_low_level_max_iter,
        cbs_low_level_max_iter=cbs_low_level_max_iter,
        cbs_max_iter=cbs_max_iter,
        cbs_max_process=cbs_max_process,
        motion=motion,
        run_cbs=run_cbs,
        run_pp=True,
        crop=crop,
    )
    grid_config, starts, goals, budget = prepare_mapf_problem(
        occ_grid, routes, cfg, demo_open=demo_open
    )
    print_grid_summary(occ_grid, grid_config, budget, routes, motion)
    print("\n--- MAPF solvers ---")
    results = run_mapf_solvers(
        occ_grid, routes, grid_config, starts, goals, budget, cfg
    )

    if run_milp or run_event_milp:
        milp_routes = routes
        if milp_astar_mode != "vanilla":
            if demo_open:
                from social_path_planning.benchmark_sparse_snapshots import (
                    enrich_routes_with_astar_paths,
                )

                social_graph = None
                if heatmap_prefix or heatmap_file:
                    from social_path_planning.compare_astar import (
                        build_sparse_graph_from_heatmap,
                    )

                    social_graph = build_sparse_graph_from_heatmap(
                        occ_grid,
                        heatmap_prefix=heatmap_prefix,
                        heatmap_file=heatmap_file,
                    )
                milp_routes = enrich_routes_with_astar_paths(
                    occ_grid,
                    statespace_hi,
                    map_resolution,
                    routes,
                    mode=milp_astar_mode,
                    social_graph=social_graph,
                )
            else:
                social_graph = None
                if heatmap_prefix or heatmap_file:
                    from social_path_planning.compare_astar import (
                        build_sparse_graph_from_heatmap,
                    )

                    social_graph = build_sparse_graph_from_heatmap(
                        occ_grid,
                        heatmap_prefix=heatmap_prefix,
                        heatmap_file=heatmap_file,
                    )
                    print("  Using heatmap sparse graph for social A*")
                milp_routes = get_or_compute_astar_paths(
                    occ_grid,
                    statespace_hi,
                    map_resolution,
                    routes,
                    milp_astar_mode,
                    path_bank_path,
                    social_graph=social_graph,
                    heatmap_prefix=heatmap_prefix,
                    heatmap_file=heatmap_file,
                    refresh=refresh_social_bank,
                )
        if run_milp:
            print("\n--- MILP timing ---")
            results["milp"] = run_milp_solver(
                occ_grid, milp_routes, norm=milp_norm, motion=motion, stride=milp_stride
            )
        else:
            print("\nSkipping classic MILP (--no-milp).")

        if run_event_milp:
            print("\n--- Event MILP timing ---")
            results["event_milp"] = run_event_milp_solver(
                occ_grid, milp_routes, norm=milp_norm, motion=motion, stride=milp_stride
            )
        else:
            print("\nSkipping event MILP (--no-event-milp).")
    else:
        print("\nSkipping MILP and event MILP (--no-milp and --no-event-milp).")

    write_verification_report(
        output_dir / "verification_report.json",
        scenario=scenario_name,
        num_agents=num_agents,
        benchmark_seed=benchmark_seed,
        motion=motion,
        results=results,
        milp_astar_mode=milp_astar_mode if (run_milp or run_event_milp) else None,
    )

    if make_static:
        plot_routes_overlay(
            occ_grid, fixed_paths, results, output_dir / "routes_overlay.png"
        )

    viz_methods = ("milp", "event_milp", "cbs", "pp")
    for method in viz_methods:
        if method not in results:
            continue
        entry = results[method]
        if not entry.get("success") or not entry.get("world_paths"):
            print(f"Skipping {method} animation (no successful solution).")
            continue
        if make_video:
            title = {
                "milp": f"MILP (social A*) — {scenario_name}, N={num_agents}",
                "event_milp": f"Event MILP — {scenario_name}, N={num_agents}",
                "cbs": f"CBS — {scenario_name}, N={num_agents}",
                "pp": f"PP — {scenario_name}, N={num_agents}",
            }.get(method, method)
            create_mapf_animation(
                occ_grid,
                entry["world_paths"],
                entry["time_lists"],
                output_dir / f"{method}_animation.gif",
                title=title,
                grid_paths_for_conflict=entry.get("grid_paths"),
                robot_radius_cells=entry.get("robot_radius_cells", 1),
                conflict_threshold_m=entry.get("conflict_threshold_m"),
                show=show,
            )

    if make_snapshots:
        end_times = []
        viz_methods = ("milp", "event_milp", "cbs", "pp")
        for method in viz_methods:
            if method not in results:
                continue
            if results[method].get("time_lists") and results[method]["time_lists"][0]:
                end_times.append(max(t[-1] for t in results[method]["time_lists"] if t))
        t_end = max(end_times) if end_times else 1.0
        snapshot_times = [0.0, 0.25 * t_end, 0.5 * t_end, 0.75 * t_end, t_end]
        plot_side_by_side_snapshots(
            occ_grid, results, output_dir / "snapshots_grid.png", snapshot_times
        )

    if make_video:
        comparison_panels = [
            (results[m], lbl)
            for m, lbl in (
                ("milp", "MILP"),
                ("event_milp", "Event MILP"),
                ("cbs", "CBS"),
                ("pp", "PP"),
            )
            if m in results and results[m].get("success")
        ]
        if len(comparison_panels) >= 2:
            create_multi_panel_animation(
                occ_grid, comparison_panels, output_dir / "methods_side_by_side.gif"
            )
        if results.get("cbs", {}).get("success") and results.get("pp", {}).get("success"):
            create_side_by_side_animation(
                occ_grid,
                results["cbs"],
                results["pp"],
                output_dir / "cbs_vs_pp_side_by_side.gif",
            )

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Animate or plot MILP, CBS, and PP for visual verification."
    )
    parser.add_argument("--scenario", default="sample2_default")
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--benchmark-seed", type=int, default=44)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--path-bank-path", default=None)
    parser.add_argument(
        "--output-dir",
        default="results/mapf_comparison/verify",
    )
    parser.add_argument(
        "--cbs-max-iter",
        type=int,
        default=0,
        help="CBS high-level iteration cap (0 = auto, scales with downsample)",
    )
    parser.add_argument(
        "--cbs-low-level-max-iter",
        type=int,
        default=0,
        help="STA* expansion cap for CBS (0 = auto)",
    )
    parser.add_argument("--cbs-max-process", type=int, default=1)
    parser.add_argument(
        "--pp-low-level-max-iter",
        type=int,
        default=0,
        help="STA* expansion cap for PP (0 = auto)",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=40,
        help="Cells padding around agents when cropping MAPF workspace",
    )
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--no-static", action="store_true")
    parser.add_argument("--no-cbs", action="store_true", help="Skip CBS solver")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--demo-open", action="store_true")
    parser.add_argument("--no-milp", action="store_true")
    parser.add_argument("--no-event-milp", action="store_true")
    parser.add_argument("--milp-norm", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--milp-stride",
        type=int,
        default=1,
        help="Subsample MILP/event polylines every N waypoints (default: 1).",
    )
    parser.add_argument(
        "--milp-astar-mode",
        default="modified",
        choices=["vanilla", "modified"],
    )
    parser.add_argument("--heatmap-prefix", default=None)
    parser.add_argument("--heatmap-file", default=None)
    parser.add_argument(
        "--refresh-social-bank",
        action="store_true",
        help="Re-run social/modified A* and overwrite cached paths in the path bank",
    )
    parser.add_argument("--max-velocity", type=float, default=None)
    parser.add_argument("--mapf-downsample", type=int, default=None)
    parser.add_argument(
        "--mapf-coarse-block-policy",
        default="fine_center",
        choices=["any", "all", "majority", "center", "fine_center"],
    )
    cli = parser.parse_args(argv)
    if cli.heatmap_prefix and cli.heatmap_file:
        parser.error("Use only one of --heatmap-prefix or --heatmap-file.")
    if cli.milp_stride < 1:
        parser.error("--milp-stride must be >= 1.")

    run_visualization(
        scenario_name=cli.scenario,
        num_agents=cli.num_agents,
        output_dir=Path(cli.output_dir),
        benchmark_seed=cli.benchmark_seed,
        max_attempts=cli.max_attempts,
        path_bank_path=Path(cli.path_bank_path) if cli.path_bank_path else None,
        cbs_max_iter=cli.cbs_max_iter,
        cbs_low_level_max_iter=cli.cbs_low_level_max_iter,
        cbs_max_process=cli.cbs_max_process,
        pp_low_level_max_iter=cli.pp_low_level_max_iter,
        make_video=not cli.no_video,
        make_snapshots=not cli.no_snapshots,
        make_static=not cli.no_static,
        show=cli.show,
        demo_open=cli.demo_open,
        crop=not cli.no_crop,
        crop_padding_cells=cli.crop_padding,
        run_milp=not cli.no_milp,
        run_event_milp=not cli.no_event_milp,
        milp_norm=cli.milp_norm,
        milp_stride=cli.milp_stride,
        milp_astar_mode=cli.milp_astar_mode,
        heatmap_prefix=cli.heatmap_prefix,
        heatmap_file=cli.heatmap_file,
        refresh_social_bank=cli.refresh_social_bank,
        max_velocity_mps=cli.max_velocity,
        mapf_downsample=cli.mapf_downsample,
        coarse_block_policy=cli.mapf_coarse_block_policy,
        run_cbs=not cli.no_cbs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
