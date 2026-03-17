import argparse
import csv
import json
import pickle
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from a_star import AStar, AStar_With_Graph
from grid_loader import load_grid_scenario
from occupancy_grid import StochOccupancyGrid2D
from utils import snap_to_grid


def build_occ_grid(scenario_name):
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    map_dim = [round(map_size[i] / map_resolution) for i in range(2)]
    occ_grid = StochOccupancyGrid2D(
        map_resolution,
        map_dim[0],
        map_dim[1],
        0,
        0,
        10,
        occ.T,
    )
    statespace_hi = snap_to_grid(map_size, map_resolution)
    return occ_grid, map_size, map_resolution, statespace_hi


def compute_path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        p0 = np.array(path[i], dtype=float)
        p1 = np.array(path[i + 1], dtype=float)
        total += float(np.linalg.norm(p1 - p0))
    return total


def generate_random_free_point(occ_grid, rng):
    low = np.array([occ_grid.origin_x, occ_grid.origin_y], dtype=float)
    high = np.array(
        [
            occ_grid.origin_x + occ_grid.width * occ_grid.resolution,
            occ_grid.origin_y + occ_grid.height * occ_grid.resolution,
        ],
        dtype=float,
    )
    while True:
        xy = rng.uniform(low=low, high=high)
        x, y = occ_grid.snap_to_grid(xy)
        if occ_grid.is_free((x, y)):
            return (x, y)


def discover_graph_snapshots(timeline_dir):
    timeline_dir = Path(timeline_dir)
    metadata_path = timeline_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    graph_paths = []
    if isinstance(metadata.get("graph_snapshots"), list) and metadata["graph_snapshots"]:
        for p in metadata["graph_snapshots"]:
            pp = Path(p)
            if not pp.is_absolute():
                pp = timeline_dir / pp
            if pp.exists():
                graph_paths.append(pp.resolve())

    if not graph_paths:
        graph_paths = sorted(timeline_dir.glob("graph_*.pkl"), key=extract_snapshot_idx)

    if not graph_paths:
        raise FileNotFoundError(f"No graph snapshots found in {timeline_dir}")

    return graph_paths, metadata


def extract_snapshot_idx(path_obj):
    path_obj = Path(path_obj)
    match = re.search(r"graph_(\d+)\.pkl$", path_obj.name)
    if not match:
        return -1
    return int(match.group(1))


def sample_fixed_routes(occ_grid, num_routes, rng_seed, max_attempts):
    rng = np.random.default_rng(rng_seed)
    routes = []
    seen = set()
    attempts = 0
    while len(routes) < num_routes:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Reached max attempts ({max_attempts}) before collecting {num_routes} routes.")

        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        if x_init == x_goal:
            continue
        key = (float(x_init[0]), float(x_init[1]), float(x_goal[0]), float(x_goal[1]))
        if key in seen:
            continue
        seen.add(key)
        routes.append((x_init, x_goal))
    return routes, attempts


def run_baseline_for_routes(occ_grid, statespace_hi, map_resolution, routes):
    baseline = []
    for route_id, (x_init, x_goal) in enumerate(routes):
        planner = AStar([0, 0], statespace_hi, x_init, x_goal, occ_grid, resolution=map_resolution)
        t0 = time.perf_counter()
        solved = planner.solve(mode="vanilla")
        dt = time.perf_counter() - t0
        baseline.append(
            {
                "route_id": route_id,
                "x_init": [float(x_init[0]), float(x_init[1])],
                "x_goal": [float(x_goal[0]), float(x_goal[1])],
                "success": bool(solved),
                "solve_time_s": float(dt),
                "path_length": float(compute_path_length(planner.path if solved else None)),
                "explored_nodes": int(len(planner.closed_set)),
            }
        )
    return baseline


def run_snapshot_for_routes(occ_grid, statespace_hi, map_resolution, routes, graph):
    records = []
    for route_id, (x_init, x_goal) in enumerate(routes):
        planner = AStar_With_Graph([0, 0], statespace_hi, x_init, x_goal, occ_grid, graph, resolution=map_resolution)
        t0 = time.perf_counter()
        solved = planner.solve(mode="vanilla")
        dt = time.perf_counter() - t0
        records.append(
            {
                "route_id": route_id,
                "success": bool(solved),
                "solve_time_s": float(dt),
                "path_length": float(compute_path_length(planner.path if solved else None)),
                "explored_nodes": int(len(planner.closed_set)),
            }
        )
    return records


def summarize(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }

def run_benchmark(
    timeline_dir,
    scenario_name,
    num_routes,
    benchmark_seed,
    max_attempts,
    output_prefix,
    make_plot=False,
):
    graph_paths, metadata = discover_graph_snapshots(timeline_dir)
    occ_grid, _, map_resolution, statespace_hi = build_occ_grid(scenario_name)
    routes, sampling_attempts = sample_fixed_routes(occ_grid, num_routes, benchmark_seed, max_attempts)

    baseline_records = run_baseline_for_routes(occ_grid, statespace_hi, map_resolution, routes)
    baseline_by_route = {r["route_id"]: r for r in baseline_records}

    detailed_rows = []
    summary_rows = []

    baseline_times = [r["solve_time_s"] for r in baseline_records if r["success"]]
    baseline_summary = summarize(baseline_times)
    baseline_success_rate = float(np.mean([r["success"] for r in baseline_records])) if baseline_records else 0.0
    summary_rows.append(
        {
            "snapshot_idx": 0,
            "snapshot_file": "baseline_vanilla",
            "graph_nodes": 0,
            "graph_edges": 0,
            "num_routes": len(baseline_records),
            "success_rate": baseline_success_rate,
            "solve_time_mean_s": baseline_summary["mean"],
            "solve_time_median_s": baseline_summary["median"],
            "solve_time_p95_s": baseline_summary["p95"],
            "speedup_mean_vs_baseline": 1.0,
            "speedup_median_vs_baseline": 1.0,
            "speedup_p95_vs_baseline": 1.0,
        }
    )

    for row in baseline_records:
        detailed_rows.append(
            {
                "snapshot_idx": 0,
                "snapshot_file": "baseline_vanilla",
                "route_id": row["route_id"],
                "x_init": row["x_init"],
                "x_goal": row["x_goal"],
                "success": row["success"],
                "solve_time_s": row["solve_time_s"],
                "path_length": row["path_length"],
                "explored_nodes": row["explored_nodes"],
                "graph_nodes": 0,
                "graph_edges": 0,
                "speedup_vs_baseline": 1.0,
                "baseline_solve_time_s": row["solve_time_s"],
            }
        )

    for graph_path in graph_paths:
        snapshot_idx = extract_snapshot_idx(graph_path)
        with graph_path.open("rb") as f:
            graph = pickle.load(f)

        snapshot_records = run_snapshot_for_routes(occ_grid, statespace_hi, map_resolution, routes, graph)

        speedups = []
        solve_times_success = []
        success_flags = []
        for row in snapshot_records:
            baseline_row = baseline_by_route[row["route_id"]]
            speedup = float("nan")
            if row["success"] and baseline_row["success"] and row["solve_time_s"] > 1e-12:
                speedup = baseline_row["solve_time_s"] / row["solve_time_s"]
                speedups.append(speedup)

            if row["success"]:
                solve_times_success.append(row["solve_time_s"])
            success_flags.append(row["success"])

            detailed_rows.append(
                {
                    "snapshot_idx": snapshot_idx,
                    "snapshot_file": str(graph_path),
                    "route_id": row["route_id"],
                    "x_init": baseline_row["x_init"],
                    "x_goal": baseline_row["x_goal"],
                    "success": row["success"],
                    "solve_time_s": row["solve_time_s"],
                    "path_length": row["path_length"],
                    "explored_nodes": row["explored_nodes"],
                    "graph_nodes": int(graph.number_of_nodes()),
                    "graph_edges": int(graph.number_of_edges()),
                    "speedup_vs_baseline": speedup,
                    "baseline_solve_time_s": baseline_row["solve_time_s"],
                }
            )

        time_stats = summarize(solve_times_success)
        speedup_stats = summarize(speedups)
        success_rate = float(np.mean(success_flags)) if success_flags else 0.0
        summary_rows.append(
            {
                "snapshot_idx": snapshot_idx,
                "snapshot_file": str(graph_path),
                "graph_nodes": int(graph.number_of_nodes()),
                "graph_edges": int(graph.number_of_edges()),
                "num_routes": len(snapshot_records),
                "success_rate": success_rate,
                "solve_time_mean_s": time_stats["mean"],
                "solve_time_median_s": time_stats["median"],
                "solve_time_p95_s": time_stats["p95"],
                "speedup_mean_vs_baseline": speedup_stats["mean"],
                "speedup_median_vs_baseline": speedup_stats["median"],
                "speedup_p95_vs_baseline": speedup_stats["p95"],
            }
        )

    summary_rows = sorted(summary_rows, key=lambda x: x["snapshot_idx"])

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detailed_csv_path = Path(f"{output_prefix}_detailed.csv")
    summary_csv_path = Path(f"{output_prefix}_summary.csv")
    summary_json_path = Path(f"{output_prefix}_summary.json")

    with detailed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot_idx",
                "snapshot_file",
                "route_id",
                "x_init",
                "x_goal",
                "success",
                "solve_time_s",
                "path_length",
                "explored_nodes",
                "graph_nodes",
                "graph_edges",
                "speedup_vs_baseline",
                "baseline_solve_time_s",
            ],
        )
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(row)

    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot_idx",
                "snapshot_file",
                "graph_nodes",
                "graph_edges",
                "num_routes",
                "success_rate",
                "solve_time_mean_s",
                "solve_time_median_s",
                "solve_time_p95_s",
                "speedup_mean_vs_baseline",
                "speedup_median_vs_baseline",
                "speedup_p95_vs_baseline",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    payload = {
        "config": {
            "timeline_dir": str(Path(timeline_dir).resolve()),
            "scenario_name": scenario_name,
            "num_routes": num_routes,
            "benchmark_seed": benchmark_seed,
            "max_attempts": max_attempts,
            "metadata_scenario_name": metadata.get("scenario_name"),
        },
        "sampling_attempts": sampling_attempts,
        "baseline": baseline_records,
        "summary": summary_rows,
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_path = None
    if make_plot:
        baseline_row = next((r for r in summary_rows if int(r["snapshot_idx"]) == 0), None)
        summary_no_baseline = [r for r in summary_rows if int(r["snapshot_idx"]) > 0]
        if summary_no_baseline and baseline_row is not None:
            xs_time = [0] + [int(r["snapshot_idx"]) for r in summary_no_baseline]
            xs_speedup = [int(r["snapshot_idx"]) for r in summary_no_baseline]
            ys_time = [baseline_row["solve_time_mean_s"]] + [r["solve_time_mean_s"] for r in summary_no_baseline]
            ys_speedup = [r["speedup_mean_vs_baseline"] for r in summary_no_baseline]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(xs_time, ys_time, marker="o")
            axes[0].set_xlabel("Social Graph Index")
            axes[0].set_ylabel("Mean Solve Time (s)")
            axes[0].set_title("Social Graph Mean Solve Time")
            axes[0].set_xticks(xs_time)
            axes[0].set_xticklabels(["A*"] + [str(x) for x in xs_time[1:]])
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(xs_speedup, ys_speedup, marker="o")
            axes[1].axhline(1.0, linestyle="--", linewidth=1.0)
            axes[1].set_xlabel("Social Graph Index")
            axes[1].set_ylabel("Mean Speedup vs A*")
            axes[1].set_title("Social Graph Speedup")
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            plot_path = Path(f"{output_prefix}_plot.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved detailed results: {detailed_csv_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved summary JSON: {summary_json_path}")
    if plot_path is not None:
        print(f"Saved summary plot: {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark intermediate sparse graph snapshots vs vanilla A* baseline.")
    parser.add_argument("--timeline-dir", type=str, required=True, help="Directory containing graph_*.pkl and run_metadata.json")
    parser.add_argument("--scenario-name", type=str, default=None, help="Scenario name for occupancy reconstruction")
    parser.add_argument("--num-routes", type=int, required=True, help="Number of fixed seeded route pairs to benchmark")
    parser.add_argument("--benchmark-seed", type=int, default=0, help="Seed for fixed benchmark route generation")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum attempts to sample fixed unique routes")
    parser.add_argument("--output-prefix", type=str, default=None, help="Output prefix path (without extension)")
    parser.add_argument("--plot", action="store_true", help="Generate benchmark plot image")
    args = parser.parse_args()

    if args.num_routes <= 0:
        raise ValueError("--num-routes must be > 0")
    if args.max_attempts is None:
        args.max_attempts = max(100, 100 * args.num_routes)
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be > 0")

    timeline_dir = Path(args.timeline_dir)
    metadata_path = timeline_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    if args.scenario_name is None:
        args.scenario_name = metadata.get("scenario_name", None)
    if args.scenario_name is None:
        raise ValueError("--scenario-name is required when scenario is missing from run_metadata.json")

    if args.output_prefix is None:
        args.output_prefix = str(timeline_dir / "benchmark")
    return args


if __name__ == "__main__":
    cli = parse_args()
    run_benchmark(
        timeline_dir=cli.timeline_dir,
        scenario_name=cli.scenario_name,
        num_routes=cli.num_routes,
        benchmark_seed=cli.benchmark_seed,
        max_attempts=cli.max_attempts,
        output_prefix=cli.output_prefix,
        make_plot=cli.plot,
    )
