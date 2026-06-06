import argparse
import csv
import json
import os
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from social_path_planning.a_star import AStar, AStar_With_Graph
from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.mapf_comparison.grid_traversability import ROBOT_DIAMETER_M
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.rrt_star import RRTStar
from social_path_planning.utils import snap_to_grid


SOLVER_MODES = ("vanilla", "modified", "rrt_vanilla")
FAILURE_POLICY = "resample_until_all_succeed"
SOLVER_PLOT_TITLES = {
    "vanilla": "A* vanilla",
    "modified": "A* social",
    "rrt_vanilla": "RRT* vanilla",
}
RIGHT_WALL_EXCLUSION_RADIUS = 5.0
RIGHT_WALL_LARGE_RATIO = 0.7


def route_pair_key(x_init, x_goal):
    return (
        float(x_init[0]),
        float(x_init[1]),
        float(x_goal[0]),
        float(x_goal[1]),
    )


def build_occ_grid(scenario_name):
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    map_dim = [round(map_size[i] / map_resolution) for i in range(2)]
    cache_path = _MODULE_DIR / "environments" / f"{scenario_name}_wall_dist.npz"
    occ_grid = StochOccupancyGrid2D(
        map_resolution,
        map_dim[0],
        map_dim[1],
        0,
        0,
        10,
        occ.T,
        robot_d=ROBOT_DIAMETER_M,
        wall_distance_cache_path=cache_path if cache_path.is_file() else None,
        auto_build_wall_distance_cache=True,
    )
    statespace_hi = snap_to_grid(map_size, map_resolution)
    return occ_grid, map_size, map_resolution, statespace_hi


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


def compute_path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        p0 = np.array(path[i], dtype=float)
        p1 = np.array(path[i + 1], dtype=float)
        total += float(np.linalg.norm(p1 - p0))
    return total


def compute_right_wall_distance_analysis(path, occ_grid, dist_thresh=15.0):
    dists = []
    used_segment_mask = []
    if path is None or len(path) < 2:
        return {
            "distances": dists,
            "used_segment_mask": used_segment_mask,
        }
    start = np.array(path[0], dtype=float)
    goal = np.array(path[-1], dtype=float)
    large_dist_thresh = RIGHT_WALL_LARGE_RATIO * dist_thresh
    for i in range(len(path) - 1):
        p0 = np.array(path[i], dtype=float)
        p1 = np.array(path[i + 1], dtype=float)
        if (
            np.linalg.norm(p1 - start) <= RIGHT_WALL_EXCLUSION_RADIUS
            or np.linalg.norm(p1 - goal) <= RIGHT_WALL_EXCLUSION_RADIUS
        ):
            used_segment_mask.append(False)
            continue
        step = p1 - p0
        norm = np.linalg.norm(step)
        if norm < 1e-9:
            used_segment_mask.append(False)
            continue
        travel_dir = step / norm
        d_right = float(occ_grid.dist_to_wall_right(path[i + 1], travel_dir, dist_thresh=dist_thresh))

        # Only compute left distance when right distance looks very large.
        if d_right >= large_dist_thresh:
            d_left = float(occ_grid.dist_to_wall_left(path[i + 1], travel_dir, dist_thresh=dist_thresh))
            # Exclude intersection/open-space samples where both sides are very large.
            if d_left >= large_dist_thresh and d_right >= large_dist_thresh:
                used_segment_mask.append(False)
                continue

        dists.append(d_right)
        used_segment_mask.append(True)
    return {
        "distances": dists,
        "used_segment_mask": used_segment_mask,
    }


def compute_metrics(path, occ_grid, dist_thresh=15.0, return_analysis=False):
    path_length = compute_path_length(path)
    wall_analysis = compute_right_wall_distance_analysis(path, occ_grid, dist_thresh=dist_thresh)
    wall_dists = wall_analysis["distances"]
    if wall_dists:
        right_wall_avg = float(np.mean(wall_dists))
        right_wall_std = float(np.std(wall_dists))
    else:
        right_wall_avg = float("nan")
        right_wall_std = float("nan")
    metrics = {
        "path_length": path_length,
        "right_wall_avg": right_wall_avg,
        "right_wall_std": right_wall_std,
        "num_wall_samples": len(wall_dists),
    }
    if return_analysis:
        return metrics, wall_analysis
    return metrics


def build_sparse_graph_from_heatmap(
    occ_grid,
    *,
    heatmap_prefix=None,
    heatmap_path=None,
    sparse_graph_threshold=2.0,
    sparse_min_component_size=15,
):
    """Load a directional heatmap and build the pruned sparse graph for AStar_With_Graph."""
    from social_path_planning.sparse_graph import FrequentSubgraph

    if not heatmap_prefix and not heatmap_path:
        raise ValueError("build_sparse_graph_from_heatmap requires heatmap_prefix or heatmap_path")

    frequent = FrequentSubgraph(occ_grid)
    hm = frequent.heat_map_object
    expected = (hm.height, hm.width, 8)

    if heatmap_path:
        heatmap_path = os.path.abspath(heatmap_path)
        if not os.path.exists(heatmap_path):
            raise FileNotFoundError(f"Heatmap file not found: {heatmap_path}")
        hm.heatmap = np.load(heatmap_path)
        print(f"Loaded heatmap from {heatmap_path}")
    else:
        frequent = FrequentSubgraph(occ_grid, heat_map_filename=heatmap_prefix)

    if frequent.heat_map.shape != expected:
        raise ValueError(
            f"Heatmap shape mismatch. Expected {expected}, got {frequent.heat_map.shape}. "
            "Wrong scenario or not a HeatMap2DVector save."
        )

    frequent.build_graph(threshold=sparse_graph_threshold, reset_graph=True)
    frequent.prune_graph(min_component_size=sparse_min_component_size)
    print(
        f"Sparse graph: {frequent.graph.number_of_nodes()} nodes, "
        f"{frequent.graph.number_of_edges()} edges "
        f"(threshold={sparse_graph_threshold}, min_component_size={sparse_min_component_size})"
    )

    ax = frequent.visualize_graph(
        show=False,
        title="Environment 3",
        xlabel="",
        ylabel="",
        title_fontsize=35,
        label_fontsize=16,
        tick_fontsize=25,
        edge_color="red",
        edge_linewidth=1,
        edge_alpha=1,
        figsize=(10, 10))

    # Save the figure
    fig = ax.figure
    fig.tight_layout()
    fig.savefig("outputs_timeline/env3.png", dpi=600)

    return frequent.graph


def run_solver(
    mode,
    occ_grid,
    statespace_hi,
    x_init,
    x_goal,
    resolution,
    *,
    social_graph=None,
    rrt_kwargs=None,
):
    rrt_kwargs = {} if rrt_kwargs is None else dict(rrt_kwargs)

    if mode in ("vanilla", "modified"):
        if mode == "modified" and social_graph is not None:
            planner = AStar_With_Graph(
                [0, 0],
                statespace_hi,
                x_init,
                x_goal,
                occ_grid,
                social_graph,
                resolution=resolution,
                desired_dist_right_extra=0.25,
            )
        else:
            planner = AStar(
                [0, 0],
                statespace_hi,
                x_init,
                x_goal,
                occ_grid,
                resolution=resolution,
            )
        solved, solve_time = planner.solve(mode=mode, return_timing=True)
    elif mode == "rrt_vanilla":
        planner = RRTStar(
            [0, 0],
            statespace_hi,
            x_init,
            x_goal,
            occ_grid,
            resolution=resolution,
            **rrt_kwargs,
        )
        solved, solve_time = planner.solve(mode="vanilla", return_timing=True)
    else:
        raise ValueError(f"Unsupported solver mode '{mode}'")

    if not solved:
        return None, solve_time
    return planner.path, solve_time


def summarize_by_solver(route_records):
    summary = {}
    for mode in SOLVER_MODES:
        lengths = np.array([r[mode]["path_length"] for r in route_records], dtype=float)
        wall_avgs = np.array([r[mode]["right_wall_avg"] for r in route_records], dtype=float)
        wall_stds = np.array([r[mode]["right_wall_std"] for r in route_records], dtype=float)
        solve_times = np.array([r[mode].get("solve_time_sec", np.nan) for r in route_records], dtype=float)

        summary[mode] = {
            "num_routes": int(len(route_records)),
            "path_length_mean": float(np.nanmean(lengths)),
            "path_length_std": float(np.nanstd(lengths)),
            "right_wall_avg_mean": float(np.nanmean(wall_avgs)),
            "right_wall_avg_std": float(np.nanstd(wall_avgs)),
            "right_wall_std_mean": float(np.nanmean(wall_stds)),
            "right_wall_std_std": float(np.nanstd(wall_stds)),
            "solve_time_mean_sec": float(np.nanmean(solve_times)),
            "solve_time_std_sec": float(np.nanstd(solve_times)),
        }
    return summary


def load_resume_payload(path):
    resume_path = Path(path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume file not found: {resume_path}")
    if resume_path.suffix.lower() != ".json":
        raise ValueError("Resume file must be a .json file generated by compare_astar.py")

    with resume_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "routes" not in payload or "config" not in payload:
        raise ValueError("Resume file is missing required keys: 'config' and/or 'routes'")
    if not isinstance(payload["routes"], list):
        raise ValueError("Resume file 'routes' must be a list")
    return payload


def _modified_solver_config(heatmap_prefix, heatmap_file, sparse_graph_threshold, sparse_min_component_size):
    if heatmap_prefix or heatmap_file:
        return {
            "modified_solver": "heatmap_graph",
            "heatmap_prefix": heatmap_prefix,
            "heatmap_file": heatmap_file,
            "sparse_graph_threshold": sparse_graph_threshold,
            "sparse_min_component_size": sparse_min_component_size,
        }
    return {"modified_solver": "rightness_penalty"}


def validate_resume_compatibility(
    resume_payload,
    scenario,
    wall_dist_thresh,
    *,
    heatmap_prefix=None,
    heatmap_file=None,
    sparse_graph_threshold=2.0,
    sparse_min_component_size=15,
):
    cfg = resume_payload.get("config", {})
    existing_scenario = cfg.get("scenario")
    existing_wall_thresh = cfg.get("wall_dist_thresh")
    existing_policy = cfg.get("failure_policy")
    existing_source = cfg.get("path_metric_source")
    requested_modified = _modified_solver_config(
        heatmap_prefix, heatmap_file, sparse_graph_threshold, sparse_min_component_size
    )
    existing_modified_solver = cfg.get("modified_solver", "rightness_penalty")
    if existing_modified_solver != requested_modified["modified_solver"]:
        raise ValueError(
            f"Resume modified_solver mismatch. Existing='{existing_modified_solver}', "
            f"requested='{requested_modified['modified_solver']}'."
        )
    if requested_modified["modified_solver"] == "heatmap_graph":
        for key in ("heatmap_prefix", "heatmap_file", "sparse_graph_threshold", "sparse_min_component_size"):
            if cfg.get(key) != requested_modified.get(key):
                raise ValueError(
                    f"Resume heatmap config mismatch for {key}. "
                    f"Existing={cfg.get(key)!r}, requested={requested_modified.get(key)!r}."
                )

    if existing_scenario != scenario:
        raise ValueError(
            f"Resume scenario mismatch. Existing='{existing_scenario}', requested='{scenario}'."
        )
    if existing_policy not in (FAILURE_POLICY, "resample_until_both_succeed"):
        raise ValueError(
            f"Resume file uses unsupported failure policy. Expected '{FAILURE_POLICY}'."
        )
    if existing_policy == "resample_until_both_succeed":
        raise ValueError(
            "Resume file uses legacy 2-planner failure policy "
            "('resample_until_both_succeed'). Re-run without --resume-from."
        )
    if existing_source != "raw_path":
        raise ValueError("Resume file uses unsupported path metric source. Expected 'raw_path'.")
    if existing_wall_thresh is not None and float(existing_wall_thresh) != float(wall_dist_thresh):
        raise ValueError(
            f"Resume wall threshold mismatch. Existing={existing_wall_thresh}, requested={wall_dist_thresh}."
        )

    for row in resume_payload.get("routes", []):
        missing = [mode for mode in SOLVER_MODES if mode not in row]
        if missing:
            raise ValueError(
                "Resume file is missing planner results for: "
                f"{', '.join(missing)}. Re-run without --resume-from."
            )


def save_results(output_path, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return

    if output_path.suffix.lower() == ".csv":
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "trial",
                    "x_init",
                    "x_goal",
                    "solver",
                    "path_length",
                    "right_wall_avg",
                    "right_wall_std",
                    "solve_time_sec",
                    "num_wall_samples",
                ],
            )
            writer.writeheader()
            for row in payload["routes"]:
                for solver in SOLVER_MODES:
                    metrics = row[solver]
                    writer.writerow(
                        {
                            "trial": row["trial"],
                            "x_init": row["x_init"],
                            "x_goal": row["x_goal"],
                            "solver": solver,
                            "path_length": metrics["path_length"],
                            "right_wall_avg": metrics["right_wall_avg"],
                            "right_wall_std": metrics["right_wall_std"],
                            "solve_time_sec": metrics.get("solve_time_sec", np.nan),
                            "num_wall_samples": metrics["num_wall_samples"],
                        }
                    )
        return

    raise ValueError("Unsupported output format. Use .json or .csv")


def print_summary(summary, attempts, requested_routes):
    print("\n=== Comparison Summary ===")
    print(f"Requested paired routes: {requested_routes}")
    print(f"Sampling attempts: {attempts}")
    for mode in SOLVER_MODES:
        s = summary[mode]
        print(f"\n[{mode}]")
        print(f"  routes: {s['num_routes']}")
        print(f"  path_length mean/std: {s['path_length_mean']:.4f} / {s['path_length_std']:.4f}")
        print(f"  right_wall_avg mean/std: {s['right_wall_avg_mean']:.4f} / {s['right_wall_avg_std']:.4f}")
        print(f"  right_wall_std mean/std: {s['right_wall_std_mean']:.4f} / {s['right_wall_std_std']:.4f}")
        print(f"  solve_time_sec mean/std: {s['solve_time_mean_sec']:.4f} / {s['solve_time_std_sec']:.4f}")


def plot_debug_pair(
    occ_grid,
    trial_num,
    x_init,
    x_goal,
    paths_by_solver,
    metrics_by_solver,
    used_masks_by_solver,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    occ_grid.plot_grid(ax=ax)

    colors = {
        "vanilla": "tab:blue",
        "modified": "tab:orange",
        "rrt_vanilla": "tab:green",
    }

    def draw_segments(path, used_mask, base_color):
        for k in range(len(path) - 1):
            x0, y0 = path[k]
            x1, y1 = path[k + 1]
            used = used_mask[k] if k < len(used_mask) else False
            if used:
                ax.plot([x0, x1], [y0, y1], color=base_color, linewidth=2.2, alpha=0.95)
            else:
                ax.plot([x0, x1], [y0, y1], color=base_color, linewidth=1.2, alpha=0.25, linestyle="--")

    handles = []
    for solver in SOLVER_MODES:
        draw_segments(
            paths_by_solver[solver],
            used_masks_by_solver[solver],
            colors[solver],
        )
        avg = metrics_by_solver[solver]["right_wall_avg"]
        avg_label = f"{avg:.3f}" if not np.isnan(avg) else "nan"
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[solver],
                linewidth=2.2,
                label=f"{solver} used (avg right dist={avg_label})",
            )
        )

    ax.scatter(x_init[0], x_init[1], c="green", s=40, zorder=5, label="start")
    ax.scatter(x_goal[0], x_goal[1], c="gold", marker="*", s=70, zorder=5, label="goal")
    ax.set_title(f"Trial {trial_num}: Three-planner overlay")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_sample_paths(occ_grid, sample_pairs, plot_output=None, modified_title=None):
    if not sample_pairs:
        print("No sample paths available to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cmap = plt.get_cmap("tab10", max(len(sample_pairs), 1))
    for j, solver in enumerate(SOLVER_MODES):
        ax = axes[j]
        occ_grid.plot_grid(ax=ax)
        for i, sample in enumerate(sample_pairs):
            path = sample[f"{solver}_path"]
            xs, ys = zip(*path)
            color = cmap(i % cmap.N)
            ax.plot(xs, ys, color=color, linewidth=2.5, alpha=1)
            ax.scatter(sample["x_init"][0], sample["x_init"][1], c=[color], s=40, zorder=5, marker="o")
            ax.scatter(sample["x_goal"][0], sample["x_goal"][1], c=[color], marker="*", s=80, zorder=5)

        title = SOLVER_PLOT_TITLES[solver]
        if solver == "modified" and modified_title:
            title = modified_title
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        handles = [
            Line2D([0], [0], marker="o", color="k", markerfacecolor="green", markersize=6, label="start"),
            Line2D([0], [0], marker="*", color="k", markerfacecolor="gold", markersize=10, label="goal"),
        ]
        ax.legend(handles=handles, loc="best", fontsize=12)

    fig.tight_layout()
    if plot_output:
        plot_path = Path(plot_output)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved sample path figure to: {plot_output}")
    else:
        plt.show()


def _rrt_kwargs_from_config(
    *,
    rrt_max_iter,
    rrt_step_size,
    rrt_goal_sample_rate,
    rrt_goal_tolerance,
    rrt_rewire_radius,
    resolution,
):
    kwargs = {
        "max_iterations": rrt_max_iter,
        "goal_sample_rate": rrt_goal_sample_rate,
    }
    if rrt_step_size is not None:
        kwargs["step_size"] = rrt_step_size
    if rrt_goal_tolerance is not None:
        kwargs["goal_tolerance"] = rrt_goal_tolerance
    if rrt_rewire_radius is not None:
        kwargs["rewire_radius"] = rrt_rewire_radius
    elif rrt_step_size is not None:
        kwargs["rewire_radius"] = 2.0 * rrt_step_size
    elif resolution is not None:
        kwargs["rewire_radius"] = 2.0 * resolution
    return kwargs


def run_experiment(
    scenario,
    num_routes,
    seed,
    output,
    max_attempts,
    wall_dist_thresh,
    resume_from=None,
    plot_samples=0,
    plot_output=None,
    debug_plot_each_run=False,
    heatmap_prefix=None,
    heatmap_file=None,
    sparse_graph_threshold=2.0,
    sparse_min_component_size=15,
    rrt_max_iter=5000,
    rrt_step_size=None,
    rrt_goal_sample_rate=0.10,
    rrt_goal_tolerance=None,
    rrt_rewire_radius=None,
):
    rng = np.random.default_rng(seed)
    occ_grid, _, resolution, statespace_hi = build_occ_grid(scenario)

    social_graph = None
    modified_config = _modified_solver_config(
        heatmap_prefix, heatmap_file, sparse_graph_threshold, sparse_min_component_size
    )
    if modified_config["modified_solver"] == "heatmap_graph":
        social_graph = build_sparse_graph_from_heatmap(
            occ_grid,
            heatmap_prefix=heatmap_prefix,
            heatmap_path=heatmap_file,
            sparse_graph_threshold=sparse_graph_threshold,
            sparse_min_component_size=sparse_min_component_size,
        )

    modified_plot_title = (
        "A* social (heatmap graph)" if social_graph is not None else SOLVER_PLOT_TITLES["modified"]
    )
    rrt_kwargs = _rrt_kwargs_from_config(
        rrt_max_iter=rrt_max_iter,
        rrt_step_size=rrt_step_size,
        rrt_goal_sample_rate=rrt_goal_sample_rate,
        rrt_goal_tolerance=rrt_goal_tolerance,
        rrt_rewire_radius=rrt_rewire_radius,
        resolution=resolution,
    )

    records = []
    prior_attempts = 0
    start_trial = 1
    seen_pairs = set()
    if resume_from:
        prior_payload = load_resume_payload(resume_from)
        validate_resume_compatibility(
            prior_payload,
            scenario,
            wall_dist_thresh,
            heatmap_prefix=heatmap_prefix,
            heatmap_file=heatmap_file,
            sparse_graph_threshold=sparse_graph_threshold,
            sparse_min_component_size=sparse_min_component_size,
        )
        records = list(prior_payload["routes"])
        prior_attempts = int(prior_payload.get("sampling_attempts", 0))
        start_trial = len(records) + 1
        for row in records:
            seen_pairs.add(route_pair_key(row["x_init"], row["x_goal"]))

    attempts = 0
    target_total_routes = len(records) + num_routes
    sample_pairs = []
    print(
        f"Collecting paired routes: {len(records)}/{target_total_routes} "
        f"(need {num_routes} new; max_attempts={max_attempts})"
    )
    while len(records) < target_total_routes:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Reached max attempts ({max_attempts}) before collecting {num_routes} new paired successful routes."
            )

        if attempts % 20 == 0:
            print(
                f"[progress] attempts={attempts}, "
                f"collected={len(records)}/{target_total_routes} paired routes"
            )

        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        if x_init == x_goal:
            continue
        pair_key = route_pair_key(x_init, x_goal)
        if pair_key in seen_pairs:
            continue

        solver_results = {}
        all_succeeded = True
        for mode in SOLVER_MODES:
            path, solve_time = run_solver(
                mode,
                occ_grid,
                statespace_hi,
                x_init,
                x_goal,
                resolution,
                social_graph=social_graph,
                rrt_kwargs=rrt_kwargs,
            )
            if mode == "rrt_vanilla":
                if path is None:
                    print(
                        f"  RRT*: no path ({solve_time:.2f}s) "
                        f"init={x_init} goal={x_goal}"
                    )
                else:
                    print(
                        f"  RRT*: found path ({solve_time:.2f}s, "
                        f"{len(path)} waypoints) init={x_init} goal={x_goal}"
                    )
            if path is None:
                all_succeeded = False
            else:
                solver_results[mode] = (path, solve_time)
        if not all_succeeded:
            continue

        trial_num = len(records) + 1
        metrics_by_solver = {}
        analysis_by_solver = {}
        for mode in SOLVER_MODES:
            path, solve_time = solver_results[mode]
            metrics, analysis = compute_metrics(
                path, occ_grid, dist_thresh=wall_dist_thresh, return_analysis=True
            )
            metrics_by_solver[mode] = {
                **metrics,
                "solve_time_sec": float(solve_time),
            }
            analysis_by_solver[mode] = analysis

        record = {
            "trial": trial_num,
            "x_init": [float(x_init[0]), float(x_init[1])],
            "x_goal": [float(x_goal[0]), float(x_goal[1])],
        }
        for mode in SOLVER_MODES:
            record[mode] = metrics_by_solver[mode]
        records.append(record)
        seen_pairs.add(pair_key)
        if len(records) % 20 == 0:
            print(
                f"[progress] collected {len(records)}/{target_total_routes} "
                f"paired routes (attempts={attempts})"
            )
        if debug_plot_each_run:
            plot_debug_pair(
                occ_grid=occ_grid,
                trial_num=trial_num,
                x_init=x_init,
                x_goal=x_goal,
                paths_by_solver={mode: solver_results[mode][0] for mode in SOLVER_MODES},
                metrics_by_solver=metrics_by_solver,
                used_masks_by_solver={
                    mode: analysis_by_solver[mode]["used_segment_mask"] for mode in SOLVER_MODES
                },
            )
        if len(sample_pairs) < plot_samples:
            sample_entry = {
                "trial": trial_num,
                "x_init": [float(x_init[0]), float(x_init[1])],
                "x_goal": [float(x_goal[0]), float(x_goal[1])],
            }
            for mode in SOLVER_MODES:
                sample_entry[f"{mode}_path"] = solver_results[mode][0]
            sample_pairs.append(sample_entry)

    summary = summarize_by_solver(records)
    print_summary(summary, attempts=prior_attempts + attempts, requested_routes=len(records))
    if resume_from:
        print(
            f"Resumed from {len(records) - num_routes} routes and added {num_routes} new paired successful routes "
            f"(trial {start_trial} to {len(records)})."
        )

    payload = {
        "config": {
            "scenario": scenario,
            "num_routes": len(records),
            "seed": seed,
            "max_attempts": max_attempts,
            "wall_dist_thresh": wall_dist_thresh,
            "wall_metric_exclusion_radius": RIGHT_WALL_EXCLUSION_RADIUS,
            "failure_policy": FAILURE_POLICY,
            "path_metric_source": "raw_path",
            "resumed_from": resume_from,
            "rrt_max_iter": rrt_max_iter,
            "rrt_step_size": rrt_step_size,
            "rrt_goal_sample_rate": rrt_goal_sample_rate,
            "rrt_goal_tolerance": rrt_goal_tolerance,
            "rrt_rewire_radius": rrt_rewire_radius,
            **modified_config,
        },
        "sampling_attempts": prior_attempts + attempts,
        "summary": summary,
        "routes": records,
    }
    if output:
        save_results(output, payload)
        print(f"\nSaved results to: {output}")
    elif resume_from:
        save_results(resume_from, payload)
        print(f"\nSaved resumed results to: {resume_from}")

    if plot_samples > 0:
        plot_sample_paths(
            occ_grid,
            sample_pairs,
            plot_output=plot_output,
            modified_title=modified_plot_title,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare vanilla A*, social A*, and vanilla RRT* on paired random "
            "start/goal routes. By default social A* uses rightness-penalty costs; pass "
            "--heatmap-prefix or --heatmap-file to use graph-based social costs."
        )
    )
    parser.add_argument("--scenario", default="sample2_default", help="Grid scenario name from "
                                                                      "environments/grid_scenarios.json")
    parser.add_argument(
        "--num-routes",
        type=int,
        required=True,
        help="Number of new paired successful routes to collect",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path (.json or .csv)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum random pair attempts before aborting (default: 100 * num_routes)",
    )
    parser.add_argument(
        "--wall-dist-thresh",
        type=float,
        default=10.0,
        help="Maximum right-wall search distance in meters",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a previous JSON output from compare_astar.py to continue from",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=0,
        help="Number of new sample route sets to plot in a 1x3 three-planner figure",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Optional output image path for sample plot (e.g. samples.png); if omitted, shows figure",
    )
    parser.add_argument(
        "--debug-plot-each-run",
        action="store_true",
        help="Show an interactive per-trial overlay plot for all three planners",
    )
    rrt_group = parser.add_argument_group("RRT* parameters")
    rrt_group.add_argument("--rrt-max-iter", type=int, default=10000, help="Max RRT* iterations "
                                                                           "per solve")
    rrt_group.add_argument("--rrt-step-size", type=float, default=1, help="RRT* steer step size ("
                                                                          "default: map resolution)")
    rrt_group.add_argument("--rrt-goal-sample-rate", type=float, default=0.10, help="Probability of sampling the goal")
    rrt_group.add_argument("--rrt-goal-tolerance", type=float, default=None, help="Goal connection tolerance (default: resolution)")
    rrt_group.add_argument("--rrt-rewire-radius", type=float, default=None, help="RRT* rewiring radius (default: 2 * step size)")
    heatmap_group = parser.add_argument_group("heatmap graph (optional modified solver)")
    heatmap_group.add_argument(
        "--heatmap-prefix",
        type=str,
        default=None,
        help="Prefix P such that P_heatmap.npy exists; builds sparse graph for modified A*.",
    )
    heatmap_group.add_argument(
        "--heatmap-file",
        type=str,
        default=None,
        help="Path to directional heatmap .npy (overrides --heatmap-prefix).",
    )
    heatmap_group.add_argument(
        "--sparse-graph-threshold",
        type=float,
        default=2.0,
        help="Minimum directional heat for a sparse-graph edge (default: 2.0).",
    )
    heatmap_group.add_argument(
        "--sparse-min-component-size",
        type=int,
        default=15,
        help="Prune graph components with fewer than this many nodes (default: 15).",
    )
    args = parser.parse_args()
    if args.heatmap_prefix and args.heatmap_file:
        parser.error("Use only one of --heatmap-prefix or --heatmap-file.")
    if args.num_routes <= 0:
        raise ValueError("--num-routes must be > 0")
    if args.max_attempts is None:
        args.max_attempts = max(100, 100 * args.num_routes)
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be > 0")
    if args.plot_samples < 0:
        raise ValueError("--plot-samples must be >= 0")
    return args


if __name__ == "__main__":
    cli = parse_args()
    run_experiment(
        scenario=cli.scenario,
        num_routes=cli.num_routes,
        seed=cli.seed,
        output=cli.output,
        max_attempts=cli.max_attempts,
        wall_dist_thresh=cli.wall_dist_thresh,
        resume_from=cli.resume_from,
        plot_samples=cli.plot_samples,
        plot_output=cli.plot_output,
        debug_plot_each_run=cli.debug_plot_each_run,
        heatmap_prefix=cli.heatmap_prefix,
        heatmap_file=cli.heatmap_file,
        sparse_graph_threshold=cli.sparse_graph_threshold,
        sparse_min_component_size=cli.sparse_min_component_size,
        rrt_max_iter=cli.rrt_max_iter,
        rrt_step_size=cli.rrt_step_size,
        rrt_goal_sample_rate=cli.rrt_goal_sample_rate,
        rrt_goal_tolerance=cli.rrt_goal_tolerance,
        rrt_rewire_radius=cli.rrt_rewire_radius,
    )
