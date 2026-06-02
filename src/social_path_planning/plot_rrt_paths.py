"""Visualize vanilla and social RRT* paths on a grid scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from social_path_planning.compare_astar import build_occ_grid, generate_random_free_point
from social_path_planning.rrt_star import RRTStar


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan and plot vanilla vs social RRT* paths on a grid scenario."
    )
    parser.add_argument(
        "--scenario",
        default="sample2_default",
        help="Scenario name from environments/grid_scenarios.json (default: sample2_default)",
    )
    parser.add_argument(
        "-n",
        "--num-paths",
        type=int,
        default=5,
        help="Number of successful route pairs to plot (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max sampling attempts (default: max(100, 50 * num_paths))",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=0.0,
        help="Minimum start-goal distance in meters (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional PNG output path (e.g. outputs_timeline/rrt_paths.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    parser.add_argument("--rrt-max-iter", type=int, default=5000)
    parser.add_argument("--rrt-step-size", type=float, default=None)
    parser.add_argument("--rrt-goal-sample-rate", type=float, default=0.15)
    return parser.parse_args()


def _rrt_kwargs(args, resolution):
    kwargs = {
        "max_iterations": args.rrt_max_iter,
        "goal_sample_rate": args.rrt_goal_sample_rate,
    }
    if args.rrt_step_size is not None:
        kwargs["step_size"] = args.rrt_step_size
        kwargs["rewire_radius"] = 2.0 * args.rrt_step_size
    return kwargs


def _plan(mode, occ_grid, statespace_hi, x_init, x_goal, resolution, rrt_kwargs, rng):
    planner = RRTStar(
        [0, 0],
        statespace_hi,
        x_init,
        x_goal,
        occ_grid,
        resolution=resolution,
        **rrt_kwargs,
    )
    ok, elapsed = planner.solve(mode=mode, return_timing=True, rng=rng)
    return ok, elapsed, planner.path


def plot_paths(occ_grid, records, output=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ("RRT* vanilla", "RRT* social")
    modes = ("vanilla", "modified")
    cmap = plt.get_cmap("tab10", max(len(records), 1))

    for ax, title, mode in zip(axes, titles, modes):
        occ_grid.plot_grid(ax=ax)
        for i, rec in enumerate(records):
            path = rec[f"{mode}_path"]
            color = cmap(i % cmap.N)
            xs, ys = zip(*path)
            ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.95)
            ax.scatter(rec["x_init"][0], rec["x_init"][1], c=[color], s=45, marker="o", zorder=6)
            ax.scatter(rec["x_goal"][0], rec["x_goal"][1], c=[color], marker="*", s=90, zorder=6)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        handles = [
            Line2D([0], [0], marker="o", color="k", markerfacecolor="green", markersize=6, label="start"),
            Line2D([0], [0], marker="*", color="k", markerfacecolor="gold", markersize=10, label="goal"),
        ]
        ax.legend(handles=handles, loc="best", fontsize=10)

    fig.tight_layout()
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    if args.num_paths < 1:
        print("error: --num-paths must be >= 1", file=sys.stderr)
        return 2

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(100, 50 * args.num_paths)

    rng = np.random.default_rng(args.seed)
    occ_grid, _, resolution, statespace_hi = build_occ_grid(args.scenario)
    rrt_kwargs = _rrt_kwargs(args, resolution)

    records = []
    attempts = 0
    while len(records) < args.num_paths and attempts < max_attempts:
        attempts += 1
        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        if x_init == x_goal:
            continue
        if args.min_separation > 0.0:
            if np.linalg.norm(np.array(x_goal) - np.array(x_init)) < args.min_separation:
                continue

        ok_v, t_v, path_v = _plan(
            "vanilla", occ_grid, statespace_hi, x_init, x_goal, resolution, rrt_kwargs, rng
        )
        if not ok_v:
            continue
        ok_m, t_m, path_m = _plan(
            "modified", occ_grid, statespace_hi, x_init, x_goal, resolution, rrt_kwargs, rng
        )
        if not ok_m:
            continue

        records.append(
            {
                "trial": len(records) + 1,
                "x_init": x_init,
                "x_goal": x_goal,
                "vanilla_path": path_v,
                "modified_path": path_m,
                "vanilla_time_sec": t_v,
                "modified_time_sec": t_m,
            }
        )
        print(
            f"Trial {len(records)}: start={x_init} goal={x_goal} "
            f"vanilla={t_v:.2f}s modified={t_m:.2f}s"
        )

    print(f"\nScenario: {args.scenario}")
    print(f"Collected {len(records)} / {args.num_paths} paired RRT* paths in {attempts} attempts.")

    if not records:
        print("No paths to plot.", file=sys.stderr)
        return 1

    plot_paths(
        occ_grid,
        records,
        output=args.output,
        show=not args.no_show,
    )
    return 0 if len(records) >= args.num_paths else 1


if __name__ == "__main__":
    sys.exit(main())
