"""
Random-route smoke test: social (modified) A* on the Y2E2 occupancy map.

Loads the `y2e2` scenario from `environments/grid_scenarios.json` (y2e2.yaml + PGM).
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from social_path_planning.a_star import AStar
from social_path_planning.compare_astar import build_occ_grid, generate_random_free_point

SCENARIO = "y2e2"
_PATH_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run social (modified) A* on random free start/goal pairs on the Y2E2 map."
    )
    parser.add_argument(
        "-n",
        "--num-paths",
        type=int,
        default=5,
        metavar="N",
        help="Number of successful social paths to collect (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible start/goal sampling (default: 42).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max planner trials (default: max(100, 50 * num_paths)).",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=0.0,
        metavar="M",
        help="Minimum Euclidean distance between start and goal (meters); 0 disables.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show the matplotlib figure (default: plot all paths on one map).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_paths < 1:
        print("error: --num-paths must be >= 1", file=sys.stderr)
        return 2

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(100, 50 * args.num_paths)
    if max_attempts < 1:
        print("error: --max-attempts must be >= 1", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)

    occ_grid, map_size, map_resolution, statespace_hi = build_occ_grid(SCENARIO)

    successes = []
    solve_times = []
    attempts = 0
    failures = 0

    while len(successes) < args.num_paths and attempts < max_attempts:
        attempts += 1
        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        while np.linalg.norm(np.array(x_goal) - np.array(x_init)) > 30:
            x_goal = generate_random_free_point(occ_grid, rng)
        if x_init == x_goal:
            continue
        if args.min_separation > 0.0:
            if np.linalg.norm(np.array(x_goal) - np.array(x_init)) < args.min_separation:
                continue

        planner = AStar(
            [0, 0],
            statespace_hi,
            x_init,
            x_goal,
            occ_grid,
            resolution=map_resolution,
            desired_dist_right_extra=0.25,
        )
        ok, elapsed = planner.solve(mode="modified", return_timing=True)
        if not ok:
            failures += 1
            continue

        solve_times.append(float(elapsed))
        successes.append(
            {
                "trial": len(successes) + 1,
                "x_init": x_init,
                "x_goal": x_goal,
                "path": planner.path,
                "solve_sec": float(elapsed),
            }
        )

    print(f"Scenario: {SCENARIO}")
    print(f"RNG seed: {args.seed}")
    print(f"Collected {len(successes)} / {args.num_paths} successful social paths.")
    print(f"Planner attempts: {attempts} (max {max_attempts}), failures (no path / timeout): {failures}")
    if solve_times:
        print(f"Solve time mean: {np.mean(solve_times):.3f}s, std: {np.std(solve_times):.3f}s")

    if not args.no_plot and successes:
        fig, ax = plt.subplots(figsize=(10, 10))
        occ_grid.plot_grid(ax=ax)
        for i, rec in enumerate(successes):
            color = _PATH_COLORS[i % len(_PATH_COLORS)]
            path = rec["path"]
            xs, ys = zip(*path)
            ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.9, label=f"Trial {rec['trial']}")
            ax.scatter(
                rec["x_init"][0],
                rec["x_init"][1],
                color=color,
                s=55,
                marker="o",
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
            )
            ax.scatter(
                rec["x_goal"][0],
                rec["x_goal"][1],
                color=color,
                s=120,
                marker="*",
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
            )
        ax.set_title("Y2E2 — social A* (random start/goal)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        plt.show()
    elif not args.no_plot and not successes:
        print("No paths to plot.")

    if len(successes) < args.num_paths:
        print(
            f"error: only {len(successes)} successes before hitting --max-attempts ({max_attempts}).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
