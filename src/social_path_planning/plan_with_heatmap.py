"""
Plan a single start→goal path using a saved directional heatmap (sparse graph).

The heatmap must match the scenario used when it was built: same occupancy grid
shape as ``<prefix>_heatmap.npy`` from ``HeatMap2DVector`` / ``heat_map.py``.

Example::

    set PYTHONPATH=src
    python -m social_path_planning.plan_with_heatmap --scenario y2e2 \\
        --heatmap-prefix path/to/y2e2_routes --start 10 20 --goal 40 35

Loads ``{prefix}_heatmap.npy``, builds ``FrequentSubgraph``, runs
``AStar_With_Graph`` in ``modified`` mode, then plots map + heatmap + start/goal
(and the path when the planner succeeds).
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib.pyplot as plt

from social_path_planning.a_star import AStar_With_Graph, _telemetry_json_safe
from social_path_planning.compare_astar import build_occ_grid
from social_path_planning.sparse_graph import FrequentSubgraph
from social_path_planning.utils import snap_to_grid


def parse_args():
    p = argparse.ArgumentParser(
        description="Load a directional heatmap, build sparse graph, plan once, plot."
    )
    p.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name in environments/grid_scenarios.json (must match heatmap).",
    )
    p.add_argument(
        "--heatmap-prefix",
        type=str,
        required=True,
        help="Prefix P such that P_heatmap.npy exists (same as heat_map.py --heatmap-prefix).",
    )
    p.add_argument(
        "--start",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        required=True,
        help="Start position in world meters (snapped to grid resolution).",
    )
    p.add_argument(
        "--goal",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        required=True,
        help="Goal position in world meters (snapped to grid resolution).",
    )
    p.add_argument(
        "--sparse-graph-threshold",
        type=float,
        default=2.0,
        help="Minimum directional heat for a sparse-graph edge (default: 1.0).",
    )
    p.add_argument(
        "--sparse-min-component-size",
        type=int,
        default=15,
        help="Prune graph components with fewer than this many nodes (default: 15).",
    )
    p.add_argument(
        "--min-heatmap-intensity",
        type=float,
        default=5.0,
        help="HSV overlay hides cells with total heat below this (default: 5.0).",
    )
    p.add_argument(
        "--no-heatmap-layer",
        action="store_true",
        help="Plot only occupancy + path (no directional heat overlay).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib window (still prints success/failure).",
    )
    p.add_argument(
        "--save-fig",
        type=str,
        default=None,
        metavar="PATH",
        help="If set, save the figure to this path (png/pdf/svg).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    occ_grid, _map_size, map_resolution, statespace_hi = build_occ_grid(args.scenario)

    frequent = FrequentSubgraph(occ_grid, heat_map_filename=args.heatmap_prefix)
    hm = frequent.heat_map_object
    expected = (hm.height, hm.width, 8)
    if hm.heatmap.shape != expected:
        print(
            f"error: loaded heatmap shape {hm.heatmap.shape} != expected {expected} "
            f"(height, width, 8). Wrong scenario or not a HeatMap2DVector save.",
            file=sys.stderr,
        )
        return 2

    frequent.build_graph(threshold=args.sparse_graph_threshold, reset_graph=True)
    frequent.prune_graph(min_component_size=args.sparse_min_component_size)

    x_init = snap_to_grid(tuple(args.start), map_resolution)
    x_goal = snap_to_grid(tuple(args.goal), map_resolution)

    problem = AStar_With_Graph(
        [0, 0],
        statespace_hi,
        x_init,
        x_goal,
        occ_grid,
        frequent.graph,
        resolution=map_resolution,
        desired_dist_right_extra=0.25,
    )
    ok, _elapsed, telemetry = problem.solve(
        mode="modified",
        return_timing=True,
        return_telemetry=True,
        log_telemetry=False,
    )
    path = problem.path if ok and problem.path else None

    print("A* solve telemetry:")
    print(json.dumps(_telemetry_json_safe(telemetry), indent=2, sort_keys=True))

    if path:
        print(f"Path found with {len(path)} waypoints.")
    else:
        print(
            "Planner did not find a path (try different start/goal or graph parameters).",
            file=sys.stderr,
        )

    want_fig = (not args.no_plot) or bool(args.save_fig)
    if want_fig:
        fig, ax = plt.subplots(figsize=(10, 10))
        if args.no_heatmap_layer:
            occ_grid.plot_grid(ax=ax)
        else:
            hm.plot_heatmap(
                ax=ax,
                show=False,
                min_visible_intensity=args.min_heatmap_intensity,
                add_legend=True,
            )

        if path:
            xs, ys = zip(*path)
            ax.plot(xs, ys, color="cyan", linewidth=2.5, zorder=8, label="Planned path")
        ax.scatter(
            x_init[0],
            x_init[1],
            c="lime",
            s=120,
            edgecolors="black",
            linewidths=0.8,
            zorder=9,
            label="Start",
        )
        ax.scatter(
            x_goal[0],
            x_goal[1],
            c="gold",
            marker="*",
            s=220,
            edgecolors="black",
            linewidths=0.8,
            zorder=9,
            label="Goal",
        )
        status = "path found" if path else "no path found"
        ax.set_title(
            f"{args.scenario} — A* with graph from heatmap ({args.heatmap_prefix}) — {status}"
        )
        ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
        fig.tight_layout()
        if args.save_fig:
            fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {args.save_fig}")
        if not args.no_plot:
            plt.show()

    return 0 if path else 1


if __name__ == "__main__":
    sys.exit(main())
