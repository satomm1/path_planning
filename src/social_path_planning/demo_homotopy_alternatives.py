"""Demo: K socially-ranked alternative paths in distinct homotopy classes.

Plans one start/goal pair with the class-aware social A* (``homotopy_astar``) and writes a
figure plus a JSON summary. The figure shows the obstacles that define the classes, each
obstacle's ray, and the alternatives ranked by cost. With ``--heatmap`` the search uses the
graph-biased cost (Eq. 4) on the social graph H built from that heatmap. The graph uses
the same threshold and pruning defaults as ``compare_astar``, and its edges are drawn faintly.

Examples::

    python -m social_path_planning.demo_homotopy_alternatives \\
        --scenario circular_default --start 2 50 --goal 98 50 --k 4 --alpha 0.5
    python -m social_path_planning.demo_homotopy_alternatives \\
        --scenario sample2_default --start 2 2 --goal 98 98 --heatmap grid2_heatmap.npy
    python -m social_path_planning.demo_homotopy_alternatives --scenario hall_with_room_default --seed 3

Use the ``path_planning`` conda env (numba) so the compiled search is used.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from social_path_planning.common import WALL_CACHE_POLICIES, build_occ_grid
from social_path_planning.homotopy import blocked_mask
from social_path_planning.homotopy_astar import HomotopyAStar, HomotopyAStar_With_Graph

DEFAULT_OUTPUT_DIR = Path("results/homotopy_alternatives")
PATH_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]


def make_planner(occ, statespace_hi, res, start, goal, *, graph=None, **kwargs):
    if graph is None:
        return HomotopyAStar([0, 0], statespace_hi, start, goal, occ, resolution=res, **kwargs)
    return HomotopyAStar_With_Graph([0, 0], statespace_hi, start, goal, occ, graph, resolution=res, **kwargs)


def random_pair(occ, rng, min_separation_m):
    from social_path_planning.common import generate_random_free_point

    while True:
        start = generate_random_free_point(occ, rng)
        goal = generate_random_free_point(occ, rng)
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) >= min_separation_m:
            return start, goal


def plot_alternatives(ax, planner, alternatives, *, graph=None, show_rays=True, title=None, merged=()):
    """Draw the planner's obstacles, rays, the ranked alternatives and (faintly) merged ones."""
    occ = planner.occupancy
    obs = planner.obstacles
    res = float(occ.resolution)
    extent = [occ.origin_x, occ.origin_x + occ.width * res, occ.origin_y, occ.origin_y + occ.height * res]
    blocked = blocked_mask(occ, planner.statespace_lo, planner.statespace_hi)
    ax.imshow(blocked, origin="lower", extent=extent, cmap="Greys", alpha=0.45, interpolation="nearest")

    if graph is not None and graph.number_of_nodes():
        nodes = np.array(list(graph.nodes()), dtype=float)
        ax.scatter(occ.origin_x + nodes[:, 0] * res, occ.origin_y + nodes[:, 1] * res,
                   s=0.6, c="#f4a261", alpha=0.5, linewidths=0, zorder=1)
        ax.scatter([], [], s=25, c="#f4a261", label=f"social graph H ({graph.number_of_nodes()} nodes)")

    reach = max(extent[1] - extent[0], extent[3] - extent[2])
    for obstacle_id in obs.ids:
        ray = obs.rays[obstacle_id]
        x0, y0 = obs.to_world(ray.origin)
        if show_rays:
            dx, dy = (ray.sign, 0) if ray.axis == 0 else (0, ray.sign)
            ax.plot([x0, x0 + dx * reach], [y0, y0 + dy * reach], ls=":", color="0.45", lw=0.8, zorder=2)
        ax.text(x0, y0, str(obstacle_id), color="black", ha="center", va="center", fontsize=10,
                weight="bold", zorder=3,
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="0.4", lw=0.6, alpha=0.85))

    for alt in merged:
        path = np.asarray(alt.smoothed_path if alt.smoothed_path else alt.path, dtype=float)
        ax.plot(path[:, 0], path[:, 1], color="0.35", lw=0.9, ls="--", alpha=0.6, zorder=3,
                label=(f"merged {format_word(alt.word)} into #{alt.merged_into + 1}  "
                       f"(overlap {100 * alt.max_overlap:.0f}%, cost {alt.cost:.1f})"))

    for rank, alt in enumerate(alternatives):
        path = np.asarray(alt.smoothed_path if alt.smoothed_path else alt.path, dtype=float)
        label = (f"#{rank + 1} {format_word(alt.word)}  len {alt.length_m:.1f} m  "
                 f"social {alt.social_penalty:.1f}  cost {alt.cost:.1f}")
        if alt.fraction_on_graph is not None:
            label += f"  on H {100 * alt.fraction_on_graph:.0f}%"
        if rank > 0:
            label += f"  overlap {100 * alt.max_overlap:.0f}%"
        ax.plot(path[:, 0], path[:, 1], color=PATH_COLORS[rank % len(PATH_COLORS)],
                lw=max(1.0, 2.6 - 0.35 * rank), zorder=4 + len(alternatives) - rank, label=label)

    ax.scatter(*planner.x_init, c="green", s=110, zorder=20, edgecolor="k", label="start")
    ax.scatter(*planner.x_goal, c="gold", marker="*", s=260, zorder=20, edgecolor="k", label="goal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title, fontsize=10)


def format_word(word):
    """Readable class label: () -> 'direct', (1, -3) -> '1+ 3-'."""
    if not word:
        return "[ ]"
    return "[" + " ".join(f"{abs(v)}{'+' if v > 0 else '-'}" for v in word) + "]"


def alternatives_to_json(alternatives):
    return [
        {
            "rank": rank + 1,
            "word": list(alt.word),
            "winding_rad": [float(v) for v in alt.winding],
            "cost": float(alt.cost),
            "length_m": float(alt.length_m),
            "social_penalty": float(alt.social_penalty),
            "fraction_on_graph": None if alt.fraction_on_graph is None else float(alt.fraction_on_graph),
            "max_overlap": float(alt.max_overlap),
            "merged_into_rank": None if alt.merged_into is None else alt.merged_into + 1,
            "path": [[float(x), float(y)] for x, y in alt.path],
        }
        for rank, alt in enumerate(alternatives)
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", default="circular_default")
    parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--seed", type=int, help="random free start/goal (ignored if --start/--goal given)")
    parser.add_argument("--min-separation", type=float, default=30.0,
                        help="minimum start-goal distance for --seed pairs [m]")
    parser.add_argument("--k", type=int, default=4, help="maximum number of classes")
    parser.add_argument("--alpha", type=float, default=0.5, help="keep classes up to (1+alpha) x best cost")
    parser.add_argument("--min-area", type=float, default=1.0, help="obstacles smaller than this [m^2] are ignored")
    parser.add_argument("--no-winding-filter", action="store_true", help="allow paths that loop around obstacles")
    parser.add_argument("--merge-overlap", type=float, default=None,
                        help="merge a class into a cheaper one it overlaps by at least this fraction "
                             "(e.g. 0.65; off by default)")
    parser.add_argument("--merge-distance", type=float, default=0.5,
                        help="proximity for the overlap test [m] (default: robot diameter, as in Fig. 11)")
    parser.add_argument("--mode", choices=["modified", "vanilla"], default="modified",
                        help="modified = social cost (paper), vanilla = distance only")
    parser.add_argument("--heatmap", help="directional heatmap .npy; plans with the graph-biased cost (Eq. 4)")
    parser.add_argument("--wall-cache", choices=WALL_CACHE_POLICIES, default="if_present")
    parser.add_argument("--time-limit", type=float, default=None, help="search time limit [s]")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", help="output file stem (default: derived from scenario and pair)")
    parser.add_argument("--no-rays", action="store_true", help="hide obstacle rays in the figure")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--show", action="store_true", help="open the figure window")
    args = parser.parse_args(argv)
    if (args.start is None) != (args.goal is None):
        parser.error("--start and --goal must be given together")
    if args.start is None and args.seed is None:
        parser.error("give --start/--goal or --seed")
    return args


def run(args):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occ, _, res, statespace_hi = build_occ_grid(args.scenario, wall_cache=args.wall_cache)
    if args.start is not None:
        start, goal = tuple(args.start), tuple(args.goal)
    else:
        start, goal = random_pair(occ, np.random.default_rng(args.seed), args.min_separation)

    graph = None
    if args.heatmap:
        from social_path_planning.compare_astar import build_sparse_graph_from_heatmap

        graph = build_sparse_graph_from_heatmap(occ, heatmap_path=args.heatmap)

    planner = make_planner(
        occ, statespace_hi, res, start, goal, graph=graph,
        k=args.k, alpha=args.alpha, min_area_m2=args.min_area,
        max_abs_winding=None if args.no_winding_filter else 2 * math.pi,
        time_limit_s=args.time_limit,
        merge_overlap=args.merge_overlap, merge_distance_m=args.merge_distance,
    )
    for name, point in (("start", planner.x_init), ("goal", planner.x_goal)):
        if not planner.is_free(point):
            raise SystemExit(f"{name} {point} is not in free space on {args.scenario}")

    t0 = time.time()
    alternatives = planner.solve_alternatives(mode=args.mode)
    elapsed = time.time() - t0
    engine = "compiled" if planner.used_compiled else "python"

    cost_label = "graph-biased social" if graph is not None else ("social" if args.mode == "modified" else "distance")
    title = (f"{args.scenario}: {len(alternatives)} classes ({cost_label} cost, K={args.k}, "
             f"alpha={args.alpha})\n{planner.obstacles.num_obstacles} obstacles, "
             f"{planner.explored_states} states, {elapsed:.1f} s ({engine})")
    # Size the axes to the map's aspect ratio; the legend goes below and is kept by
    # bbox_inches="tight" when saving.
    aspect = (occ.height * res) / (occ.width * res)
    width_in = 9.0
    fig, ax = plt.subplots(figsize=(width_in, max(3.5, width_in * aspect) + 0.8))
    plot_alternatives(ax, planner, alternatives, graph=graph, show_rays=not args.no_rays, title=title,
                      merged=planner.merged_alternatives)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=1, frameon=False,
              markerscale=1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or (
        f"{args.scenario}_{start[0]:g}_{start[1]:g}_to_{goal[0]:g}_{goal[1]:g}"
        + ("_graph" if graph is not None else "") + ("" if args.mode == "modified" else "_vanilla")
        + ("" if args.merge_overlap is None else f"_merge{args.merge_overlap:g}")
    )
    fig_path = args.output_dir / f"{stem}.png"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    summary = {
        "scenario": args.scenario,
        "start": list(planner.x_init),
        "goal": list(planner.x_goal),
        "mode": args.mode,
        "heatmap": args.heatmap,
        "k": args.k,
        "alpha": args.alpha,
        "min_area_m2": args.min_area,
        "winding_filter": not args.no_winding_filter,
        "num_obstacles": planner.obstacles.num_obstacles,
        "explored_states": planner.explored_states,
        "solve_time_s": elapsed,
        "engine": engine,
        "merge_overlap": args.merge_overlap,
        "merge_distance_m": args.merge_distance,
        "alternatives": alternatives_to_json(alternatives),
        "merged": alternatives_to_json(planner.merged_alternatives),
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=1))
    print(f"{len(alternatives)} classes in {elapsed:.2f} s; wrote {fig_path} and {json_path}")
    for rank, alt in enumerate(alternatives):
        print(f"  #{rank + 1} {format_word(alt.word):14s} cost {alt.cost:8.2f}  len {alt.length_m:7.2f} m  "
              f"social {alt.social_penalty:7.2f}  overlap {alt.max_overlap:.2f}")
    for alt in planner.merged_alternatives:
        print(f"  merged {format_word(alt.word):10s} cost {alt.cost:8.2f} into #{alt.merged_into + 1} "
              f"(overlap {alt.max_overlap:.2f})")
    if args.show:
        plt.show()
    plt.close(fig)
    return summary


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
