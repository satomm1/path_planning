"""
Route-consistency (Shannon entropy) metric: per-cell directional entropy.

Compares how consistently independently-planned paths share the same exit
direction through occupied corridor cells, across up to four planners:

    vanilla           -- plain A* (Euclidean cost only)
    rrt_vanilla        -- RRT*
    modified_nograph   -- social A* cost function in isolation (rightness_penalty,
                            ``AStar.solve(mode="modified")`` with no social graph)
    modified_graph      -- social A* with graph-biasing
                            (``AStar_With_Graph.solve(mode="modified")``)

The two social conditions were already a runtime toggle in ``a_star.py`` /
``compare_astar.run_solver`` (pass ``social_graph=None`` vs. a built
``FrequentSubgraph.graph``); this module can ask for both, label them
distinctly, and reuse/extend an existing ``compare_path_overlap`` path bank
so planners are compared on identical start/goal pairs. Pass ``--analysis-only``
to score only polylines already cached in the bank (no replan).

Metric definition
------------------
For each occupancy-grid cell that at least ``min_visiting_paths`` planned paths
actually traverse:

1. Arc-length-resample every path to the map resolution
   (``compare_path_overlap.resample_polyline_by_spacing``) so coarse polylines
   (e.g. RRT*'s ~1 m steps) still contribute cells along their geometry.
2. Convert each path to a deduped ``(col, row)`` cell sequence.
3. For every consecutive cell pair, map the step to one of 8 exit directions
   using the same indexing as ``heat_map.HeatMap2DVector`` (NW…SE).
4. Shannon entropy (bits) of the 8-bin exit-direction distribution among
   visitors is that cell's entropy for the planner. Normalize by
   ``log2(8) = 3``. Low entropy: visitors leave the cell the same way.
   High entropy: exit directions are scattered.

The reported summary is the mean cell entropy per planner per environment
(plus a visit-count-weighted mean).

Social-graph junctions / detour-ratio OD filters are not used for scoring.
A graph snapshot or heatmap is only required when planning the missing
graph-biased social condition (not needed for ``--analysis-only``).

CLI examples
------------

    python -m social_path_planning.route_entropy \\
        --analysis-only --path-bank results/env1_overlap_path_bank.json \\
        --scenario sample2_default --environment-label env1

    python -m social_path_planning.route_entropy \\
        --path-bank results/env1_overlap_path_bank.json --scenario sample2_default \\
        --environment-label env1 --graph-snapshot results/grid2_timeline/graph_0007.pkl
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from social_path_planning.benchmark_sparse_snapshots import (
    astar_path_cache_meta,
    merge_astar_paths_into_path_bank,
)
from social_path_planning.compare_astar import (
    build_occ_grid,
    build_sparse_graph_from_heatmap,
    run_solver,
)
from social_path_planning.compare_path_overlap import resample_polyline_by_spacing

PLANNERS = ("vanilla", "rrt_vanilla", "modified_nograph", "modified_graph")
PLANNER_LABELS = {
    "vanilla": "A*",
    "rrt_vanilla": "RRT*",
    "modified": "Social A*",
    "modified_nograph": "Social A* (no graph)",
    "modified_graph": "Social A* (graph-biased)",
}
DEFAULT_MIN_VISITING_PATHS = 3
DEFAULT_SPARSE_GRAPH_THRESHOLD = 5.0
DEFAULT_SPARSE_MIN_COMPONENT_SIZE = 15
DEFAULT_MASTER_CSV = "results/route_entropy/route_entropy_summary.csv"
NUM_DIRECTIONS = 8
H_MAX_BITS = float(np.log2(NUM_DIRECTIONS))  # 3.0

# Same indexing as heat_map.HeatMap2DVector: 0 NW … 7 SE
DIRECTION_MAP = {
    (-1, 1): 0,  # NW
    (0, 1): 1,   # N
    (1, 1): 2,   # NE
    (1, 0): 3,   # E
    (-1, 0): 4,  # W
    (-1, -1): 5, # SW
    (0, -1): 6,  # S
    (1, -1): 7,  # SE
}

MASTER_CSV_FIELDNAMES = [
    "environment",
    "scenario",
    "planner",
    "pool_size",
    "num_cells_qualified",
    "min_visiting_paths",
    "mean_cell_entropy_bits",
    "std_cell_entropy_bits",
    "mean_cell_entropy_bits_visit_weighted",
    "mean_cell_entropy_normalized",
    "mean_cell_entropy_normalized_visit_weighted",
]


# --------------------------------------------------------------------------
# Graph loading (only needed when planning the graph-biased social condition)
# --------------------------------------------------------------------------

def load_social_graph(
    occ_grid,
    *,
    graph_snapshot=None,
    heatmap_file=None,
    sparse_graph_threshold=DEFAULT_SPARSE_GRAPH_THRESHOLD,
    sparse_min_component_size=DEFAULT_SPARSE_MIN_COMPONENT_SIZE,
):
    """Load a pruned ``nx.DiGraph`` either from a saved snapshot or by rebuilding from a heatmap."""
    if graph_snapshot:
        with open(graph_snapshot, "rb") as f:
            graph = pickle.load(f)
        print(f"Loaded social graph snapshot from {graph_snapshot} "
              f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
        return graph
    if not heatmap_file:
        raise ValueError("Provide --graph-snapshot or --heatmap-file to obtain the social graph.")
    frequent = build_sparse_graph_from_heatmap(
        occ_grid,
        heatmap_path=heatmap_file,
        sparse_graph_threshold=sparse_graph_threshold,
        sparse_min_component_size=sparse_min_component_size,
    )
    graph = frequent.graph
    print(f"Built social graph from {heatmap_file} "
          f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
    return graph


# --------------------------------------------------------------------------
# Path bank: locate or fill in the two social conditions
# --------------------------------------------------------------------------

def _social_variant_key_in_bank(payload, routes, solver_label):
    """Return the ``paths_by_mode`` key already caching ``solver_label`` ('rightness_penalty'
    or 'heatmap_graph') for every route, preferring a disambiguated key over the legacy
    single ``"modified"`` key used by ``compare_path_overlap`` banks."""
    meta = payload.get("paths_by_mode_meta") or {}
    disamb_key = "modified_nograph" if solver_label == "rightness_penalty" else "modified_graph"

    if meta.get(disamb_key, {}).get("solver") == solver_label and all(
        disamb_key in (r.get("paths_by_mode") or {}) for r in routes
    ):
        return disamb_key

    legacy = meta.get("modified")
    if legacy and legacy.get("solver") == solver_label and all(
        "modified" in (r.get("paths_by_mode") or {}) for r in routes
    ):
        return "modified"

    return None


def _plan_and_merge_social_variant(
    *, path_bank_path, routes, occ_grid, statespace_hi, resolution, social_graph,
    mode_key, cache_meta, chunk_size=10,
):
    """Plan ``mode="modified"`` (with/without ``social_graph``) for every route in ``routes``
    still missing ``mode_key``, merging into the bank every ``chunk_size`` solves (including a
    final partial chunk) so an interrupted run resumes instead of losing all progress.

    A route where ``AStar[_With_Graph].solve(mode="modified")`` fails to find a path within its
    internal time budget (rare, but the graph-biased search can occasionally struggle where the
    social graph is sparse) is skipped -- not fatal -- and its id is returned so the caller can
    drop it from every planner's pool uniformly, keeping the comparison apples-to-apples.
    """
    todo = [r for r in routes if mode_key not in (r.get("paths_by_mode") or {})]
    failed_route_ids = set()
    if not todo:
        return failed_route_ids
    condition = "graph-biased" if social_graph is not None else "no-graph"
    print(f"Planning {condition} condition for {len(todo)}/{len(routes)} routes "
          f"(mode_key={mode_key!r})...")
    pending = []
    for i, route in enumerate(todo, 1):
        x_init = tuple(float(v) for v in route["x_init"])
        x_goal = tuple(float(v) for v in route["x_goal"])
        path, _solve_time = run_solver(
            "modified", occ_grid, statespace_hi, x_init, x_goal, resolution,
            social_graph=social_graph,
        )
        if path is None or len(path) < 2:
            print(
                f"  WARNING: route {route.get('route_id')} ({x_init} -> {x_goal}) failed to "
                f"solve for the {condition} condition; excluding it from the entropy pool.",
                flush=True,
            )
            failed_route_ids.add(int(route["route_id"]))
        else:
            pending.append({
                "route_id": int(route["route_id"]),
                "x_init": list(x_init),
                "x_goal": list(x_goal),
                "path": [(float(p[0]), float(p[1])) for p in path],
            })
        if pending and (len(pending) >= chunk_size or i == len(todo)):
            merge_astar_paths_into_path_bank(path_bank_path, pending, mode_key, cache_meta)
            print(f"  planned+saved {condition} condition: {i}/{len(todo)} (bank updated)", flush=True)
            pending = []
    return failed_route_ids


def ensure_social_variants(
    *,
    path_bank_path,
    payload,
    routes,
    occ_grid,
    statespace_hi,
    resolution,
    social_graph,
    heatmap_file,
    graph_snapshot,
    sparse_graph_threshold,
    sparse_min_component_size,
):
    """Make sure both social conditions are cached in the bank; plan+merge whichever is missing.

    Returns (nograph_key, graph_key, routes, dropped_route_ids). ``routes`` excludes any route
    that failed to solve for either social condition; ``routes`` is reloaded from disk if either
    condition had to be planned.
    """
    nograph_key = _social_variant_key_in_bank(payload, routes, "rightness_penalty")
    graph_key = _social_variant_key_in_bank(payload, routes, "heatmap_graph")
    needs_reload = False
    dropped_route_ids = set()

    if nograph_key is None:
        cache_meta = astar_path_cache_meta("modified_nograph", None)
        dropped_route_ids |= _plan_and_merge_social_variant(
            path_bank_path=path_bank_path, routes=routes,
            occ_grid=occ_grid, statespace_hi=statespace_hi, resolution=resolution,
            social_graph=None, mode_key="modified_nograph", cache_meta=cache_meta,
        )
        nograph_key = "modified_nograph"
        needs_reload = True

    if graph_key is None:
        if social_graph is None:
            raise ValueError(
                "Planning the graph-biased social condition requires --graph-snapshot "
                "or --heatmap-file."
            )
        cache_meta = astar_path_cache_meta(
            "modified_graph",
            social_graph,
            heatmap_file=heatmap_file or graph_snapshot,
            sparse_graph_threshold=sparse_graph_threshold,
            sparse_min_component_size=sparse_min_component_size,
        )
        dropped_route_ids |= _plan_and_merge_social_variant(
            path_bank_path=path_bank_path, routes=routes,
            occ_grid=occ_grid, statespace_hi=statespace_hi, resolution=resolution,
            social_graph=social_graph, mode_key="modified_graph", cache_meta=cache_meta,
        )
        graph_key = "modified_graph"
        needs_reload = True

    if needs_reload:
        with Path(path_bank_path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        by_id = {int(r["route_id"]): r for r in payload.get("routes") or []}
        routes = [by_id[int(r["route_id"])] for r in routes]

    if dropped_route_ids:
        routes = [r for r in routes if int(r["route_id"]) not in dropped_route_ids]

    return nograph_key, graph_key, routes, dropped_route_ids


def _mode_present_on_all_routes(routes, mode_key):
    return all(mode_key in (r.get("paths_by_mode") or {}) for r in routes)


def resolve_cached_planner_path_getters(payload, routes):
    """Build planner->path getters from whatever polylines the bank already caches.

    Never plans. Social with/without graph are not forced apart: whichever of
    ``modified_nograph`` / ``modified_graph`` / legacy ``modified`` is present is
    scored under that name (legacy ``modified`` keeps the ambiguous label when
    meta does not identify the solver, or is remapped to ``modified_nograph`` /
    ``modified_graph`` when ``paths_by_mode_meta`` says which it was).
    """
    if not routes:
        raise ValueError("No routes to score.")
    if not all(r.get("path") for r in routes):
        raise ValueError("Analysis-only mode requires a vanilla ``path`` on every route.")

    getters = {"vanilla": lambda r: r["path"]}
    planners = ["vanilla"]

    if _mode_present_on_all_routes(routes, "rrt_vanilla"):
        getters["rrt_vanilla"] = lambda r: r["paths_by_mode"]["rrt_vanilla"]
        planners.append("rrt_vanilla")

    nograph_key = _social_variant_key_in_bank(payload, routes, "rightness_penalty")
    graph_key = _social_variant_key_in_bank(payload, routes, "heatmap_graph")

    if nograph_key is not None:
        getters["modified_nograph"] = (
            lambda r, k=nograph_key: r["paths_by_mode"][k]
        )
        planners.append("modified_nograph")
    if graph_key is not None and graph_key != nograph_key:
        getters["modified_graph"] = (
            lambda r, k=graph_key: r["paths_by_mode"][k]
        )
        planners.append("modified_graph")

    # Legacy bank with ``modified`` but missing / unrecognized paths_by_mode_meta.
    if (
        nograph_key is None
        and graph_key is None
        and _mode_present_on_all_routes(routes, "modified")
    ):
        getters["modified"] = lambda r: r["paths_by_mode"]["modified"]
        planners.append("modified")

    if len(planners) < 2:
        raise ValueError(
            "Analysis-only mode found only "
            f"{planners!r} in the path bank; need at least one non-vanilla "
            "cached mode (rrt_vanilla and/or modified*) to compare."
        )
    return planners, getters


# --------------------------------------------------------------------------
# Entropy computation
# --------------------------------------------------------------------------

def _path_to_cell_sequence(path, occ_grid):
    """World-space polyline -> deduped sequence of (col, row) grid-index nodes
    on the occupancy lattice (matches ``AStar.get_index``)."""
    resolution = occ_grid.resolution
    seq = []
    for x, y in path:
        col = int(round((float(x) - occ_grid.origin_x) / resolution))
        row = int(round((float(y) - occ_grid.origin_y) / resolution))
        if not seq or seq[-1] != (col, row):
            seq.append((col, row))
    return seq


def _node_world_xy(node, occ_grid):
    col, row = node
    return (
        occ_grid.origin_x + col * occ_grid.resolution,
        occ_grid.origin_y + row * occ_grid.resolution,
    )


def _direction_index(from_cell, to_cell):
    """8-way direction index from ``from_cell`` to ``to_cell``, or None if not a unit step."""
    dx = int(to_cell[0]) - int(from_cell[0])
    dy = int(to_cell[1]) - int(from_cell[1])
    # Collapse multi-cell jumps (after densification these should be rare) to signs.
    sx = int(np.sign(dx))
    sy = int(np.sign(dy))
    return DIRECTION_MAP.get((sx, sy))


def _shannon_entropy_bits(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return None
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _accumulate_cell_direction_counts(cell_seqs):
    """Return dict[(col, row)] -> length-8 count array of exit directions."""
    counts_by_cell = defaultdict(lambda: np.zeros(NUM_DIRECTIONS, dtype=float))
    for seq in cell_seqs:
        for i in range(len(seq) - 1):
            d = _direction_index(seq[i], seq[i + 1])
            if d is None:
                continue
            counts_by_cell[seq[i]][d] += 1.0
    return counts_by_cell


def compute_route_entropy(
    *,
    occ_grid,
    routes,
    planner_path_getters,
    min_visiting_paths=DEFAULT_MIN_VISITING_PATHS,
    resample_spacing_m=None,
):
    """Per-cell directional entropy for every planner in ``planner_path_getters``.

    ``planner_path_getters``: dict[name -> callable(route_dict) -> [[x, y], ...]].
    Every path is arc-length resampled to ``resample_spacing_m`` (default: the
    occupancy grid's resolution) before cell classification. A cell is scored
    for a planner only when at least ``min_visiting_paths`` paths leave it with
    a recognized 8-way step. Returns (per_cell_rows, num_cells_with_any_traffic).
    """
    spacing = float(occ_grid.resolution) if resample_spacing_m is None else float(resample_spacing_m)
    planners = list(planner_path_getters.keys())

    counts_by_planner = {}
    for planner, getter in planner_path_getters.items():
        cell_seqs = [
            _path_to_cell_sequence(resample_polyline_by_spacing(getter(r), spacing), occ_grid)
            for r in routes
        ]
        counts_by_planner[planner] = _accumulate_cell_direction_counts(cell_seqs)

    all_cells = set()
    for counts in counts_by_planner.values():
        all_cells.update(counts.keys())

    per_cell_rows = []
    for node in sorted(all_cells):
        visitor_totals = {
            planner: float(counts_by_planner[planner][node].sum())
            if node in counts_by_planner[planner]
            else 0.0
            for planner in planners
        }
        # Qualify the cell if any planner has enough visitors; per-planner entropy
        # is still only filled when that planner meets the threshold.
        if max(visitor_totals.values(), default=0.0) < min_visiting_paths:
            continue

        row = {
            "node": node,
            "world_xy": _node_world_xy(node, occ_grid),
            "num_visitors": {p: int(visitor_totals[p]) for p in planners},
        }
        # Representative visit count for weighting (max across planners keeps
        # heavily used corridor cells influential in the visit-weighted mean).
        row["num_visitors_max"] = int(max(visitor_totals.values()))

        for planner in planners:
            n_vis = visitor_totals[planner]
            if n_vis < min_visiting_paths:
                row[f"{planner}_entropy_bits"] = None
                row[f"{planner}_entropy_normalized"] = None
                row[f"{planner}_num_visitors"] = int(n_vis)
                continue
            counts = counts_by_planner[planner][node]
            h = _shannon_entropy_bits(counts)
            row[f"{planner}_entropy_bits"] = h
            row[f"{planner}_entropy_normalized"] = (
                (h / H_MAX_BITS) if (h is not None and H_MAX_BITS > 0) else None
            )
            row[f"{planner}_num_visitors"] = int(n_vis)
        per_cell_rows.append(row)

    return per_cell_rows, len(all_cells)


def summarize_route_entropy(per_cell_rows, planners):
    summary = {}
    for planner in planners:
        vals, weights, vals_norm, weights_norm = [], [], [], []
        for row in per_cell_rows:
            h = row.get(f"{planner}_entropy_bits")
            if h is None:
                continue
            n_vis = row.get(f"{planner}_num_visitors", row.get("num_visitors_max", 1))
            w = max(1, int(n_vis))
            vals.append(h)
            weights.append(w)
            h_norm = row.get(f"{planner}_entropy_normalized")
            if h_norm is not None:
                vals_norm.append(h_norm)
                weights_norm.append(w)
        summary[planner] = {
            "num_cells_qualified": len(vals),
            "mean_cell_entropy_bits": float(np.mean(vals)) if vals else float("nan"),
            "std_cell_entropy_bits": float(np.std(vals)) if vals else float("nan"),
            "mean_cell_entropy_bits_visit_weighted": (
                float(np.average(vals, weights=weights)) if vals else float("nan")
            ),
            "mean_cell_entropy_normalized": (
                float(np.mean(vals_norm)) if vals_norm else float("nan")
            ),
            "mean_cell_entropy_normalized_visit_weighted": (
                float(np.average(vals_norm, weights=weights_norm)) if vals_norm else float("nan")
            ),
        }
    return summary


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _summary_rows(env_label, scenario, planners, summary, *, pool_size, min_visiting_paths):
    rows = []
    for planner in planners:
        s = summary[planner]
        rows.append({
            "environment": env_label,
            "scenario": scenario,
            "planner": planner,
            "pool_size": pool_size,
            "num_cells_qualified": s["num_cells_qualified"],
            "min_visiting_paths": min_visiting_paths,
            "mean_cell_entropy_bits": s["mean_cell_entropy_bits"],
            "std_cell_entropy_bits": s["std_cell_entropy_bits"],
            "mean_cell_entropy_bits_visit_weighted": s["mean_cell_entropy_bits_visit_weighted"],
            "mean_cell_entropy_normalized": s["mean_cell_entropy_normalized"],
            "mean_cell_entropy_normalized_visit_weighted": s[
                "mean_cell_entropy_normalized_visit_weighted"
            ],
        })
    return rows


def save_summary_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved summary to: {path.resolve()}")


def upsert_master_csv(path, rows):
    path = Path(path)
    existing = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        # Drop rows from an older schema (e.g. junction-era columns) so a metric
        # migration does not crash DictWriter or leave incompatible rows around.
        existing = [
            r for r in existing
            if "mean_cell_entropy_bits" in r and r.get("mean_cell_entropy_bits") not in (None, "")
        ]

    def key(r):
        return (r["environment"], r["planner"])

    by_key = {key(r): r for r in existing}
    for row in rows:
        by_key[key(row)] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=MASTER_CSV_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        for row in by_key.values():
            writer.writerow(row)
    print(f"Updated master summary: {path.resolve()}")


def save_cell_csv(path, per_cell_rows, planners):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["node_col", "node_row", "world_x", "world_y", "num_visitors_max"]
    for planner in planners:
        fieldnames += [
            f"{planner}_num_visitors",
            f"{planner}_entropy_bits",
            f"{planner}_entropy_normalized",
        ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_cell_rows:
            out = {
                "node_col": row["node"][0],
                "node_row": row["node"][1],
                "world_x": row["world_xy"][0],
                "world_y": row["world_xy"][1],
                "num_visitors_max": row["num_visitors_max"],
            }
            for planner in planners:
                out[f"{planner}_num_visitors"] = row.get(f"{planner}_num_visitors")
                out[f"{planner}_entropy_bits"] = row.get(f"{planner}_entropy_bits")
                out[f"{planner}_entropy_normalized"] = row.get(f"{planner}_entropy_normalized")
            writer.writerow(out)
    print(f"Saved per-cell detail to: {path.resolve()}")


def plot_route_entropy_map(occ_grid, per_cell_rows, planners, output_path):
    """Optional multi-panel overlay: cells colored by directional entropy (bits)."""
    import matplotlib.pyplot as plt

    n = len(planners)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    all_vals = [
        row[f"{planner}_entropy_bits"]
        for row in per_cell_rows
        for planner in planners
        if row.get(f"{planner}_entropy_bits") is not None
    ]
    vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0.0, H_MAX_BITS)

    sc = None
    for ax, planner in zip(axes, planners):
        occ_grid.plot_grid(ax=ax)
        xs, ys, vals = [], [], []
        for row in per_cell_rows:
            h = row.get(f"{planner}_entropy_bits")
            if h is None:
                continue
            xs.append(row["world_xy"][0])
            ys.append(row["world_xy"][1])
            vals.append(h)
        sc = ax.scatter(
            xs, ys, c=vals, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
            s=18, edgecolors="none", alpha=0.85, zorder=6,
        )
        ax.set_title(PLANNER_LABELS.get(planner, planner), fontsize=12)
        ax.set_axis_off()
    for ax in axes[len(planners):]:
        ax.axis("off")

    if sc is not None:
        fig.colorbar(
            sc, ax=list(axes[: len(planners)]), label="Cell directional entropy (bits)", shrink=0.8
        )
    fig.suptitle("Route-consistency: per-cell exit-direction entropy", fontsize=14)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path.resolve()}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Route-consistency metric: per-cell Shannon entropy of exit directions "
            "across A*, RRT*, and social A* variants, reusing an existing "
            "compare_path_overlap path bank."
        )
    )
    parser.add_argument("--path-bank", required=True, help="Existing compare_path_overlap path bank JSON.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--environment-label", default=None, help="Defaults to --scenario.")
    parser.add_argument("--pool-size", type=int, default=None, help="Defaults to all routes in the bank.")
    parser.add_argument(
        "--graph-snapshot",
        default=None,
        help="Pickled nx.DiGraph; only required when planning the graph-biased social condition.",
    )
    parser.add_argument(
        "--heatmap-file",
        default=None,
        help="HeatMap2DVector .npy (alternative to --graph-snapshot when planning).",
    )
    parser.add_argument("--sparse-graph-threshold", type=float, default=DEFAULT_SPARSE_GRAPH_THRESHOLD)
    parser.add_argument("--sparse-min-component-size", type=int, default=DEFAULT_SPARSE_MIN_COMPONENT_SIZE)
    parser.add_argument(
        "--detour-ratio",
        type=float,
        default=None,
        help="Deprecated/ignored (junction detour filter removed). Accepted for old command lines.",
    )
    parser.add_argument(
        "--min-visiting-paths",
        type=int,
        default=None,
        help="Minimum visiting paths required for a cell to be scored (default: 3).",
    )
    parser.add_argument(
        "--min-plausible-paths",
        type=int,
        default=None,
        help="Deprecated alias for --min-visiting-paths.",
    )
    parser.add_argument(
        "--resample-spacing",
        type=float,
        default=None,
        help=(
            "Arc-length resampling spacing (m) before cell classification. "
            "Defaults to the occupancy grid's resolution."
        ),
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help=(
            "Do not plan missing social variants. Score only planners whose "
            "polylines are already cached in the path bank (vanilla, rrt_vanilla, "
            "and whichever social ``modified*`` key is present)."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Defaults to results/route_entropy/<environment-label>_route_entropy",
    )
    parser.add_argument(
        "--master-csv",
        default=DEFAULT_MASTER_CSV,
        help="Combined per-environment/per-planner summary, upserted across runs.",
    )
    parser.add_argument(
        "--figure",
        default=None,
        help="Optional PNG path for the per-planner cell-entropy overlay figure.",
    )
    args = parser.parse_args()

    if args.detour_ratio is not None:
        print("Warning: --detour-ratio is ignored (per-cell directional entropy has no OD filter).")

    if args.min_visiting_paths is not None and args.min_plausible_paths is not None:
        if args.min_visiting_paths != args.min_plausible_paths:
            raise ValueError("Pass only one of --min-visiting-paths / --min-plausible-paths.")
    if args.min_visiting_paths is None:
        args.min_visiting_paths = (
            args.min_plausible_paths
            if args.min_plausible_paths is not None
            else DEFAULT_MIN_VISITING_PATHS
        )

    if args.environment_label is None:
        args.environment_label = args.scenario
    if args.output_prefix is None:
        args.output_prefix = f"results/route_entropy/{args.environment_label}_route_entropy"
    return args


def main():
    args = parse_args()
    occ_grid, _map_size, resolution, statespace_hi = build_occ_grid(args.scenario)

    path_bank_path = Path(args.path_bank)
    with path_bank_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    routes = payload.get("routes") or []
    if args.pool_size is not None:
        routes = routes[: args.pool_size]
    if not routes:
        raise ValueError(f"No routes found in path bank {path_bank_path}")

    if args.analysis_only:
        planners, planner_path_getters = resolve_cached_planner_path_getters(payload, routes)
        print(
            f"Analysis-only: scoring cached planners {planners} "
            f"({len(routes)} routes); no planning."
        )
    else:
        social_graph = None
        graph_key_existing = _social_variant_key_in_bank(payload, routes, "heatmap_graph")
        if graph_key_existing is None:
            if not args.graph_snapshot and not args.heatmap_file:
                raise ValueError(
                    "Planning the graph-biased social condition requires "
                    "--graph-snapshot or --heatmap-file (or use --analysis-only)."
                )
            social_graph = load_social_graph(
                occ_grid,
                graph_snapshot=args.graph_snapshot,
                heatmap_file=args.heatmap_file,
                sparse_graph_threshold=args.sparse_graph_threshold,
                sparse_min_component_size=args.sparse_min_component_size,
            )
        elif args.graph_snapshot or args.heatmap_file:
            social_graph = load_social_graph(
                occ_grid,
                graph_snapshot=args.graph_snapshot,
                heatmap_file=args.heatmap_file,
                sparse_graph_threshold=args.sparse_graph_threshold,
                sparse_min_component_size=args.sparse_min_component_size,
            )

        nograph_key, graph_key, routes, dropped_route_ids = ensure_social_variants(
            path_bank_path=path_bank_path,
            payload=payload,
            routes=routes,
            occ_grid=occ_grid,
            statespace_hi=statespace_hi,
            resolution=resolution,
            social_graph=social_graph,
            heatmap_file=args.heatmap_file,
            graph_snapshot=args.graph_snapshot,
            sparse_graph_threshold=args.sparse_graph_threshold,
            sparse_min_component_size=args.sparse_min_component_size,
        )
        if dropped_route_ids:
            print(
                f"Dropped {len(dropped_route_ids)} route(s) that failed to solve for a social "
                f"condition from the pool (all planners): {sorted(dropped_route_ids)}"
            )

        planners = list(PLANNERS)
        planner_path_getters = {
            "vanilla": lambda r: r["path"],
            "rrt_vanilla": lambda r: r["paths_by_mode"]["rrt_vanilla"],
            "modified_nograph": lambda r, k=nograph_key: r["paths_by_mode"][k],
            "modified_graph": lambda r, k=graph_key: r["paths_by_mode"][k],
        }

    per_cell_rows, num_cells_with_traffic = compute_route_entropy(
        occ_grid=occ_grid,
        routes=routes,
        planner_path_getters=planner_path_getters,
        min_visiting_paths=args.min_visiting_paths,
        resample_spacing_m=args.resample_spacing,
    )

    summary = summarize_route_entropy(per_cell_rows, planners)

    print(f"\n=== Route-consistency entropy: {args.environment_label} ({args.scenario}) ===")
    print(
        f"Pool size: {len(routes)}  |  cells with any traffic: {num_cells_with_traffic}  |  "
        f"rows kept (>= {args.min_visiting_paths} visitors on some planner): {len(per_cell_rows)}"
    )
    for planner in planners:
        s = summary[planner]
        print(
            f"  {PLANNER_LABELS.get(planner, planner):24s} "
            f"mean={s['mean_cell_entropy_bits']:.3f} bits  "
            f"(normalized={s['mean_cell_entropy_normalized']:.3f})  "
            f"visit-weighted={s['mean_cell_entropy_bits_visit_weighted']:.3f}  "
            f"n={s['num_cells_qualified']}"
        )

    rows = _summary_rows(
        args.environment_label,
        args.scenario,
        planners,
        summary,
        pool_size=len(routes),
        min_visiting_paths=args.min_visiting_paths,
    )
    save_summary_csv(f"{args.output_prefix}_summary.csv", rows)
    save_cell_csv(f"{args.output_prefix}_by_cell.csv", per_cell_rows, planners)
    upsert_master_csv(args.master_csv, rows)

    if args.figure:
        plot_route_entropy_map(occ_grid, per_cell_rows, planners, args.figure)


if __name__ == "__main__":
    main()
