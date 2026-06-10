"""Compare spatial path overlap for multi-robot route sets across A*, RRT*, and Social A*."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

from social_path_planning.benchmark_sparse_snapshots import (
    _path_from_serializable,
    _path_to_serializable,
    astar_path_cache_meta,
    merge_astar_paths_into_path_bank,
)
from social_path_planning.compare_astar import (
    SOLVER_MODES,
    SOLVER_PLOT_TITLES,
    build_occ_grid,
    build_sparse_graph_from_heatmap,
    generate_random_free_point,
    parse_solver_modes,
    run_solver,
)
from social_path_planning.compare_astar import _rrt_kwargs_from_config as rrt_kwargs_from_config
from social_path_planning.mapf_comparison.ensemble import sample_route_subset
from social_path_planning.mapf_comparison.solution_metrics import dedupe_world_path

FAILURE_POLICY = "all_solvers_same_endpoints"
BANK_PLANNING_MODES = SOLVER_MODES
NON_VANILLA_MODES = ("modified", "rrt_vanilla")
ROBOT_DIAMETER = 0.5
DEFAULT_SAME_DIRECTION_ANGLE_TOL_DEG = 1.0


def _same_direction_cos_min(angle_tol_deg):
    return float(np.cos(np.radians(float(angle_tol_deg))))


def resample_polyline_by_spacing(path, spacing_m):
    """Insert points along arc length every ``spacing_m`` meters; always keeps endpoints."""
    if spacing_m is None or spacing_m <= 0:
        return dedupe_world_path([(float(p[0]), float(p[1])) for p in path])
    if not path:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]

    spacing_m = float(spacing_m)
    result = [(float(path[0][0]), float(path[0][1]))]
    dist_along = 0.0
    next_sample = spacing_m

    for i in range(len(path) - 1):
        p0 = np.array(path[i], dtype=float)
        p1 = np.array(path[i + 1], dtype=float)
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-12:
            continue
        u = seg / seg_len
        seg_start = dist_along
        dist_along += seg_len
        while next_sample <= dist_along + 1e-9:
            local = next_sample - seg_start
            pt = p0 + u * local
            result.append((float(pt[0]), float(pt[1])))
            next_sample += spacing_m

    end = (float(path[-1][0]), float(path[-1][1]))
    if result[-1] != end:
        result.append(end)
    return dedupe_world_path(result)


def normalize_path_for_overlap(path, spacing_m):
    """Arc-length resample all polylines before overlap scoring (no grid snap)."""
    return resample_polyline_by_spacing(path, spacing_m)


def _unique_overlap_segments_for_pair(path_idx_a, path_idx_b, seg_i, seg_j):
    """Each (path, segment index) counts at most once per agent pair."""
    unique = set()
    for si, sj in zip(seg_i, seg_j):
        unique.add((path_idx_a, int(si)))
        unique.add((path_idx_b, int(sj)))
    return unique


def _conflicting_segment_indices(
    path_a,
    path_b,
    threshold=ROBOT_DIAMETER,
    *,
    same_direction_cos_min=None,
):
    """Vectorized segment-endpoint conflict detection (same rule as multi_planning)."""
    empty = {"seg_i": [], "seg_j": [], "seg_i_excl_same_dir": [], "seg_j_excl_same_dir": []}
    if len(path_a) < 2 or len(path_b) < 2:
        return empty

    a0 = np.asarray(path_a[:-1], dtype=float)
    a1 = np.asarray(path_a[1:], dtype=float)
    b0 = np.asarray(path_b[:-1], dtype=float)
    b1 = np.asarray(path_b[1:], dtype=float)
    t = float(threshold)

    conflict = (
        (cdist(a0, b0) <= t)
        | (cdist(a0, b1) <= t)
        | (cdist(a1, b0) <= t)
        | (cdist(a1, b1) <= t)
    )
    seg_i, seg_j = np.nonzero(conflict)

    if same_direction_cos_min is None:
        excl_i, excl_j = seg_i, seg_j
    else:
        v_a = a1 - a0
        v_b = b1 - b0
        na = np.maximum(np.linalg.norm(v_a, axis=1, keepdims=True), 1e-12)
        nb = np.maximum(np.linalg.norm(v_b, axis=1, keepdims=True), 1e-12)
        dot = (v_a / na) @ (v_b / nb).T
        conflict_excl = conflict & (dot < float(same_direction_cos_min))
        excl_i, excl_j = np.nonzero(conflict_excl)

    return {
        "seg_i": seg_i.astype(int).tolist(),
        "seg_j": seg_j.astype(int).tolist(),
        "seg_i_excl_same_dir": excl_i.astype(int).tolist(),
        "seg_j_excl_same_dir": excl_j.astype(int).tolist(),
    }


def _overlap_for_agent_pair(
    path_a,
    path_b,
    path_idx_a,
    path_idx_b,
    threshold=ROBOT_DIAMETER,
    *,
    same_direction_cos_min=None,
):
    """Return deduped overlap counts for one agent pair."""
    conflicts = _conflicting_segment_indices(
        path_a,
        path_b,
        threshold,
        same_direction_cos_min=same_direction_cos_min,
    )
    seg_i = conflicts["seg_i"]
    seg_j = conflicts["seg_j"]
    seg_i_excl = conflicts["seg_i_excl_same_dir"]
    seg_j_excl = conflicts["seg_j_excl_same_dir"]
    unique = _unique_overlap_segments_for_pair(path_idx_a, path_idx_b, seg_i, seg_j)
    unique_excl = _unique_overlap_segments_for_pair(
        path_idx_a, path_idx_b, seg_i_excl, seg_j_excl
    )
    return {
        "overlap_segment_count": len(unique),
        "overlap_segment_count_excl_same_dir": len(unique_excl),
        "overlap_segment_pair_count": len(seg_i),
        "overlap_segment_pair_count_excl_same_dir": len(seg_i_excl),
        "unique_segments": unique,
        "unique_segments_excl_same_dir": unique_excl,
        "seg_i": seg_i,
        "seg_j": seg_j,
    }


def _aggregate_overlap(
    paths,
    threshold=ROBOT_DIAMETER,
    *,
    same_direction_cos_min=None,
    collect_segments=False,
):
    n = len(paths)
    num_pairs = n * (n - 1) // 2
    total_segments = 0
    total_segments_excl = 0
    total_segment_pairs = 0
    overlap_pairs = 0
    overlap_pairs_excl = 0
    by_path = [set() for _ in range(n)] if collect_segments else None
    by_path_excl_same_dir = [set() for _ in range(n)] if collect_segments else None

    for i in range(n):
        for j in range(i + 1, n):
            pair = _overlap_for_agent_pair(
                paths[i],
                paths[j],
                i,
                j,
                threshold,
                same_direction_cos_min=same_direction_cos_min,
            )
            total_segment_pairs += pair["overlap_segment_pair_count"]
            total_segments += pair["overlap_segment_count"]
            total_segments_excl += pair["overlap_segment_count_excl_same_dir"]
            if pair["overlap_segment_count"] > 0:
                overlap_pairs += 1
            if pair["overlap_segment_count_excl_same_dir"] > 0:
                overlap_pairs_excl += 1
            if collect_segments:
                for path_idx, seg_idx in pair["unique_segments"]:
                    by_path[path_idx].add(seg_idx)
                for path_idx, seg_idx in pair["unique_segments_excl_same_dir"]:
                    by_path_excl_same_dir[path_idx].add(seg_idx)

    mean_per_pair = float(total_segments) / num_pairs if num_pairs else 0.0
    mean_per_pair_excl = float(total_segments_excl) / num_pairs if num_pairs else 0.0
    metrics = {
        "overlap_segment_count": int(total_segments),
        "overlap_segment_count_excl_same_dir": int(total_segments_excl),
        "overlap_segment_pair_count": int(total_segment_pairs),
        "overlap_pairs": int(overlap_pairs),
        "overlap_pairs_excl_same_dir": int(overlap_pairs_excl),
        "mean_per_pair": float(mean_per_pair),
        "mean_per_pair_excl_same_dir": float(mean_per_pair_excl),
    }
    if collect_segments:
        return metrics, by_path, by_path_excl_same_dir
    return metrics


def spatial_overlap_metrics(paths, threshold=ROBOT_DIAMETER, *, same_direction_cos_min=None):
    return _aggregate_overlap(
        paths,
        threshold,
        same_direction_cos_min=same_direction_cos_min,
        collect_segments=False,
    )


def conflicting_segments_by_path(paths, threshold=ROBOT_DIAMETER, *, same_direction_cos_min=None):
    """For each path index, return segment indices that participate in any overlap."""
    _, by_path, _ = _aggregate_overlap(
        paths,
        threshold,
        same_direction_cos_min=same_direction_cos_min,
        collect_segments=True,
    )
    return by_path


def _plot_path_with_overlap_highlights(
    ax,
    path,
    base_color,
    overlap_seg_indices,
    overlap_excl_same_dir_indices=None,
):
    excl = overlap_excl_same_dir_indices or set()
    for k in range(len(path) - 1):
        x0, y0 = path[k]
        x1, y1 = path[k + 1]
        if k in overlap_seg_indices:
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="crimson",
                linewidth=4.5,
                alpha=0.95,
                solid_capstyle="round",
                zorder=8,
            )
        if k in excl:
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="royalblue",
                linewidth=4.5,
                alpha=0.95,
                solid_capstyle="round",
                zorder=9,
            )
        if k not in overlap_seg_indices:
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=base_color,
                linewidth=2.5,
                alpha=0.85,
                zorder=5,
            )


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


def _rrt_config_from_kwargs(rrt_kwargs):
    if not rrt_kwargs:
        return None
    return {k: v for k, v in rrt_kwargs.items()}


def _path_for_mode(route, mode):
    if mode == "vanilla":
        return _path_from_serializable(route["path"])
    mode_paths = route.get("paths_by_mode") or {}
    if mode not in mode_paths:
        raise KeyError(f"Route {route.get('route_id')} missing paths_by_mode[{mode!r}]")
    return _path_from_serializable(mode_paths[mode])


def _mode_cache_meta(mode, *, social_graph, heatmap_prefix, heatmap_file, rrt_kwargs):
    return astar_path_cache_meta(
        mode,
        social_graph,
        heatmap_prefix=heatmap_prefix,
        heatmap_file=heatmap_file,
        rrt_config=_rrt_config_from_kwargs(rrt_kwargs) if mode == "rrt_vanilla" else None,
    )


def _route_endpoint_key(x_init, x_goal):
    return (
        float(x_init[0]),
        float(x_init[1]),
        float(x_goal[0]),
        float(x_goal[1]),
    )


def _plan_all_modes_on_pair(
    x_init,
    x_goal,
    *,
    occ_grid,
    statespace_hi,
    resolution,
    social_graph,
    rrt_kwargs,
    modes=BANK_PLANNING_MODES,
):
    """Return mode->path when every solver succeeds on the same start/goal; else None."""
    paths = {}
    for mode in modes:
        path, _solve_time = run_solver(
            mode,
            occ_grid,
            statespace_hi,
            x_init,
            x_goal,
            resolution,
            social_graph=social_graph if mode == "modified" else None,
            rrt_kwargs=rrt_kwargs if mode == "rrt_vanilla" else None,
        )
        if path is None or len(path) < 2:
            return None
        paths[mode] = [(float(p[0]), float(p[1])) for p in path]
    return paths


def _route_record_from_paths(route_id, x_init, x_goal, paths_by_mode):
    return {
        "route_id": int(route_id),
        "x_init": [float(x_init[0]), float(x_init[1])],
        "x_goal": [float(x_goal[0]), float(x_goal[1])],
        "path": _path_to_serializable(paths_by_mode["vanilla"]),
        "paths_by_mode": {
            mode: _path_to_serializable(paths_by_mode[mode])
            for mode in NON_VANILLA_MODES
        },
    }


def _save_overlap_bank(
    path_bank_path,
    *,
    scenario,
    seed,
    pool_size,
    attempts,
    routes,
    cache_meta_by_mode,
    modified_config,
):
    path_bank_path = Path(path_bank_path)
    path_bank_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, route in enumerate(routes):
        route["route_id"] = int(idx)
    payload = {
        "scenario_name": scenario,
        "benchmark_seed": int(seed),
        "num_robots": int(pool_size),
        "sampling_attempts": int(attempts),
        "failure_policy": FAILURE_POLICY,
        "paths_by_mode_meta": {mode: cache_meta_by_mode[mode] for mode in NON_VANILLA_MODES},
        **modified_config,
        "routes": routes,
    }
    with path_bank_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _try_complete_existing_route(
    route,
    *,
    occ_grid,
    statespace_hi,
    resolution,
    social_graph,
    rrt_kwargs,
    refresh_modes,
):
    """Fill missing/refreshed mode paths on fixed endpoints; None if any required solve fails."""
    x_init = tuple(float(v) for v in route["x_init"])
    x_goal = tuple(float(v) for v in route["x_goal"])
    mode_paths = route.get("paths_by_mode") or {}
    need = set()
    for mode in BANK_PLANNING_MODES:
        if mode == "vanilla":
            if "vanilla" in refresh_modes or not route.get("path"):
                need.add(mode)
        elif mode in refresh_modes or mode not in mode_paths:
            need.add(mode)

    if not need:
        return route

    planned = _plan_all_modes_on_pair(
        x_init,
        x_goal,
        occ_grid=occ_grid,
        statespace_hi=statespace_hi,
        resolution=resolution,
        social_graph=social_graph,
        rrt_kwargs=rrt_kwargs,
        modes=tuple(need),
    )
    if planned is None:
        return None

    updated = dict(route)
    if "vanilla" in planned:
        updated["path"] = _path_to_serializable(planned["vanilla"])
    mode_paths = dict(updated.get("paths_by_mode") or {})
    for mode in NON_VANILLA_MODES:
        if mode in planned:
            mode_paths[mode] = _path_to_serializable(planned[mode])
    updated["paths_by_mode"] = mode_paths
    return updated


def verify_bank_modes(path_bank_path, pool_size, solver_modes, cache_meta_by_mode):
    with Path(path_bank_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    routes = payload.get("routes") or []
    if len(routes) < pool_size:
        raise ValueError(
            f"Path bank has {len(routes)} routes but pool_size={pool_size}. "
            "Run with --build-bank-only or omit it to finish building the bank."
        )
    missing = set()
    for mode in solver_modes:
        if mode == "vanilla":
            continue
        stored_meta = (payload.get("paths_by_mode_meta") or {}).get(mode)
        if stored_meta != cache_meta_by_mode[mode]:
            missing.add(mode)
            continue
        for route in routes[:pool_size]:
            if mode not in (route.get("paths_by_mode") or {}):
                missing.add(mode)
                break
    return sorted(missing)


def build_path_bank(
    *,
    scenario,
    pool_size,
    seed,
    max_attempts,
    path_bank_path,
    occ_grid,
    statespace_hi,
    resolution,
    social_graph,
    heatmap_prefix,
    heatmap_file,
    sparse_graph_threshold,
    sparse_min_component_size,
    rrt_kwargs,
    refresh_modes,
    cache_meta_by_mode,
    modified_config,
):
    """
    Grow a path bank where every route uses the same start/goal for vanilla A*,
    Social A*, and RRT*. Pairs where any solver fails are discarded (not resampled
    in place).
    """
    path_bank_path = Path(path_bank_path)
    refresh_modes = set(refresh_modes or [])

    routes = []
    attempts = 0
    seen = set()
    if path_bank_path.exists():
        with path_bank_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        attempts = int(payload.get("sampling_attempts", 0))
        policy = payload.get("failure_policy")
        if policy and policy != FAILURE_POLICY:
            print(
                f"Warning: existing bank failure_policy={policy!r}; "
                f"re-validating routes under {FAILURE_POLICY!r}."
            )
        original_routes = list(payload.get("routes") or [])
        for raw in original_routes:
            completed = _try_complete_existing_route(
                raw,
                occ_grid=occ_grid,
                statespace_hi=statespace_hi,
                resolution=resolution,
                social_graph=social_graph,
                rrt_kwargs=rrt_kwargs,
                refresh_modes=refresh_modes,
            )
            if completed is None:
                print(
                    f"Dropping route_id={raw.get('route_id')}: not all solvers succeed on "
                    f"{raw.get('x_init')} -> {raw.get('x_goal')}",
                    flush=True,
                )
                continue
            routes.append(completed)
            seen.add(_route_endpoint_key(completed["x_init"], completed["x_goal"]))
        if routes or original_routes:
            _save_overlap_bank(
                path_bank_path,
                scenario=scenario,
                seed=seed,
                pool_size=pool_size,
                attempts=attempts,
                routes=routes,
                cache_meta_by_mode=cache_meta_by_mode,
                modified_config=modified_config,
            )
        if routes:
            print(
                f"Loaded {len(routes)} paired routes from {path_bank_path.resolve()}",
                flush=True,
            )

    if len(routes) < pool_size:
        print(
            f"Collecting paired routes ({len(routes)}/{pool_size}): "
            f"all of {', '.join(BANK_PLANNING_MODES)} must succeed on the same start/goal.",
            flush=True,
        )
    rng = np.random.default_rng(int(seed) + len(routes) * 1009)
    while len(routes) < pool_size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Reached max attempts ({max_attempts}) before collecting {pool_size} "
                "paired routes where all solvers succeed."
            )

        if attempts % 20 == 0:
            print(
                f"[progress] attempts={attempts}, paired routes={len(routes)}/{pool_size}",
                flush=True,
            )

        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        if x_init == x_goal:
            continue
        key = _route_endpoint_key(x_init, x_goal)
        if key in seen:
            continue

        paths_by_mode = _plan_all_modes_on_pair(
            x_init,
            x_goal,
            occ_grid=occ_grid,
            statespace_hi=statespace_hi,
            resolution=resolution,
            social_graph=social_graph,
            rrt_kwargs=rrt_kwargs,
        )
        if paths_by_mode is None:
            continue

        seen.add(key)
        routes.append(_route_record_from_paths(len(routes), x_init, x_goal, paths_by_mode))
        _save_overlap_bank(
            path_bank_path,
            scenario=scenario,
            seed=seed,
            pool_size=pool_size,
            attempts=attempts,
            routes=routes,
            cache_meta_by_mode=cache_meta_by_mode,
            modified_config=modified_config,
        )
        print(
            f"Paired routes: {len(routes)}/{pool_size} "
            f"(start={x_init}, goal={x_goal})",
            flush=True,
        )

    if refresh_modes:
        print(f"Refreshing modes: {', '.join(sorted(refresh_modes))}", flush=True)
        refreshed = []
        for raw in routes:
            updated = _try_complete_existing_route(
                raw,
                occ_grid=occ_grid,
                statespace_hi=statespace_hi,
                resolution=resolution,
                social_graph=social_graph,
                rrt_kwargs=rrt_kwargs,
                refresh_modes=refresh_modes,
            )
            if updated is None:
                raise RuntimeError(
                    f"Refresh failed for route {raw.get('route_id')}: "
                    f"{raw.get('x_init')} -> {raw.get('x_goal')}"
                )
            refreshed.append(updated)
            for mode in refresh_modes:
                if mode == "vanilla":
                    continue
                merge_astar_paths_into_path_bank(
                    path_bank_path,
                    [
                        {
                            "route_id": updated["route_id"],
                            "x_init": updated["x_init"],
                            "x_goal": updated["x_goal"],
                            "path": _path_from_serializable(
                                updated["paths_by_mode"][mode]
                            ),
                        }
                    ],
                    mode,
                    cache_meta_by_mode[mode],
                )
        routes = refreshed
        _save_overlap_bank(
            path_bank_path,
            scenario=scenario,
            seed=seed,
            pool_size=pool_size,
            attempts=attempts,
            routes=routes,
            cache_meta_by_mode=cache_meta_by_mode,
            modified_config=modified_config,
        )

    print(f"Path bank pool: {len(routes)} paired routes at {path_bank_path.resolve()}")
    return routes, {"path_bank_path": str(path_bank_path.resolve()), "num_routes": len(routes)}


def plot_trial_debug(
    occ_grid,
    trial_id,
    route_ids,
    subset_routes,
    paths_by_mode,
    metrics_by_mode,
    overlap_highlight_by_mode,
    overlap_excl_highlight_by_mode,
    solver_modes,
):
    """Show one figure per trial; block until the user presses Enter."""
    n = len(solver_modes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10", max(len(subset_routes), 1))
    for j, mode in enumerate(solver_modes):
        ax = axes[j]
        occ_grid.plot_grid(ax=ax)
        paths = paths_by_mode[mode]
        overlap_by_path = overlap_highlight_by_mode[mode]
        overlap_excl_by_path = overlap_excl_highlight_by_mode[mode]
        for i, (route, path) in enumerate(zip(subset_routes, paths)):
            color = cmap(i % cmap.N)
            _plot_path_with_overlap_highlights(
                ax,
                path,
                color,
                overlap_by_path[i],
                overlap_excl_by_path[i],
            )
            x_init = route["x_init"]
            x_goal = route["x_goal"]
            ax.scatter(x_init[0], x_init[1], c=[color], s=40, zorder=9, marker="o")
            ax.scatter(x_goal[0], x_goal[1], c=[color], marker="*", s=80, zorder=9)

        overlap = metrics_by_mode[mode]["overlap_segment_count"]
        overlap_excl = metrics_by_mode[mode]["overlap_segment_count_excl_same_dir"]
        ax.set_title(
            f"{SOLVER_PLOT_TITLES.get(mode, mode)}\n"
            f"{overlap} overlapping segments\n"
            f"({overlap_excl} excl. same direction)",
            fontsize=16,
        )
        ax.set_axis_off()
        handles = [
            Line2D([0], [0], color="crimson", linewidth=4.5, label="all overlapping segments"),
            Line2D(
                [0],
                [0],
                color="royalblue",
                linewidth=4.5,
                label="overlap (excl. same direction)",
            ),
            Line2D([0], [0], marker="o", color="k", markerfacecolor="green", markersize=6, label="start"),
            Line2D([0], [0], marker="*", color="k", markerfacecolor="gold", markersize=10, label="goal"),
        ]
        ax.legend(handles=handles, loc="best", fontsize=10)

    fig.suptitle(
        f"Trial {trial_id} — routes {route_ids}",
        fontsize=20,
        y=1.02,
    )
    fig.tight_layout()
    plt.show(block=False)
    fig.canvas.draw_idle()
    try:
        input(f"Trial {trial_id}: press Enter for next trial...")
    except EOFError:
        pass
    plt.close(fig)


def run_overlap_trials(
    *,
    path_bank_path,
    pool_size,
    num_agents,
    num_trials,
    trial_seed,
    occ_grid,
    solver_modes,
    output,
    debug_plot_trials=False,
    overlap_sample_spacing=None,
    same_direction_cos_min=None,
    same_direction_angle_tol_deg=DEFAULT_SAME_DIRECTION_ANGLE_TOL_DEG,
):
    with Path(path_bank_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    bank_routes = (payload.get("routes") or [])[:pool_size]

    rng = np.random.default_rng(trial_seed)
    trial_rows = []
    for trial_id in range(1, num_trials + 1):
        subset_routes, route_ids = sample_route_subset(bank_routes, num_agents, rng)
        paths_by_mode = {}
        metrics_by_mode = {}
        overlap_highlight_by_mode = {}
        overlap_excl_highlight_by_mode = {}
        for mode in solver_modes:
            raw_paths = [_path_for_mode(r, mode) for r in subset_routes]
            paths = [normalize_path_for_overlap(p, overlap_sample_spacing) for p in raw_paths]
            if debug_plot_trials:
                metrics, by_path, by_path_excl = _aggregate_overlap(
                    paths,
                    same_direction_cos_min=same_direction_cos_min,
                    collect_segments=True,
                )
            else:
                metrics = spatial_overlap_metrics(
                    paths,
                    same_direction_cos_min=same_direction_cos_min,
                )
                by_path = None
                by_path_excl = None
            paths_by_mode[mode] = paths
            metrics_by_mode[mode] = metrics
            if by_path is not None:
                overlap_highlight_by_mode[mode] = by_path
                overlap_excl_highlight_by_mode[mode] = by_path_excl
            trial_rows.append(
                {
                    "trial": trial_id,
                    "solver": mode,
                    "sampled_route_ids": ",".join(str(r) for r in route_ids),
                    "num_agents": num_agents,
                    **metrics,
                }
            )
        if debug_plot_trials:
            plot_trial_debug(
                occ_grid,
                trial_id,
                route_ids,
                subset_routes,
                paths_by_mode,
                metrics_by_mode,
                overlap_highlight_by_mode,
                overlap_excl_highlight_by_mode,
                solver_modes,
            )

    summary = {}
    for mode in solver_modes:
        values = np.array(
            [r["overlap_segment_count"] for r in trial_rows if r["solver"] == mode],
            dtype=float,
        )
        values_excl = np.array(
            [
                r["overlap_segment_count_excl_same_dir"]
                for r in trial_rows
                if r["solver"] == mode
            ],
            dtype=float,
        )
        summary[mode] = {
            "num_trials": int(len(values)),
            "overlap_segment_count_mean": float(np.mean(values)) if len(values) else float("nan"),
            "overlap_segment_count_std": float(np.std(values)) if len(values) else float("nan"),
            "overlap_segment_count_excl_same_dir_mean": (
                float(np.mean(values_excl)) if len(values_excl) else float("nan")
            ),
            "overlap_segment_count_excl_same_dir_std": (
                float(np.std(values_excl)) if len(values_excl) else float("nan")
            ),
        }

    spacing_label = (
        "disabled" if overlap_sample_spacing is None or overlap_sample_spacing <= 0
        else f"{overlap_sample_spacing:.4g}m"
    )
    print("\n=== Spatial overlap (unique segments, robot_d={:.2f}m) ===".format(ROBOT_DIAMETER))
    print(f"Trials: {num_trials}, agents/trial: {num_agents}, pool: {pool_size}, sample spacing: {spacing_label}")
    print("Secondary metric omits co-directional overlaps (same travel direction within angle tolerance).")
    for mode in solver_modes:
        s = summary[mode]
        print(
            f"{mode:12s}  total mean={s['overlap_segment_count_mean']:.1f}  "
            f"std={s['overlap_segment_count_std']:.1f}  |  "
            f"excl same-dir mean={s['overlap_segment_count_excl_same_dir_mean']:.1f}  "
            f"std={s['overlap_segment_count_excl_same_dir_std']:.1f}"
        )

    if output:
        save_results(
            output,
            payload,
            trial_rows,
            summary,
            solver_modes,
            num_agents,
            pool_size,
            trial_seed,
            overlap_sample_spacing=overlap_sample_spacing,
            same_direction_angle_tol_deg=same_direction_angle_tol_deg,
        )

    return trial_rows, summary


def save_results(
    output_path,
    bank_payload,
    trial_rows,
    summary,
    solver_modes,
    num_agents,
    pool_size,
    trial_seed,
    overlap_sample_spacing=None,
    same_direction_angle_tol_deg=DEFAULT_SAME_DIRECTION_ANGLE_TOL_DEG,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        payload = {
            "config": {
                "scenario": bank_payload.get("scenario_name"),
                "pool_size": pool_size,
                "num_agents": num_agents,
                "num_trials": len(trial_rows) // len(solver_modes) if solver_modes else 0,
                "trial_seed": trial_seed,
                "solver_modes": list(solver_modes),
                "robot_diameter_m": ROBOT_DIAMETER,
                "overlap_sample_spacing_m": overlap_sample_spacing,
                "same_direction_angle_tol_deg": same_direction_angle_tol_deg,
            },
            "summary": summary,
            "trials": trial_rows,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to: {output_path}")
        return

    if output_path.suffix.lower() == ".csv":
        fieldnames = [
            "trial",
            "solver",
            "sampled_route_ids",
            "num_agents",
            "overlap_segment_count",
            "overlap_segment_count_excl_same_dir",
            "overlap_segment_pair_count",
            "overlap_pairs",
            "overlap_pairs_excl_same_dir",
            "mean_per_pair",
            "mean_per_pair_excl_same_dir",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in trial_rows:
                writer.writerow(row)
        print(f"\nSaved results to: {output_path}")
        return

    raise ValueError("Unsupported output format. Use .json or .csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a cached path bank (A*, RRT*, Social A*) and compare spatial "
            "path overlap across random multi-robot subsets."
        )
    )
    parser.add_argument("--scenario", default="sample2_default")
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--path-bank", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for path bank generation")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--build-bank-only", action="store_true")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--trial-seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--debug-plot-trials",
        action="store_true",
        help="Show an interactive 1x3 figure per trial; press Enter to advance.",
    )
    parser.add_argument(
        "--overlap-sample-spacing",
        type=float,
        default=None,
        help=(
            "Arc-length spacing (m) for overlap polylines before scoring. "
            "Default: map resolution. Use 0 to disable resampling."
        ),
    )
    parser.add_argument(
        "--same-direction-angle-tol-deg",
        type=float,
        default=DEFAULT_SAME_DIRECTION_ANGLE_TOL_DEG,
        help=(
            "Segment pairs with travel directions within this angle (degrees) "
            "are treated as same-direction and omitted from the secondary overlap metric."
        ),
    )
    parser.add_argument(
        "--refresh-mode",
        action="append",
        default=[],
        choices=list(SOLVER_MODES),
        help="Force replan cached paths for a solver mode (repeatable).",
    )
    parser.add_argument(
        "--solvers",
        default="all",
        help="Solvers for overlap trials: vanilla, rrt_vanilla, modified (default: all).",
    )
    rrt_group = parser.add_argument_group("RRT* parameters")
    rrt_group.add_argument("--rrt-max-iter", type=int, default=10000)
    rrt_group.add_argument("--rrt-step-size", type=float, default=1)
    rrt_group.add_argument("--rrt-goal-sample-rate", type=float, default=0.10)
    rrt_group.add_argument("--rrt-goal-tolerance", type=float, default=None)
    rrt_group.add_argument("--rrt-rewire-radius", type=float, default=None)
    heatmap_group = parser.add_argument_group("heatmap graph (optional modified solver)")
    heatmap_group.add_argument("--heatmap-prefix", type=str, default=None)
    heatmap_group.add_argument("--heatmap-file", type=str, default=None)
    heatmap_group.add_argument("--sparse-graph-threshold", type=float, default=2.0)
    heatmap_group.add_argument("--sparse-min-component-size", type=int, default=15)
    args = parser.parse_args()

    if args.pool_size <= 0:
        raise ValueError("--pool-size must be > 0")
    if args.num_agents <= 0:
        raise ValueError("--num-agents must be > 0")
    if args.num_trials < 0:
        raise ValueError("--num-trials must be >= 0")
    if args.overlap_sample_spacing is not None and args.overlap_sample_spacing < 0:
        raise ValueError("--overlap-sample-spacing must be >= 0")
    if args.same_direction_angle_tol_deg < 0 or args.same_direction_angle_tol_deg > 180:
        raise ValueError("--same-direction-angle-tol-deg must be in [0, 180]")
    if args.max_attempts is None:
        args.max_attempts = max(100, 50 * args.pool_size)
    if args.path_bank is None:
        args.path_bank = f"results/overlap_banks/{args.scenario}_seed{args.seed}.json"
    args.solver_modes = parse_solver_modes(args.solvers)
    return args


def main():
    args = parse_args()
    path_bank_path = Path(args.path_bank)
    occ_grid, _, resolution, statespace_hi = build_occ_grid(args.scenario)
    overlap_sample_spacing = resolution if args.overlap_sample_spacing is None else args.overlap_sample_spacing
    if overlap_sample_spacing == 0:
        overlap_sample_spacing = None
    same_direction_cos_min = _same_direction_cos_min(args.same_direction_angle_tol_deg)

    modified_config = _modified_solver_config(
        args.heatmap_prefix,
        args.heatmap_file,
        args.sparse_graph_threshold,
        args.sparse_min_component_size,
    )
    social_graph = None
    if modified_config["modified_solver"] == "heatmap_graph":
        social_graph = build_sparse_graph_from_heatmap(
            occ_grid,
            heatmap_prefix=args.heatmap_prefix,
            heatmap_path=args.heatmap_file,
            sparse_graph_threshold=args.sparse_graph_threshold,
            sparse_min_component_size=args.sparse_min_component_size,
        )

    rrt_kwargs = rrt_kwargs_from_config(
        rrt_max_iter=args.rrt_max_iter,
        rrt_step_size=args.rrt_step_size,
        rrt_goal_sample_rate=args.rrt_goal_sample_rate,
        rrt_goal_tolerance=args.rrt_goal_tolerance,
        rrt_rewire_radius=args.rrt_rewire_radius,
        resolution=resolution,
    )

    bank_cache_meta_by_mode = {
        mode: _mode_cache_meta(
            mode,
            social_graph=social_graph,
            heatmap_prefix=args.heatmap_prefix,
            heatmap_file=args.heatmap_file,
            rrt_kwargs=rrt_kwargs,
        )
        for mode in NON_VANILLA_MODES
    }
    trial_cache_meta_by_mode = {
        mode: bank_cache_meta_by_mode.get(
            mode,
            _mode_cache_meta(
                mode,
                social_graph=social_graph,
                heatmap_prefix=args.heatmap_prefix,
                heatmap_file=args.heatmap_file,
                rrt_kwargs=rrt_kwargs,
            ),
        )
        for mode in args.solver_modes
    }

    missing = verify_bank_modes(
        path_bank_path,
        args.pool_size,
        args.solver_modes,
        trial_cache_meta_by_mode,
    ) if path_bank_path.exists() else list(NON_VANILLA_MODES)

    if missing or args.refresh_mode or not path_bank_path.exists():
        build_path_bank(
            scenario=args.scenario,
            pool_size=args.pool_size,
            seed=args.seed,
            max_attempts=args.max_attempts,
            path_bank_path=path_bank_path,
            occ_grid=occ_grid,
            statespace_hi=statespace_hi,
            resolution=resolution,
            social_graph=social_graph,
            heatmap_prefix=args.heatmap_prefix,
            heatmap_file=args.heatmap_file,
            sparse_graph_threshold=args.sparse_graph_threshold,
            sparse_min_component_size=args.sparse_min_component_size,
            rrt_kwargs=rrt_kwargs,
            refresh_modes=set(args.refresh_mode),
            cache_meta_by_mode=bank_cache_meta_by_mode,
            modified_config=modified_config,
        )
        missing = verify_bank_modes(
            path_bank_path, args.pool_size, args.solver_modes, trial_cache_meta_by_mode
        )
        if missing:
            raise RuntimeError(f"Path bank still missing modes after build: {missing}")

    print("\n=== Path bank ===")
    print(f"Loaded {args.pool_size} routes from {path_bank_path.resolve()}")
    print(f"Modes cached: {', '.join(args.solver_modes)}")

    if args.build_bank_only:
        return

    if args.num_trials == 0:
        print("Skipping overlap trials (--num-trials 0).")
        return

    if args.num_agents > args.pool_size:
        raise ValueError(f"--num-agents ({args.num_agents}) exceeds --pool-size ({args.pool_size})")

    run_overlap_trials(
        path_bank_path=path_bank_path,
        pool_size=args.pool_size,
        num_agents=args.num_agents,
        num_trials=args.num_trials,
        trial_seed=args.trial_seed,
        occ_grid=occ_grid,
        solver_modes=args.solver_modes,
        output=args.output,
        debug_plot_trials=args.debug_plot_trials,
        overlap_sample_spacing=overlap_sample_spacing,
        same_direction_cos_min=same_direction_cos_min,
        same_direction_angle_tol_deg=args.same_direction_angle_tol_deg,
    )


if __name__ == "__main__":
    main()
