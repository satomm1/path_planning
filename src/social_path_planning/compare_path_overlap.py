"""Compare spatial path overlap for multi-robot route sets across A*, RRT*, and Social A*."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from social_path_planning.benchmark_sparse_snapshots import (
    _path_from_serializable,
    _path_to_serializable,
    astar_path_cache_meta,
    merge_astar_paths_into_path_bank,
)
from social_path_planning.compare_astar import (
    SOLVER_MODES,
    build_occ_grid,
    build_sparse_graph_from_heatmap,
    generate_random_free_point,
    parse_solver_modes,
    run_solver,
)
from social_path_planning.compare_astar import _rrt_kwargs_from_config as rrt_kwargs_from_config
from social_path_planning.mapf_comparison.ensemble import sample_route_subset
from social_path_planning.mapf_comparison.solution_metrics import dedupe_world_path
from social_path_planning.multi_planning import (
    ROBOT_DIAMETER,
    detect_collision_pairs_for_agent_pair,
)

FAILURE_POLICY = "all_solvers_same_endpoints"
BANK_PLANNING_MODES = SOLVER_MODES
NON_VANILLA_MODES = ("modified", "rrt_vanilla")


def normalize_path_for_metrics(path, occ_grid):
    snapped = []
    for point in path:
        x, y = occ_grid.snap_to_grid(point)
        snapped.append((float(x), float(y)))
    return dedupe_world_path(snapped)


def spatial_overlap_metrics(paths, threshold=ROBOT_DIAMETER):
    n = len(paths)
    num_pairs = n * (n - 1) // 2
    total_segments = 0
    overlap_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            _, _, seg_i, _ = detect_collision_pairs_for_agent_pair(
                paths[i], paths[j], i, j, threshold
            )
            count = len(seg_i)
            total_segments += count
            if count > 0:
                overlap_pairs += 1
    mean_per_pair = float(total_segments) / num_pairs if num_pairs else 0.0
    return {
        "overlap_segment_count": int(total_segments),
        "overlap_pairs": int(overlap_pairs),
        "mean_per_pair": float(mean_per_pair),
    }


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
):
    with Path(path_bank_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    bank_routes = (payload.get("routes") or [])[:pool_size]

    rng = np.random.default_rng(trial_seed)
    trial_rows = []
    for trial_id in range(1, num_trials + 1):
        subset_routes, route_ids = sample_route_subset(bank_routes, num_agents, rng)
        for mode in solver_modes:
            raw_paths = [_path_for_mode(r, mode) for r in subset_routes]
            paths = [normalize_path_for_metrics(p, occ_grid) for p in raw_paths]
            metrics = spatial_overlap_metrics(paths)
            trial_rows.append(
                {
                    "trial": trial_id,
                    "solver": mode,
                    "sampled_route_ids": ",".join(str(r) for r in route_ids),
                    "num_agents": num_agents,
                    **metrics,
                }
            )

    summary = {}
    for mode in solver_modes:
        values = np.array(
            [r["overlap_segment_count"] for r in trial_rows if r["solver"] == mode],
            dtype=float,
        )
        summary[mode] = {
            "num_trials": int(len(values)),
            "overlap_segment_count_mean": float(np.mean(values)) if len(values) else float("nan"),
            "overlap_segment_count_std": float(np.std(values)) if len(values) else float("nan"),
        }

    print("\n=== Spatial overlap (segment conflicts, robot_d={:.2f}m) ===".format(ROBOT_DIAMETER))
    print(f"Trials: {num_trials}, agents/trial: {num_agents}, pool: {pool_size}")
    for mode in solver_modes:
        s = summary[mode]
        print(
            f"{mode:12s}  mean={s['overlap_segment_count_mean']:.1f}  "
            f"std={s['overlap_segment_count_std']:.1f}"
        )

    if output:
        save_results(output, payload, trial_rows, summary, solver_modes, num_agents, pool_size, trial_seed)

    return trial_rows, summary


def save_results(output_path, bank_payload, trial_rows, summary, solver_modes, num_agents, pool_size, trial_seed):
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
            "overlap_pairs",
            "mean_per_pair",
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
    )


if __name__ == "__main__":
    main()
