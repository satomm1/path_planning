import argparse
import csv
import json
import pickle
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from social_path_planning.a_star import AStar, AStar_With_Graph
from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.mapf_comparison.grid_traversability import ROBOT_DIAMETER_M
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.utils import snap_to_grid


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
        robot_d=ROBOT_DIAMETER_M,
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


def _path_to_serializable(path):
    return [[float(p[0]), float(p[1])] for p in path]


def _path_from_serializable(path_data):
    return [tuple(float(v) for v in point) for point in path_data]


def _route_endpoint_key(x_init, x_goal):
    return (
        float(x_init[0]),
        float(x_init[1]),
        float(x_goal[0]),
        float(x_goal[1]),
    )


def _collect_endpoint_keys(routes, *, exclude_route_id=None):
    keys = set()
    for route in routes:
        if exclude_route_id is not None and int(route["route_id"]) == int(exclude_route_id):
            continue
        keys.add(_route_endpoint_key(route["x_init"], route["x_goal"]))
    return keys


def _sample_vanilla_solvable_route(
    occ_grid,
    statespace_hi,
    map_resolution,
    rng,
    seen,
    max_attempts,
):
    """Sample a random start/goal pair solvable by vanilla A*."""
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        if x_init == x_goal:
            continue
        route_key = _route_endpoint_key(x_init, x_goal)
        if route_key in seen:
            continue

        planner = AStar([0, 0], statespace_hi, x_init, x_goal, occ_grid, resolution=map_resolution)
        solved = planner.solve(mode="vanilla")
        if not solved or planner.path is None or len(planner.path) < 2:
            continue

        seen.add(route_key)
        return x_init, x_goal, planner.path, attempts

    raise RuntimeError(
        f"Reached max attempts ({max_attempts}) while resampling a vanilla-solvable route."
    )


def update_path_bank_vanilla_route(path_bank_path, route_id, x_init, x_goal, vanilla_path):
    """Replace one pool route's endpoints and vanilla polyline; clear cached mode paths."""
    path_bank_path = Path(path_bank_path)
    if not path_bank_path.exists():
        raise FileNotFoundError(f"Path bank not found for update: {path_bank_path}")

    with path_bank_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    bank_routes = payload.get("routes") or []
    updated = False
    for bank_route in bank_routes:
        if int(bank_route["route_id"]) != int(route_id):
            continue
        bank_route["x_init"] = [float(x_init[0]), float(x_init[1])]
        bank_route["x_goal"] = [float(x_goal[0]), float(x_goal[1])]
        bank_route["path"] = _path_to_serializable(vanilla_path)
        bank_route.pop("paths_by_mode", None)
        updated = True
        break

    if not updated:
        raise ValueError(f"Route id {route_id} missing from path bank {path_bank_path}")

    with path_bank_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_path_bank(
    path_bank_path,
    scenario_name,
    benchmark_seed,
    num_robots,
    attempts,
    routes,
    astar_mode="vanilla",
):
    payload = {
        "scenario_name": scenario_name,
        "benchmark_seed": int(benchmark_seed),
        "num_robots": int(num_robots),
        "sampling_attempts": int(attempts),
        "astar_mode": str(astar_mode),
        "routes": routes,
    }
    path_bank_path.parent.mkdir(parents=True, exist_ok=True)
    with path_bank_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def enrich_routes_with_astar_paths(
    occ_grid,
    statespace_hi,
    map_resolution,
    routes,
    mode="modified",
    social_graph=None,
    *,
    path_bank_path=None,
    cache_meta=None,
    max_resample_attempts=500,
    resample_seed=None,
):
    """
    Re-plan each route's geometry with A* (vanilla or modified/social) at fixed start/goal.

    When ``max_resample_attempts > 0`` and ``path_bank_path`` is set, a failed modified/social
    solve resamples a new vanilla-solvable start/goal for that ``route_id``, updates the path
    bank, and retries until success or the attempt budget is exhausted.
    """
    from social_path_planning.compare_astar import run_solver

    enriched = []
    total = len(routes)
    working_routes = [dict(route) for route in routes]
    resample_rng = np.random.default_rng(resample_seed)

    for idx, route in enumerate(working_routes, start=1):
        route_id = int(route["route_id"])
        resample_count = 0

        while True:
            print(
                f"MILP geometry ({mode} A*): {idx}/{total} "
                f"(route_id={route_id})",
                flush=True,
            )
            path, _solve_time = run_solver(
                mode,
                occ_grid,
                statespace_hi,
                route["x_init"],
                route["x_goal"],
                map_resolution,
                social_graph=social_graph,
            )
            if path is not None and len(path) >= 2:
                enriched_route = {
                    **route,
                    "path": [(float(p[0]), float(p[1])) for p in path],
                    "astar_mode": mode,
                }
                enriched.append(enriched_route)
                working_routes[idx - 1] = {
                    "route_id": route_id,
                    "x_init": tuple(route["x_init"]),
                    "x_goal": tuple(route["x_goal"]),
                    "path": enriched_route["path"],
                }
                if path_bank_path is not None and cache_meta is not None:
                    merge_astar_paths_into_path_bank(
                        path_bank_path, [enriched_route], mode, cache_meta
                    )
                break

            if max_resample_attempts <= 0 or path_bank_path is None:
                raise RuntimeError(
                    f"A* mode={mode!r} failed for route {route_id}: "
                    f"{route['x_init']} -> {route['x_goal']}"
                )

            resample_count += 1
            if resample_count > max_resample_attempts:
                raise RuntimeError(
                    f"A* mode={mode!r} failed for route {route_id} after "
                    f"{max_resample_attempts} resample attempts."
                )

            print(
                f"  {mode} A* failed for route_id={route_id}; resampling start/goal "
                f"({resample_count}/{max_resample_attempts})",
                flush=True,
            )
            seen = _collect_endpoint_keys(working_routes, exclude_route_id=route_id)
            x_init, x_goal, vanilla_path, _ = _sample_vanilla_solvable_route(
                occ_grid,
                statespace_hi,
                map_resolution,
                resample_rng,
                seen,
                max_resample_attempts,
            )
            route = {
                "route_id": route_id,
                "x_init": tuple(x_init),
                "x_goal": tuple(x_goal),
                "path": [tuple(float(v) for v in p) for p in vanilla_path],
            }
            working_routes[idx - 1] = route
            update_path_bank_vanilla_route(
                path_bank_path, route_id, x_init, x_goal, vanilla_path
            )

    return enriched


def astar_path_cache_meta(
    mode: str,
    social_graph=None,
    *,
    heatmap_prefix=None,
    heatmap_file=None,
):
    """Metadata describing how cached ``paths_by_mode`` polylines were produced."""
    if mode == "vanilla":
        return {"mode": "vanilla", "solver": "vanilla"}
    if social_graph is not None:
        return {
            "mode": mode,
            "solver": "heatmap_graph",
            "heatmap_prefix": heatmap_prefix,
            "heatmap_file": str(heatmap_file) if heatmap_file is not None else None,
        }
    return {"mode": mode, "solver": "rightness_penalty"}


def _route_endpoints_match(route_a, route_b) -> bool:
    a_init = tuple(float(v) for v in route_a["x_init"])
    a_goal = tuple(float(v) for v in route_a["x_goal"])
    b_init = tuple(float(v) for v in route_b["x_init"])
    b_goal = tuple(float(v) for v in route_b["x_goal"])
    return a_init == b_init and a_goal == b_goal


def _try_load_cached_astar_paths(routes, path_bank_path, mode, cache_meta):
    """
    Return routes with ``path`` taken from ``paths_by_mode[mode]`` when the bank
    matches ``cache_meta`` and endpoints agree; otherwise return None.
    """
    path_bank_path = Path(path_bank_path)
    if not path_bank_path.exists():
        return None

    with path_bank_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    stored_meta = (payload.get("paths_by_mode_meta") or {}).get(mode)
    if stored_meta != cache_meta:
        return None

    bank_routes = payload.get("routes") or []
    bank_by_id = {int(r["route_id"]): r for r in bank_routes}

    loaded = []
    for route in routes:
        bank_route = bank_by_id.get(int(route["route_id"]))
        if bank_route is None or not _route_endpoints_match(route, bank_route):
            return None
        mode_paths = bank_route.get("paths_by_mode") or {}
        if mode not in mode_paths:
            return None
        path = _path_from_serializable(mode_paths[mode])
        if len(path) < 2:
            return None
        loaded.append(
            {
                **route,
                "path": path,
                "astar_mode": mode,
            }
        )
    return loaded


def merge_astar_paths_into_path_bank(path_bank_path, routes, mode, cache_meta):
    """Persist per-mode polylines into an existing path-bank JSON file."""
    path_bank_path = Path(path_bank_path)
    if not path_bank_path.exists():
        raise FileNotFoundError(f"Path bank not found for merge: {path_bank_path}")

    with path_bank_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    bank_routes = payload.get("routes") or []
    bank_by_id = {int(r["route_id"]): r for r in bank_routes}

    for route in routes:
        route_id = int(route["route_id"])
        bank_route = bank_by_id.get(route_id)
        if bank_route is None:
            raise ValueError(f"Route id {route_id} missing from path bank {path_bank_path}")
        if not _route_endpoints_match(route, bank_route):
            raise ValueError(
                f"Route {route_id} endpoints differ between run and path bank {path_bank_path}"
            )
        mode_paths = bank_route.setdefault("paths_by_mode", {})
        mode_paths[mode] = _path_to_serializable(route["path"])

    meta = payload.setdefault("paths_by_mode_meta", {})
    meta[mode] = cache_meta

    with path_bank_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_or_compute_astar_paths(
    occ_grid,
    statespace_hi,
    map_resolution,
    routes,
    mode,
    path_bank_path,
    *,
    social_graph=None,
    heatmap_prefix=None,
    heatmap_file=None,
    refresh=False,
    max_resample_attempts=500,
    resample_seed=None,
):
    """
    Use path-bank polylines for ``vanilla``; load or compute other modes and cache them
    in ``paths_by_mode`` on the same JSON file as the vanilla bank.
    """
    if mode == "vanilla":
        return routes

    path_bank_path = Path(path_bank_path)
    cache_meta = astar_path_cache_meta(
        mode,
        social_graph,
        heatmap_prefix=heatmap_prefix,
        heatmap_file=heatmap_file,
    )

    if not refresh:
        cached = _try_load_cached_astar_paths(routes, path_bank_path, mode, cache_meta)
        if cached is not None:
            print(
                f"Using cached A* mode={mode!r} paths from {path_bank_path.resolve()} "
                f"({cache_meta.get('solver', mode)})"
            )
            return cached

    print(f"Planning A* mode={mode!r} (will cache in path bank)...")
    computed = enrich_routes_with_astar_paths(
        occ_grid,
        statespace_hi,
        map_resolution,
        routes,
        mode=mode,
        social_graph=social_graph,
        path_bank_path=path_bank_path,
        cache_meta=cache_meta,
        max_resample_attempts=max_resample_attempts,
        resample_seed=resample_seed,
    )
    print(f"Cached {mode!r} paths in {path_bank_path.resolve()}")
    return computed


def load_or_generate_path_bank(
    occ_grid,
    statespace_hi,
    map_resolution,
    scenario_name,
    num_robots,
    benchmark_seed,
    max_attempts,
    path_bank_path,
    astar_mode="vanilla",
):
    path_bank_path = Path(path_bank_path)
    if path_bank_path.exists():
        with path_bank_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        routes = payload.get("routes", [])
        if len(routes) < num_robots:
            raise ValueError(
                f"Existing path bank has {len(routes)} robots but {num_robots} are required: {path_bank_path}"
            )
        parsed = []
        for route in routes[:num_robots]:
            parsed.append(
                {
                    "route_id": int(route["route_id"]),
                    "x_init": tuple(float(v) for v in route["x_init"]),
                    "x_goal": tuple(float(v) for v in route["x_goal"]),
                    "path": _path_from_serializable(route["path"]),
                }
            )
        print(
            f"Loaded {num_robots} pool routes from {path_bank_path.resolve()}",
            flush=True,
        )
        return parsed, {"source": "loaded", "path_bank_path": str(path_bank_path.resolve()), "payload": payload}

    print(
        f"Generating {num_robots} pool routes (vanilla A*)...",
        flush=True,
    )
    rng = np.random.default_rng(benchmark_seed)
    routes = []
    attempts = 0
    seen = set()
    while len(routes) < num_robots:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Reached max attempts ({max_attempts}) before collecting {num_robots} solved paths."
            )
        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        if x_init == x_goal:
            continue
        route_key = (float(x_init[0]), float(x_init[1]), float(x_goal[0]), float(x_goal[1]))
        if route_key in seen:
            continue

        planner = AStar([0, 0], statespace_hi, x_init, x_goal, occ_grid, resolution=map_resolution)
        solved = planner.solve(mode=astar_mode)
        if not solved or planner.path is None or len(planner.path) < 2:
            continue

        seen.add(route_key)
        routes.append(
            {
                "route_id": int(len(routes)),
                "x_init": [float(x_init[0]), float(x_init[1])],
                "x_goal": [float(x_goal[0]), float(x_goal[1])],
                "path": _path_to_serializable(planner.path),
            }
        )
        print(
            f"Pool routes (vanilla A*): {len(routes)}/{num_robots}",
            flush=True,
        )

    _save_path_bank(
        path_bank_path, scenario_name, benchmark_seed, num_robots, attempts, routes, astar_mode=astar_mode
    )
    parsed = [
        {
            "route_id": int(route["route_id"]),
            "x_init": tuple(float(v) for v in route["x_init"]),
            "x_goal": tuple(float(v) for v in route["x_goal"]),
            "path": _path_from_serializable(route["path"]),
        }
        for route in routes
    ]
    return parsed, {"source": "generated", "path_bank_path": str(path_bank_path.resolve()), "sampling_attempts": attempts}


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


def run_distributed_constraint_timing(
    scenario_name,
    benchmark_seed,
    max_attempts,
    output_prefix,
    stage_sizes=None,
    path_bank_path=None,
):
    from multi_planning import MultiAgentSimultaneousPlanner, MAX_VELOCITY

    stage_sizes = stage_sizes or [2, 4, 8, 16, 32]
    stage_sizes = [int(s) for s in stage_sizes]
    if not stage_sizes:
        raise ValueError("stage_sizes must contain at least one stage")
    if sorted(stage_sizes) != stage_sizes:
        raise ValueError("stage_sizes must be sorted ascending")
    if stage_sizes[0] < 2:
        raise ValueError("smallest stage must be >= 2")

    max_stage = stage_sizes[-1]
    occ_grid, _, map_resolution, statespace_hi = build_occ_grid(scenario_name)
    if path_bank_path is None:
        output_prefix_path = Path(output_prefix)
        path_bank_path = output_prefix_path.parent / f"{output_prefix_path.name}_path_bank.json"

    path_bank, path_bank_meta = load_or_generate_path_bank(
        occ_grid=occ_grid,
        statespace_hi=statespace_hi,
        map_resolution=map_resolution,
        scenario_name=scenario_name,
        num_robots=max_stage,
        benchmark_seed=benchmark_seed,
        max_attempts=max_attempts,
        path_bank_path=path_bank_path,
    )

    if len(path_bank) < max_stage:
        raise ValueError(f"Path bank contains {len(path_bank)} paths, but {max_stage} are required")

    run_id = f"distributed_seed_{benchmark_seed}_stages_{'-'.join(str(s) for s in stage_sizes)}"
    frozen_prefix = {}
    detailed_rows = []
    summary_rows = []

    for stage_size in stage_sizes:
        print(f"Beginning stage size {stage_size}")

        stage_records = path_bank[:stage_size]
        stage_paths = [record["path"] for record in stage_records]

        for robot_idx, path in enumerate(stage_paths):
            if robot_idx in frozen_prefix and frozen_prefix[robot_idx] != path:
                raise ValueError(f"Path prefix mismatch for robot {robot_idx} at stage {stage_size}")
            frozen_prefix.setdefault(robot_idx, list(path))

        planner = MultiAgentSimultaneousPlanner(occ_grid, paths=stage_paths)
        planner.assign_velocities([MAX_VELOCITY] * stage_size)
        solve_t0 = time.perf_counter()
        optimized_times = planner.plan()
        solve_time_s = float(time.perf_counter() - solve_t0)
        solver_success = bool(optimized_times is not None and len(optimized_times) == stage_size)

        pair_rows = list(planner.constraint_timing_records)
        expected_pairs = stage_size * (stage_size - 1) // 2
        if len(pair_rows) != expected_pairs:
            raise ValueError(
                f"Expected {expected_pairs} pair timing records for stage {stage_size}, got {len(pair_rows)}"
            )

        detect_times = []
        build_times = []
        tuple_counts = []
        constraint_counts = []
        for pair_row in pair_rows:
            robot_i = int(pair_row["robot_i"])
            robot_j = int(pair_row["robot_j"])
            detect_time = float(pair_row["pair_detect_time_s"])
            build_time = float(pair_row["pair_constraint_build_time_s"])
            collision_tuple_count = int(pair_row["collision_tuple_count"])
            constraint_count = int(pair_row["constraint_count"])
            route_i = stage_records[robot_i]
            route_j = stage_records[robot_j]
            detailed_rows.append(
                {
                    "run_id": run_id,
                    "stage_size": stage_size,
                    "robot_i": robot_i,
                    "robot_j": robot_j,
                    "pair_detect_time_s": detect_time,
                    "pair_constraint_build_time_s": build_time,
                    "collision_tuple_count": collision_tuple_count,
                    "constraint_count": constraint_count,
                    "solve_time_s": solve_time_s,
                    "solver_success": solver_success,
                    "seed": int(benchmark_seed),
                    "robot_i_x_init": [float(route_i["x_init"][0]), float(route_i["x_init"][1])],
                    "robot_i_x_goal": [float(route_i["x_goal"][0]), float(route_i["x_goal"][1])],
                    "robot_j_x_init": [float(route_j["x_init"][0]), float(route_j["x_init"][1])],
                    "robot_j_x_goal": [float(route_j["x_goal"][0]), float(route_j["x_goal"][1])],
                }
            )
            detect_times.append(detect_time)
            build_times.append(build_time)
            tuple_counts.append(collision_tuple_count)
            constraint_counts.append(constraint_count)

        detect_stats = summarize(detect_times)
        build_stats = summarize(build_times)
        summary_rows.append(
            {
                "run_id": run_id,
                "stage_size": stage_size,
                "pair_count": expected_pairs,
                "pair_detect_time_total_s": float(np.sum(detect_times)),
                "pair_detect_time_mean_s": detect_stats["mean"],
                "pair_detect_time_median_s": detect_stats["median"],
                "pair_detect_time_p95_s": detect_stats["p95"],
                "pair_constraint_build_time_total_s": float(np.sum(build_times)),
                "pair_constraint_build_time_mean_s": build_stats["mean"],
                "pair_constraint_build_time_median_s": build_stats["median"],
                "pair_constraint_build_time_p95_s": build_stats["p95"],
                "collision_tuple_count_total": int(np.sum(tuple_counts)),
                "constraint_count_total": int(np.sum(constraint_counts)),
                "solve_time_s": solve_time_s,
                "solver_success": solver_success,
                "seed": int(benchmark_seed),
            }
        )
        print("\n\n\n")

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detailed_csv_path = Path(f"{output_prefix}_pair_timings.csv")
    summary_csv_path = Path(f"{output_prefix}_stage_summary.csv")
    manifest_json_path = Path(f"{output_prefix}_manifest.json")

    with detailed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "stage_size",
                "robot_i",
                "robot_j",
                "pair_detect_time_s",
                "pair_constraint_build_time_s",
                "collision_tuple_count",
                "constraint_count",
                "solve_time_s",
                "solver_success",
                "seed",
                "robot_i_x_init",
                "robot_i_x_goal",
                "robot_j_x_init",
                "robot_j_x_goal",
            ],
        )
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(row)

    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "stage_size",
                "pair_count",
                "pair_detect_time_total_s",
                "pair_detect_time_mean_s",
                "pair_detect_time_median_s",
                "pair_detect_time_p95_s",
                "pair_constraint_build_time_total_s",
                "pair_constraint_build_time_mean_s",
                "pair_constraint_build_time_median_s",
                "pair_constraint_build_time_p95_s",
                "collision_tuple_count_total",
                "constraint_count_total",
                "solve_time_s",
                "solver_success",
                "seed",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    manifest = {
        "config": {
            "scenario_name": scenario_name,
            "benchmark_seed": int(benchmark_seed),
            "max_attempts": int(max_attempts),
            "stage_sizes": stage_sizes,
            "output_prefix": str(output_prefix.resolve()),
        },
        "path_bank": path_bank_meta,
        "validation": {
            "expected_pairs_by_stage": {str(stage): int(stage * (stage - 1) // 2) for stage in stage_sizes},
            "prefix_paths_frozen": True,
        },
        "summary": summary_rows,
    }
    with manifest_json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved pair timing CSV: {detailed_csv_path}")
    print(f"Saved stage summary CSV: {summary_csv_path}")
    print(f"Saved manifest JSON: {manifest_json_path}")


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
    parser.add_argument(
        "--mode",
        type=str,
        default="snapshot",
        choices=["snapshot", "distributed_constraints"],
        help="Benchmark mode: snapshot A* benchmark or distributed constraint timing benchmark",
    )
    parser.add_argument("--timeline-dir", type=str, default=None, help="Directory containing graph_*.pkl and run_metadata.json")
    parser.add_argument("--scenario-name", type=str, default=None, help="Scenario name for occupancy reconstruction")
    parser.add_argument("--num-routes", type=int, default=10, help="Number of fixed seeded route pairs to benchmark (snapshot mode)")
    parser.add_argument("--benchmark-seed", type=int, default=0, help="Seed for fixed benchmark route generation")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum attempts to sample fixed unique routes")
    parser.add_argument("--output-prefix", type=str, default=None, help="Output prefix path (without extension)")
    parser.add_argument("--plot", action="store_true", help="Generate benchmark plot image")
    parser.add_argument(
        "--stage-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Robot counts to run for distributed constraint timing mode",
    )
    parser.add_argument(
        "--path-bank-path",
        type=str,
        default=None,
        help="Optional JSON cache path for deterministic path bank in distributed constraint timing mode",
    )
    args = parser.parse_args()

    if args.mode == "snapshot" and args.timeline_dir is None:
        raise ValueError("--timeline-dir is required in snapshot mode")
    if args.mode == "snapshot" and args.num_routes <= 0:
        raise ValueError("--num-routes must be > 0 in snapshot mode")
    if args.mode == "distributed_constraints":
        if not args.stage_sizes:
            raise ValueError("--stage-sizes must contain at least one value")
        args.stage_sizes = [int(s) for s in args.stage_sizes]
        if min(args.stage_sizes) < 2:
            raise ValueError("All --stage-sizes values must be >= 2")

    if args.max_attempts is None:
        if args.mode == "snapshot":
            args.max_attempts = max(100, 100 * args.num_routes)
        else:
            args.max_attempts = max(10000, 2000 * max(args.stage_sizes))
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be > 0")

    if args.mode == "snapshot":
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
    else:
        if args.scenario_name is None:
            raise ValueError("--scenario-name is required in distributed_constraints mode")
        if args.output_prefix is None:
            args.output_prefix = (
                "results/distributed_constraint_timing/distributed_constraint_timing"
            )
    return args


if __name__ == "__main__":
    cli = parse_args()
    if cli.mode == "snapshot":
        run_benchmark(
            timeline_dir=cli.timeline_dir,
            scenario_name=cli.scenario_name,
            num_routes=cli.num_routes,
            benchmark_seed=cli.benchmark_seed,
            max_attempts=cli.max_attempts,
            output_prefix=cli.output_prefix,
            make_plot=cli.plot,
        )
    else:
        run_distributed_constraint_timing(
            scenario_name=cli.scenario_name,
            benchmark_seed=cli.benchmark_seed,
            max_attempts=cli.max_attempts,
            output_prefix=cli.output_prefix,
            stage_sizes=cli.stage_sizes,
            path_bank_path=cli.path_bank_path,
        )
