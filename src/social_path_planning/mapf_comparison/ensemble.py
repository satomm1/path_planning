"""Multi-trial MAPF ensemble benchmark: pooled routes, random subsets, aggregated metrics."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from social_path_planning.benchmark_sparse_snapshots import (
    astar_path_cache_meta,
    get_or_compute_astar_paths,
    load_or_generate_path_bank,
)
from social_path_planning.benchmark_sparse_snapshots import (
    _try_load_cached_astar_paths as try_load_cached_astar_paths,
)
from social_path_planning.mapf_comparison.motion import MotionConfig
from social_path_planning.mapf_comparison.pipeline import (
    MapfRunConfig,
    prepare_mapf_problem,
    run_mapf_solvers,
    run_milp_solver,
)

METHODS = ("cbs", "pp_path_length", "milp_soc", "milp_makespan")


def sample_route_subset(
    pool: Sequence[dict],
    num_agents: int,
    rng: np.random.Generator,
) -> Tuple[List[dict], List[int]]:
    """Sample ``num_agents`` routes without replacement; return routes and route ids."""
    if num_agents > len(pool):
        raise ValueError(f"num_agents={num_agents} exceeds pool size {len(pool)}")
    indices = rng.choice(len(pool), size=num_agents, replace=False)
    indices = sorted(int(i) for i in indices)
    routes = [pool[i] for i in indices]
    route_ids = [int(r["route_id"]) for r in routes]
    return routes, route_ids


def load_cached_milp_routes(
    routes: Sequence[dict],
    mode: str,
    path_bank_path,
    *,
    social_graph=None,
    heatmap_prefix=None,
    heatmap_file=None,
) -> List[dict]:
    """
    Load MILP geometry from the path-bank cache only.

    Raises RuntimeError if cache is missing (pool prep should have populated it).
    """
    cache_meta = astar_path_cache_meta(
        mode,
        social_graph,
        heatmap_prefix=heatmap_prefix,
        heatmap_file=heatmap_file,
    )
    if mode == "vanilla":
        return list(routes)
    cached = try_load_cached_astar_paths(routes, path_bank_path, mode, cache_meta)
    if cached is None:
        raise RuntimeError(
            f"Missing cached A* paths for mode={mode!r} in {path_bank_path}. "
            "Run pool preparation (--pregen-only) or use --refresh-social-bank."
        )
    return cached


def ensure_modified_path_pool(
    occ_grid,
    statespace_hi,
    map_resolution,
    scenario_name: str,
    pool_size: int,
    benchmark_seed: int,
    max_attempts: int,
    path_bank_path,
    milp_astar_mode: str = "modified",
    *,
    social_graph=None,
    heatmap_prefix=None,
    heatmap_file=None,
    refresh: bool = False,
) -> Tuple[List[dict], dict]:
    """
    Load or generate a route pool and ensure modified (or other) A* polylines are cached.

    Returns ``(pool, meta)`` where ``meta['pool_astar_build_s']`` is wall time spent
    computing A* paths during this call (0 when fully cache-hit).
    """
    pool, bank_meta = load_or_generate_path_bank(
        occ_grid=occ_grid,
        statespace_hi=statespace_hi,
        map_resolution=map_resolution,
        scenario_name=scenario_name,
        num_robots=pool_size,
        benchmark_seed=benchmark_seed,
        max_attempts=max_attempts,
        path_bank_path=path_bank_path,
    )

    t0 = time.perf_counter()
    if milp_astar_mode != "vanilla":
        get_or_compute_astar_paths(
            occ_grid,
            statespace_hi,
            map_resolution,
            pool,
            milp_astar_mode,
            path_bank_path,
            social_graph=social_graph,
            heatmap_prefix=heatmap_prefix,
            heatmap_file=heatmap_file,
            refresh=refresh,
        )
    pool_astar_build_s = float(time.perf_counter() - t0)

    meta = {
        **bank_meta,
        "pool_size": int(pool_size),
        "pool_astar_build_s": pool_astar_build_s,
        "milp_astar_mode": milp_astar_mode,
    }
    return pool, meta


def _trial_row(
    trial_id: int,
    trial_seed: int,
    route_ids: Sequence[int],
    method: str,
    metrics: dict,
    *,
    extra: dict | None = None,
) -> dict:
    row = {
        "trial_id": int(trial_id),
        "trial_seed": int(trial_seed),
        "sampled_route_ids": ",".join(str(r) for r in route_ids),
        "num_agents": len(route_ids),
        "method": method,
        "success": metrics.get("success"),
        "status": metrics.get("status"),
        "solver_runtime_s": metrics.get("solver_runtime_s", metrics.get("runtime_s")),
        "runtime_s": metrics.get("runtime_s"),
        "soc_seconds": metrics.get("soc_seconds"),
        "makespan_seconds": metrics.get("makespan_seconds"),
        "total_path_length_m": metrics.get("total_path_length_m"),
        "conflict_count": metrics.get("conflict_count"),
    }
    if extra:
        row.update(extra)
    return row


def run_ensemble_trial(
    trial_id: int,
    trial_seed: int,
    pool: Sequence[dict],
    num_agents: int,
    occ_grid,
    mapf_cfg: MapfRunConfig,
    motion: MotionConfig,
    path_bank_path,
    milp_astar_mode: str = "modified",
    *,
    social_graph=None,
    heatmap_prefix=None,
    heatmap_file=None,
) -> List[dict]:
    """Run one ensemble trial; return detailed rows for all methods."""
    rng = np.random.default_rng(trial_seed)
    stage_records, route_ids = sample_route_subset(pool, num_agents, rng)

    grid_config, starts, goals, budget = prepare_mapf_problem(
        occ_grid, stage_records, mapf_cfg
    )
    solver_results = run_mapf_solvers(
        occ_grid,
        stage_records,
        grid_config,
        starts,
        goals,
        budget,
        mapf_cfg,
        verbose=False,
    )

    rows: List[dict] = []

    if "cbs" in solver_results:
        rows.append(
            _trial_row(
                trial_id,
                trial_seed,
                route_ids,
                "cbs",
                solver_results["cbs"]["metrics"],
            )
        )
    if "pp" in solver_results:
        pp_entry = solver_results["pp"]
        rows.append(
            _trial_row(
                trial_id,
                trial_seed,
                route_ids,
                "pp_path_length",
                pp_entry["metrics"],
                extra={"priority_order": pp_entry.get("priority_order")},
            )
        )

    milp_routes = load_cached_milp_routes(
        stage_records,
        milp_astar_mode,
        path_bank_path,
        social_graph=social_graph,
        heatmap_prefix=heatmap_prefix,
        heatmap_file=heatmap_file,
    )

    milp_soc = run_milp_solver(
        occ_grid, milp_routes, norm=1, motion=motion, verbose=False
    )
    rows.append(
        _trial_row(
            trial_id,
            trial_seed,
            route_ids,
            "milp_soc",
            milp_soc["metrics"],
        )
    )

    milp_ms = run_milp_solver(
        occ_grid, milp_routes, norm=np.inf, motion=motion, verbose=False
    )
    rows.append(
        _trial_row(
            trial_id,
            trial_seed,
            route_ids,
            "milp_makespan",
            milp_ms["metrics"],
        )
    )

    return rows


def aggregate_ensemble_metrics(trial_rows: Sequence[dict]) -> List[dict]:
    """
    Aggregate per-method statistics over trials.

    Averages for solver time and makespan use successful trials only.
    """
    by_method: Dict[str, List[dict]] = {m: [] for m in METHODS}
    for row in trial_rows:
        method = row.get("method")
        if method in by_method:
            by_method[method].append(row)

    summary: List[dict] = []
    for method in METHODS:
        rows = by_method[method]
        if not rows:
            continue
        num_trials = len(rows)
        successes = [r for r in rows if r.get("success")]
        failures = [r for r in rows if not r.get("success")]
        success_count = len(successes)
        failure_count = len(failures)
        success_rate = float(success_count / num_trials) if num_trials else 0.0

        solver_times = [
            float(r["solver_runtime_s"])
            for r in successes
            if r.get("solver_runtime_s") is not None
        ]
        makespans = [
            float(r["makespan_seconds"])
            for r in successes
            if r.get("makespan_seconds") is not None
        ]

        entry: Dict[str, Any] = {
            "method": method,
            "num_trials": num_trials,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "avg_solver_runtime_s": float(np.mean(solver_times)) if solver_times else None,
            "std_solver_runtime_s": float(np.std(solver_times)) if len(solver_times) > 1 else None,
            "avg_makespan_seconds": float(np.mean(makespans)) if makespans else None,
            "std_makespan_seconds": float(np.std(makespans)) if len(makespans) > 1 else None,
        }
        summary.append(entry)
    return summary


def print_ensemble_summary(summary_rows: Sequence[dict], pool_astar_build_s: float) -> None:
    """Print a compact table for console reporting."""
    print(f"\nPool A* build time (excluded from MILP trial averages): {pool_astar_build_s:.2f}s")
    print(f"{'Method':<18} {'Success%':>9} {'Avg solver s':>13} {'Avg makespan s':>15} {'Failures':>9}")
    print("-" * 68)
    for row in summary_rows:
        sr = row.get("success_rate")
        sr_pct = f"{100.0 * sr:.1f}" if sr is not None else "n/a"
        avg_rt = row.get("avg_solver_runtime_s")
        avg_ms = row.get("avg_makespan_seconds")
        rt_str = f"{avg_rt:.3f}" if avg_rt is not None else "n/a"
        ms_str = f"{avg_ms:.3f}" if avg_ms is not None else "n/a"
        print(
            f"{row['method']:<18} {sr_pct:>8}% {rt_str:>13} {ms_str:>15} "
            f"{row.get('failure_count', 0):>9}"
        )
