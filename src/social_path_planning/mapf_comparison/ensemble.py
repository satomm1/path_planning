"""Multi-trial MAPF ensemble benchmark: pooled routes, random subsets, aggregated metrics."""

from __future__ import annotations

import time
import traceback
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
from social_path_planning.mapf_comparison.checkpoint import coerce_bool, normalize_trial_row
from social_path_planning.mapf_comparison.motion import MotionConfig
from social_path_planning.mapf_comparison.pipeline import (
    MapfRunConfig,
    _format_solver_error_status,
    prepare_mapf_problem,
    run_event_milp_solver,
    run_mapf_solvers,
)

from social_path_planning.mapf_comparison.constants import MAPF_METHODS, METHODS


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
    sparse_graph_threshold: float | None = None,
    sparse_min_component_size: int | None = None,
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
        sparse_graph_threshold=sparse_graph_threshold,
        sparse_min_component_size=sparse_min_component_size,
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
    sparse_graph_threshold: float | None = None,
    sparse_min_component_size: int | None = None,
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
        print(
            f"Caching MILP geometry for {len(pool)} routes (mode={milp_astar_mode!r})...",
            flush=True,
        )
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
            max_resample_attempts=max_attempts,
            resample_seed=int(benchmark_seed) + 1_000_003,
            sparse_graph_threshold=sparse_graph_threshold,
            sparse_min_component_size=sparse_min_component_size,
        )
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
        "soc_timesteps": metrics.get("soc_timesteps"),
        "makespan_timesteps": metrics.get("makespan_timesteps"),
        "total_path_length_m": metrics.get("total_path_length_m"),
        "conflict_count": metrics.get("conflict_count"),
        "error_type": metrics.get("error_type"),
        "error_message": metrics.get("error_message"),
    }
    if extra:
        row.update(extra)
    return row


def _error_metrics_for_method(method: str, exc: BaseException, *, motion: MotionConfig) -> dict:
    status = _format_solver_error_status(exc)
    if method in MAPF_METHODS:
        return aggregate_mapf_metrics(
            [],
            0,
            0.0,
            0.0,
            status,
            downsample=1,
            motion=motion,
        )
    return aggregate_milp_metrics(
        None,
        0.0,
        status,
        norm=1,
        motion=motion,
    )


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
    milp_stride: int = 1,
    run_event_milp: bool = True,
    print_event_milp_constraints: bool = False,
    print_event_milp_constraints_on_error: bool = True,
    fail_soft: bool = True,
    sparse_graph_threshold: float | None = None,
    sparse_min_component_size: int | None = None,
    methods_to_run: set[str] | None = None,
    *,
    social_graph=None,
    heatmap_prefix=None,
    heatmap_file=None,
) -> List[dict]:
    """Run one ensemble trial; return detailed rows for all methods."""
    active_methods = set(METHODS) if methods_to_run is None else set(methods_to_run)
    run_cbs = "cbs" in active_methods
    run_pp = "pp_path_length" in active_methods
    run_event = run_event_milp and "event_milp_soc" in active_methods

    rng = np.random.default_rng(trial_seed)
    stage_records, route_ids = sample_route_subset(pool, num_agents, rng)
    rows: List[dict] = []

    if run_cbs or run_pp:
        trial_mapf_cfg = MapfRunConfig(
            downsample=mapf_cfg.downsample,
            crop_padding_cells=mapf_cfg.crop_padding_cells,
            coarse_block_policy=mapf_cfg.coarse_block_policy,
            pp_low_level_max_iter=mapf_cfg.pp_low_level_max_iter,
            cbs_low_level_max_iter=mapf_cfg.cbs_low_level_max_iter,
            cbs_max_iter=mapf_cfg.cbs_max_iter,
            cbs_max_process=mapf_cfg.cbs_max_process,
            motion=mapf_cfg.motion,
            run_cbs=run_cbs,
            run_pp=run_pp,
            crop=mapf_cfg.crop,
            cbs_timeout_s=mapf_cfg.cbs_timeout_s,
            pp_timeout_s=mapf_cfg.pp_timeout_s,
        )
        try:
            grid_config, starts, goals, budget = prepare_mapf_problem(
                occ_grid, stage_records, trial_mapf_cfg
            )
            solver_results = run_mapf_solvers(
                occ_grid,
                stage_records,
                grid_config,
                starts,
                goals,
                budget,
                trial_mapf_cfg,
                verbose=False,
            )
        except Exception as exc:
            if not fail_soft:
                raise
            print(
                f"MAPF setup/solve failed for trial_id={trial_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            for method in MAPF_METHODS:
                if method not in active_methods:
                    continue
                metrics = _error_metrics_for_method(method, exc, motion=motion)
                metrics["error_type"] = type(exc).__name__
                metrics["error_message"] = str(exc)
                rows.append(
                    _trial_row(trial_id, trial_seed, route_ids, method, metrics)
                )
            solver_results = {}
        else:
            if run_cbs and "cbs" in solver_results:
                rows.append(
                    _trial_row(
                        trial_id,
                        trial_seed,
                        route_ids,
                        "cbs",
                        solver_results["cbs"]["metrics"],
                    )
                )
            if run_pp and "pp" in solver_results:
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

    if run_event:
        event_context = {
            "trial_id": trial_id,
            "trial_seed": trial_seed,
            "route_ids": list(route_ids),
            "num_agents": len(route_ids),
            "milp_stride": milp_stride,
        }
        if print_event_milp_constraints:
            print(
                f"\n=== Event MILP constraints "
                f"(trial_id={trial_id}, trial_seed={trial_seed}, route_ids={route_ids}) ===",
                flush=True,
            )
        try:
            milp_routes = load_cached_milp_routes(
                stage_records,
                milp_astar_mode,
                path_bank_path,
                social_graph=social_graph,
                heatmap_prefix=heatmap_prefix,
                heatmap_file=heatmap_file,
                sparse_graph_threshold=sparse_graph_threshold,
                sparse_min_component_size=sparse_min_component_size,
            )
            event_soc = run_event_milp_solver(
                occ_grid,
                milp_routes,
                norm=1,
                motion=motion,
                stride=milp_stride,
                verbose=False,
                print_constraints=print_event_milp_constraints,
                print_constraints_on_error=print_event_milp_constraints_on_error,
                error_context={**event_context, "method": "event_milp_soc", "norm": 1},
            )
            rows.append(
                _trial_row(
                    trial_id,
                    trial_seed,
                    route_ids,
                    "event_milp_soc",
                    event_soc["metrics"],
                    extra={
                        "interest_waypoint_count": event_soc.get("interest_waypoint_count"),
                        "encounter_count": event_soc.get("encounter_count"),
                        "binary_z_count": event_soc.get("binary_z_count"),
                    },
                )
            )
        except Exception as exc:
            if not fail_soft:
                raise
            print(
                f"Event MILP failed for trial_id={trial_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            metrics = _error_metrics_for_method("event_milp_soc", exc, motion=motion)
            metrics["error_type"] = type(exc).__name__
            metrics["error_message"] = str(exc)
            rows.append(
                _trial_row(
                    trial_id,
                    trial_seed,
                    route_ids,
                    "event_milp_soc",
                    metrics,
                )
            )

    return rows


def _aggregate_group(rows: Sequence[dict]) -> dict:
    normalized = [normalize_trial_row(row) for row in rows]
    num_trials = len(normalized)
    successes = [r for r in normalized if coerce_bool(r.get("success"))]
    failures = [r for r in normalized if not coerce_bool(r.get("success"))]
    timeouts = [r for r in normalized if r.get("status") == "timeout"]
    success_count = len(successes)
    failure_count = len(failures)
    timeout_count = len(timeouts)
    success_rate = float(success_count / num_trials) if num_trials else 0.0

    def _floats(key: str, source: Sequence[dict]) -> List[float]:
        out: List[float] = []
        for r in source:
            val = r.get(key)
            if val is not None and val != "":
                out.append(float(val))
        return out

    solver_times = _floats("solver_runtime_s", successes)
    makespans_s = _floats("makespan_seconds", successes)
    socs_s = _floats("soc_seconds", successes)
    makespans_t = _floats("makespan_timesteps", successes)
    socs_t = _floats("soc_timesteps", successes)
    path_lens = _floats("total_path_length_m", successes)

    def _mean(vals: List[float]) -> float | None:
        return float(np.mean(vals)) if vals else None

    def _std(vals: List[float]) -> float | None:
        return float(np.std(vals)) if len(vals) > 1 else None

    def _median(vals: List[float]) -> float | None:
        return float(np.median(vals)) if vals else None

    return {
        "num_trials": num_trials,
        "success_count": success_count,
        "failure_count": failure_count,
        "timeout_count": timeout_count,
        "success_rate": success_rate,
        "avg_solver_runtime_s": _mean(solver_times),
        "std_solver_runtime_s": _std(solver_times),
        "median_solver_runtime_s": _median(solver_times),
        "avg_makespan_seconds": _mean(makespans_s),
        "std_makespan_seconds": _std(makespans_s),
        "avg_soc_seconds": _mean(socs_s),
        "std_soc_seconds": _std(socs_s),
        "avg_makespan_timesteps": _mean(makespans_t),
        "std_makespan_timesteps": _std(makespans_t),
        "avg_soc_timesteps": _mean(socs_t),
        "std_soc_timesteps": _std(socs_t),
        "avg_total_path_length_m": _mean(path_lens),
        "std_total_path_length_m": _std(path_lens),
    }


def aggregate_ensemble_metrics(trial_rows: Sequence[dict]) -> List[dict]:
    """
    Aggregate statistics grouped by ``(num_agents, method)``.

    Averages use successful trials only.
    """
    groups: Dict[Tuple[int, str], List[dict]] = {}
    for row in trial_rows:
        row = normalize_trial_row(row)
        method = row.get("method")
        if method not in METHODS:
            continue
        key = (int(row.get("num_agents", 0)), str(method))
        groups.setdefault(key, []).append(row)

    summary: List[dict] = []
    for (num_agents, method) in sorted(groups.keys()):
        entry = {
            "num_agents": num_agents,
            "method": method,
            **_aggregate_group(groups[(num_agents, method)]),
        }
        summary.append(entry)
    return summary


def print_ensemble_summary(summary_rows: Sequence[dict], pool_astar_build_s: float) -> None:
    """Print a compact grid: agent count blocks with per-method stats."""
    print(f"\nPool A* build time (excluded from event MILP trial averages): {pool_astar_build_s:.2f}s")
    if not summary_rows:
        print("No summary rows.")
        return

    by_n: Dict[int, List[dict]] = {}
    for row in summary_rows:
        by_n.setdefault(int(row["num_agents"]), []).append(row)

    for num_agents in sorted(by_n):
        print(f"\n--- num_agents={num_agents} ---")
        print(
            f"{'Method':<18} {'Success%':>9} {'Timeouts':>9} {'Avg solver s':>13} "
            f"{'Avg cost':>12}"
        )
        print("-" * 72)
        for row in by_n[num_agents]:
            sr = row.get("success_rate")
            sr_pct = f"{100.0 * sr:.1f}" if sr is not None else "n/a"
            avg_rt = row.get("avg_solver_runtime_s")
            rt_str = f"{avg_rt:.3f}" if avg_rt is not None else "n/a"
            method = row["method"]
            if method in MAPF_METHODS:
                cost = row.get("avg_makespan_timesteps")
                cost_str = f"{cost:.1f} steps" if cost is not None else "n/a"
            else:
                cost = row.get("avg_makespan_seconds")
                cost_str = f"{cost:.3f} s" if cost is not None else "n/a"
            print(
                f"{method:<18} {sr_pct:>8}% {row.get('timeout_count', 0):>9} "
                f"{rt_str:>13} {cost_str:>12}"
            )
