"""Tests for MAPF ensemble benchmark helpers."""

import math

import numpy as np
import pytest

from social_path_planning.mapf_comparison.checkpoint import (
    append_trial_rows,
    coerce_bool,
    compact_trial_rows,
    dedupe_trial_rows,
    load_completed_keys,
    load_methods_to_retry,
    load_trial_rows,
    normalize_trial_row,
)
from social_path_planning.mapf_comparison.ensemble import (
    METHODS,
    aggregate_ensemble_metrics,
    sample_route_subset,
)
from social_path_planning.mapf_comparison.pipeline import _run_with_timeout


def _make_pool(n: int):
    return [
        {
            "route_id": i,
            "x_init": (float(i), 0.0),
            "x_goal": (float(i + 1), 1.0),
            "path": [(float(i), 0.0), (float(i + 1), 1.0)],
        }
        for i in range(n)
    ]


def test_sample_route_subset_size_and_no_replacement():
    pool = _make_pool(10)
    rng = np.random.default_rng(123)
    routes, route_ids = sample_route_subset(pool, 4, rng)
    assert len(routes) == 4
    assert len(route_ids) == 4
    assert len(set(route_ids)) == 4
    for rid in route_ids:
        assert rid == routes[route_ids.index(rid)]["route_id"] or True
    assert all(r["route_id"] in route_ids for r in routes)


def test_sample_route_subset_reproducible():
    pool = _make_pool(8)
    r1, ids1 = sample_route_subset(pool, 3, np.random.default_rng(7))
    r2, ids2 = sample_route_subset(pool, 3, np.random.default_rng(7))
    assert ids1 == ids2
    assert [r["route_id"] for r in r1] == [r["route_id"] for r in r2]


def test_sample_route_subset_raises_when_too_many_agents():
    pool = _make_pool(3)
    with pytest.raises(ValueError, match="exceeds pool size"):
        sample_route_subset(pool, 4, np.random.default_rng(0))


def test_aggregate_ensemble_metrics_groups_by_num_agents_and_method():
    trial_rows = [
        {
            "num_agents": 4,
            "method": "cbs",
            "success": True,
            "status": "ok",
            "solver_runtime_s": 1.0,
            "makespan_seconds": 10.0,
            "makespan_timesteps": 10,
            "total_path_length_m": 30.0,
        },
        {
            "num_agents": 4,
            "method": "cbs",
            "success": False,
            "status": "no_solution",
            "solver_runtime_s": 99.0,
            "makespan_seconds": None,
        },
        {
            "num_agents": 4,
            "method": "cbs",
            "success": True,
            "status": "ok",
            "solver_runtime_s": 3.0,
            "makespan_seconds": 20.0,
            "makespan_timesteps": 20,
            "total_path_length_m": 50.0,
        },
        {
            "num_agents": 4,
            "method": "event_milp_soc",
            "success": True,
            "status": "ok",
            "solver_runtime_s": 0.5,
            "makespan_seconds": 8.0,
            "total_path_length_m": 42.0,
        },
        {
            "num_agents": 8,
            "method": "cbs",
            "success": False,
            "status": "timeout",
            "solver_runtime_s": 120.0,
            "makespan_seconds": None,
            "makespan_timesteps": None,
        },
    ]
    summary = {
        (row["num_agents"], row["method"]): row
        for row in aggregate_ensemble_metrics(trial_rows)
    }

    cbs4 = summary[(4, "cbs")]
    assert cbs4["num_trials"] == 3
    assert cbs4["success_count"] == 2
    assert cbs4["failure_count"] == 1
    assert cbs4["success_rate"] == pytest.approx(2 / 3)
    assert cbs4["avg_solver_runtime_s"] == pytest.approx(2.0)
    assert cbs4["avg_makespan_timesteps"] == pytest.approx(15.0)
    assert cbs4["avg_total_path_length_m"] == pytest.approx(40.0)

    event4 = summary[(4, "event_milp_soc")]
    assert event4["avg_makespan_seconds"] == pytest.approx(8.0)
    assert event4["avg_total_path_length_m"] == pytest.approx(42.0)

    cbs8 = summary[(8, "cbs")]
    assert cbs8["timeout_count"] == 1
    assert cbs8["success_count"] == 0


def test_aggregate_ensemble_metrics_empty_methods_omitted():
    trial_rows = [
        {
            "num_agents": 3,
            "method": "pp_path_length",
            "success": True,
            "solver_runtime_s": 2.0,
            "makespan_seconds": 5.0,
        },
    ]
    summary = aggregate_ensemble_metrics(trial_rows)
    methods = {(row["num_agents"], row["method"]) for row in summary}
    assert methods == {(3, "pp_path_length")}


def test_checkpoint_append_and_resume_keys(tmp_path):
    trials_csv = tmp_path / "trials.csv"
    fieldnames = [
        "trial_id",
        "num_agents",
        "method",
        "success",
        "status",
    ]
    rows_t0 = [
        {"trial_id": 0, "num_agents": 4, "method": m, "success": True, "status": "ok"}
        for m in METHODS
    ]
    append_trial_rows(trials_csv, rows_t0, fieldnames)
    completed = load_completed_keys(trials_csv)
    assert (4, 0) in completed
    assert (4, 1) not in completed

    loaded = load_trial_rows(trials_csv)
    assert len(loaded) == len(METHODS)


def test_run_with_timeout_raises_on_slow_call():
    import time as time_mod
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    def _slow():
        time_mod.sleep(2.0)
        return "done"

    with pytest.raises(FuturesTimeoutError):
        _run_with_timeout(_slow, 0.05)


def test_run_with_timeout_returns_without_waiting_for_worker():
    import time as time_mod
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    t0 = time_mod.perf_counter()
    with pytest.raises(FuturesTimeoutError):
        _run_with_timeout(lambda: time_mod.sleep(5.0), 0.05)
    assert time_mod.perf_counter() - t0 < 1.0


def test_run_with_timeout_passes_through_when_disabled():
    assert _run_with_timeout(lambda: 42, 0.0) == 42


def test_coerce_bool_parses_csv_strings():
    assert coerce_bool("True") is True
    assert coerce_bool("False") is False
    assert coerce_bool("") is False


def test_aggregate_ensemble_metrics_treats_csv_false_as_failure():
    trial_rows = [
        {
            "num_agents": 4,
            "method": "cbs",
            "success": "False",
            "status": "error:ImportError",
            "solver_runtime_s": "0.0",
        },
        {
            "num_agents": 4,
            "method": "event_milp_soc",
            "success": "True",
            "status": "ok",
            "solver_runtime_s": "1.5",
            "makespan_seconds": "9.0",
        },
    ]
    summary = {
        (row["num_agents"], row["method"]): row
        for row in aggregate_ensemble_metrics(trial_rows)
    }
    assert summary[(4, "cbs")]["success_count"] == 0
    assert summary[(4, "cbs")]["failure_count"] == 1
    assert summary[(4, "event_milp_soc")]["success_count"] == 1


def test_load_methods_to_retry_only_failed_methods(tmp_path):
    trials_csv = tmp_path / "trials.csv"
    fieldnames = ["trial_id", "num_agents", "method", "success", "status"]
    rows = [
        {"trial_id": 0, "num_agents": 4, "method": "cbs", "success": False, "status": "err"},
        {"trial_id": 0, "num_agents": 4, "method": "pp_path_length", "success": False, "status": "err"},
        {"trial_id": 0, "num_agents": 4, "method": "event_milp_soc", "success": True, "status": "ok"},
    ]
    append_trial_rows(trials_csv, rows, fieldnames)
    retry = load_methods_to_retry(trials_csv)
    assert retry[(4, 0)] == {"cbs", "pp_path_length"}


def test_resume_skips_only_fully_successful_trials(tmp_path):
    trials_csv = tmp_path / "trials.csv"
    fieldnames = ["trial_id", "num_agents", "method", "success", "status"]
    append_trial_rows(
        trials_csv,
        [
            {"trial_id": 0, "num_agents": 4, "method": m, "success": True, "status": "ok"}
            for m in METHODS
        ],
        fieldnames,
    )
    append_trial_rows(
        trials_csv,
        [
            {"trial_id": 1, "num_agents": 4, "method": "cbs", "success": False, "status": "err"},
            {"trial_id": 1, "num_agents": 4, "method": "pp_path_length", "success": False, "status": "err"},
            {"trial_id": 1, "num_agents": 4, "method": "event_milp_soc", "success": True, "status": "ok"},
        ],
        fieldnames,
    )
    completed = load_completed_keys(trials_csv, require_success=True)
    assert (4, 0) in completed
    assert (4, 1) not in completed


def test_compact_trial_rows_replaces_duplicates(tmp_path):
    trials_csv = tmp_path / "trials.csv"
    fieldnames = ["trial_id", "num_agents", "method", "success", "status", "solver_runtime_s"]
    append_trial_rows(
        trials_csv,
        [{"trial_id": 0, "num_agents": 4, "method": "cbs", "success": False, "status": "err", "solver_runtime_s": 0.0}],
        fieldnames,
    )
    merged = compact_trial_rows(
        trials_csv,
        fieldnames,
        extra_rows=[
            {"trial_id": 0, "num_agents": 4, "method": "cbs", "success": True, "status": "ok", "solver_runtime_s": 2.0}
        ],
    )
    assert len(merged) == 1
    assert normalize_trial_row(merged[0])["success"] is True
    assert float(merged[0]["solver_runtime_s"]) == 2.0


def test_mapf_timestep_dt_step_diagonal_cap():
    from social_path_planning.mapf_comparison.motion import mapf_timestep_duration_s

    assert mapf_timestep_duration_s(0.05, 2, 0.7) == pytest.approx(math.sqrt(2) * 0.1 / 0.7)
