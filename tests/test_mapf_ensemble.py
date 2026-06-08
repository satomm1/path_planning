"""Tests for MAPF ensemble benchmark helpers."""

import numpy as np
import pytest

from social_path_planning.mapf_comparison.ensemble import (
    aggregate_ensemble_metrics,
    sample_route_subset,
)


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


def test_aggregate_ensemble_metrics_success_only_averages():
    trial_rows = [
        {
            "method": "cbs",
            "success": True,
            "solver_runtime_s": 1.0,
            "makespan_seconds": 10.0,
        },
        {
            "method": "cbs",
            "success": False,
            "solver_runtime_s": 99.0,
            "makespan_seconds": None,
        },
        {
            "method": "cbs",
            "success": True,
            "solver_runtime_s": 3.0,
            "makespan_seconds": 20.0,
        },
        {
            "method": "milp_makespan",
            "success": True,
            "solver_runtime_s": 0.5,
            "makespan_seconds": 8.0,
        },
        {
            "method": "milp_makespan",
            "success": False,
            "solver_runtime_s": 0.2,
            "makespan_seconds": None,
        },
    ]
    summary = {row["method"]: row for row in aggregate_ensemble_metrics(trial_rows)}

    cbs = summary["cbs"]
    assert cbs["num_trials"] == 3
    assert cbs["success_count"] == 2
    assert cbs["failure_count"] == 1
    assert cbs["success_rate"] == pytest.approx(2 / 3)
    assert cbs["avg_solver_runtime_s"] == pytest.approx(2.0)
    assert cbs["avg_makespan_seconds"] == pytest.approx(15.0)

    milp = summary["milp_makespan"]
    assert milp["num_trials"] == 2
    assert milp["failure_count"] == 1
    assert milp["avg_solver_runtime_s"] == pytest.approx(0.5)
    assert milp["avg_makespan_seconds"] == pytest.approx(8.0)


def test_aggregate_ensemble_metrics_empty_methods_omitted():
    trial_rows = [
        {
            "method": "pp_path_length",
            "success": True,
            "solver_runtime_s": 2.0,
            "makespan_seconds": 5.0,
        },
    ]
    summary = aggregate_ensemble_metrics(trial_rows)
    methods = {row["method"] for row in summary}
    assert methods == {"pp_path_length"}
