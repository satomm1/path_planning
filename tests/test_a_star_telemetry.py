"""Tests for A* solve telemetry (import from installed ``social_path_planning``)."""

import sys
from pathlib import Path
from unittest import TestCase

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from social_path_planning.a_star import AStar, AStar_With_Graph
from social_path_planning.occupancy_grid import StochOccupancyGrid2D


def _open_grid(n, res, origin_x=0.0, origin_y=0.0):
    probs = np.zeros((n, n), dtype=float)
    return StochOccupancyGrid2D(res, n, n, origin_x, origin_y, 10, probs)


class _TinyDiGraph:
    def __init__(self):
        self._adj = {}

    def add_edge(self, u, v, weight):
        self._adj.setdefault(u, {})[v] = {"weight": float(weight)}

    def has_edge(self, u, v):
        return v in self._adj.get(u, {})

    def __getitem__(self, u):
        return self._adj[u]


class TestTelemetryVanillaVsModified(TestCase):
    def test_vanilla_has_no_cost_evals(self):
        occ = _open_grid(12, 1.0)
        hi = (12.0, 12.0)
        p = AStar([0, 0], hi, (1.0, 5.0), (10.0, 5.0), occ, resolution=1.0)
        ok, telem = p.solve(mode="vanilla", return_telemetry=True)
        self.assertTrue(ok)
        self.assertEqual(telem["mode"], "vanilla")
        self.assertEqual(telem["cost_eval_count"], 0)
        self.assertAlmostEqual(telem["social_cost_sum"], 0.0)

    def test_modified_records_cost_evals(self):
        occ = _open_grid(12, 1.0)
        hi = (12.0, 12.0)
        p = AStar([0, 0], hi, (1.0, 5.0), (10.0, 5.0), occ, resolution=1.0)
        ok, telem = p.solve(mode="modified", return_telemetry=True)
        self.assertTrue(ok)
        self.assertEqual(telem["mode"], "modified")
        self.assertGreater(telem["cost_eval_count"], 0)
        self.assertIsNotNone(telem["path_geometric_length_m"])
        self.assertGreater(telem["path_geometric_length_m"], 0.0)

    def test_return_timing_and_telemetry_tuple(self):
        occ = _open_grid(12, 1.0)
        hi = (12.0, 12.0)
        p = AStar([0, 0], hi, (1.0, 5.0), (10.0, 5.0), occ, resolution=1.0)
        ok, elapsed, telem = p.solve(
            mode="modified", return_timing=True, return_telemetry=True
        )
        self.assertTrue(ok)
        self.assertGreater(elapsed, 0.0)
        self.assertIn("cost_eval_count", telem)

    def test_log_last_solve_telemetry_ros_noops_without_ros(self):
        occ = _open_grid(8, 1.0)
        hi = (8.0, 8.0)
        p = AStar([0, 0], hi, (1.0, 3.0), (6.0, 3.0), occ, resolution=1.0)
        p.solve(mode="modified")
        p.log_last_solve_telemetry_ros()


class TestGraphPlannerTelemetry(TestCase):
    def test_graph_edge_used_on_corridor(self):
        occ = _open_grid(20, 1.0, 0.0, 0.0)
        hi = (20.0, 20.0)
        g = _TinyDiGraph()
        for cx in range(2, 12):
            g.add_edge((cx, 10), (cx + 1, 10), weight=1.0)
        p = AStar_With_Graph([0, 0], hi, (2.0, 10.0), (12.0, 10.0), occ, g, resolution=1.0)
        ok, telem = p.solve(mode="modified", return_telemetry=True)
        self.assertTrue(ok)
        self.assertGreater(telem["graph_edge_cost_evals"], 0)
        self.assertGreaterEqual(telem["path_fraction_on_graph"], 0.5)
