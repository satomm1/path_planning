"""Tests for RRT* planners and segment collision checking."""

import sys
from pathlib import Path
from unittest import TestCase

import networkx as nx
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.rrt_star import RRTStar, RRTStar_With_Graph


def _open_grid(n, res, origin_x=0.0, origin_y=0.0):
    probs = np.zeros((n, n), dtype=float)
    return StochOccupancyGrid2D(res, n, n, origin_x, origin_y, 10, probs)


def _corridor_grid(n, res):
    probs = np.zeros((n, n), dtype=float)
    probs[0:3, :] = 1.0
    probs[-3:, :] = 1.0
    return StochOccupancyGrid2D(res, n, n, 0.0, 0.0, 10, probs)


class TestSegmentCollision(TestCase):
    def test_segment_through_obstacle_is_not_free(self):
        occ = StochOccupancyGrid2D(1.0, 20, 20, 0.0, 0.0, 10, np.zeros((20, 20)), robot_d=2.0)
        occ.probs[:, 8:12] = 1.0
        self.assertTrue(occ.is_free((4.0, 10.0)))
        self.assertTrue(occ.is_free((16.0, 10.0)))
        self.assertFalse(occ.is_segment_free((4.0, 10.0), (16.0, 10.0)))

    def test_open_segment_is_free(self):
        occ = _open_grid(12, 1.0)
        self.assertTrue(occ.is_segment_free((1.0, 5.0), (10.0, 5.0)))


class TestRRTStarOpenMap(TestCase):
    def test_vanilla_rrt_finds_path(self):
        occ = _open_grid(20, 1.0)
        hi = (20.0, 20.0)
        rng = np.random.default_rng(0)
        planner = RRTStar(
            [0, 0],
            hi,
            (2.0, 10.0),
            (18.0, 10.0),
            occ,
            resolution=1.0,
            max_iterations=1000,
            goal_sample_rate=0.20,
        )
        ok, elapsed = planner.solve(mode="vanilla", return_timing=True, rng=rng)
        self.assertTrue(ok)
        self.assertGreater(elapsed, 0.0)
        self.assertIsNotNone(planner.path)
        self.assertGreaterEqual(len(planner.path), 2)
        self.assertAlmostEqual(planner.path[0][0], 2.0)
        self.assertAlmostEqual(planner.path[0][1], 10.0)
        self.assertAlmostEqual(planner.path[-1][0], 18.0)
        self.assertAlmostEqual(planner.path[-1][1], 10.0)
        for i in range(len(planner.path) - 1):
            self.assertTrue(occ.is_segment_free(planner.path[i], planner.path[i + 1]))


class TestRRTStarSocial(TestCase):
    def test_modified_rrt_runs_in_corridor(self):
        occ = _corridor_grid(30, 1.0)
        hi = (30.0, 30.0)
        rng = np.random.default_rng(1)
        vanilla = RRTStar(
            [0, 0],
            hi,
            (4.0, 15.0),
            (26.0, 15.0),
            occ,
            resolution=1.0,
            max_iterations=2000,
            goal_sample_rate=0.20,
        )
        modified = RRTStar(
            [0, 0],
            hi,
            (4.0, 15.0),
            (26.0, 15.0),
            occ,
            resolution=1.0,
            max_iterations=2000,
            goal_sample_rate=0.20,
        )
        ok_v, _ = vanilla.solve(mode="vanilla", return_timing=True, rng=rng)
        ok_m, _ = modified.solve(mode="modified", return_timing=True, rng=np.random.default_rng(1))
        self.assertTrue(ok_v)
        self.assertTrue(ok_m)
        self.assertIsNotNone(vanilla.path)
        self.assertIsNotNone(modified.path)


class TestRRTStarWithGraph(TestCase):
    def test_graph_variant_smoke(self):
        occ = _open_grid(20, 1.0)
        hi = (20.0, 20.0)
        graph = nx.DiGraph()
        for cx in range(2, 12):
            graph.add_edge((cx, 10), (cx + 1, 10), weight=1.0)
        planner = RRTStar_With_Graph(
            [0, 0],
            hi,
            (2.0, 10.0),
            (12.0, 10.0),
            occ,
            graph,
            resolution=1.0,
            max_iterations=1500,
            goal_sample_rate=0.25,
        )
        ok, _ = planner.solve(mode="modified", return_timing=True, rng=np.random.default_rng(2))
        self.assertTrue(ok)
        self.assertIsNotNone(planner.path)
