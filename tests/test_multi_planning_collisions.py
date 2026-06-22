"""Tests for segment-interval collision detection and constraints in multi_planning."""

import sys
import types
import unittest
from unittest import mock

import numpy as np

if "cvxpy" not in sys.modules:
    _cp_mock = types.ModuleType("cvxpy")
    _cp_mock.Variable = mock.MagicMock()
    _cp_mock.Minimize = mock.MagicMock()
    _cp_mock.Problem = mock.MagicMock()
    _cp_mock.hstack = mock.MagicMock()
    _cp_mock.norm = mock.MagicMock()
    sys.modules["cvxpy"] = _cp_mock

from social_path_planning.multi_planning import (
    DELTA,
    ROBOT_DIAMETER,
    MultiAgentSequentialPlanner,
    MultiAgentSimultaneousPlanner,
    _assign_z_to_segment_pairs,
    _collect_segment_collision_pairs,
    _iter_conflicting_segment_pairs,
    _parse_collision_entry,
    _same_direction_bands,
    _segment_endpoint_conflict,
)


class _DummyGrid:
    pass


class TestSegmentConflictHelpers(unittest.TestCase):
    def test_segment_endpoint_conflict_shared_edge(self):
        p0, p1 = (0.0, 0.0), (1.0, 0.0)
        q0, q1 = (1.0, 0.0), (1.0, 1.0)
        self.assertTrue(_segment_endpoint_conflict(p0, p1, q0, q1, ROBOT_DIAMETER))

    def test_iter_conflicting_segment_pairs(self):
        path_a = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        path_b = [(1.0, 0.0), (1.0, 1.0)]
        pairs = list(_iter_conflicting_segment_pairs(path_a, path_b, ROBOT_DIAMETER))
        self.assertIn((0, 0), pairs)

    def test_z_reuse_consecutive_shared_vertex_segments(self):
        path_a = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        path_b = [(0.0, 0.0), (2.0, 0.0)]
        pairs = []
        num_z = _collect_segment_collision_pairs(pairs, path_a, path_b, ROBOT_DIAMETER)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(num_z, 1)
        self.assertEqual(pairs[0][-2], pairs[1][-2])

    def test_same_direction_bands_diagonal_chain(self):
        pair_set = {(0, 1), (1, 2), (2, 3), (5, 5)}
        bands = _same_direction_bands(pair_set)
        self.assertEqual(len(bands), 1)
        self.assertEqual(bands[0], {(0, 1), (1, 2), (2, 3)})

    def test_co_increment_diagonal_band_uses_one_z(self):
        segment_pairs = [(0, 1), (1, 2), (2, 3)]
        assigned, z_index = _assign_z_to_segment_pairs(segment_pairs)
        z_vals = {z for _i, _j, z in assigned}
        self.assertEqual(z_vals, {0})
        self.assertEqual(z_index, 1)

    def test_parallel_corridor_marching_band_uses_one_z(self):
        path_a = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
        path_b = [(0.0, 0.1), (1.0, 0.1), (2.0, 0.1), (3.0, 0.1), (4.0, 0.1)]
        pairs = []
        num_z = _collect_segment_collision_pairs(pairs, path_a, path_b, ROBOT_DIAMETER)
        z_vals = {entry[-2] for entry in pairs}
        self.assertEqual(len(pairs), 10)
        self.assertEqual(num_z, 1)
        self.assertEqual(z_vals, {0})


class TestSequentialSegmentPairs(unittest.TestCase):
    def test_emits_segment_indices_not_waypoint_only(self):
        ego = [(0.0, 0.0), (1.0, 0.0)]
        other = [(1.0, 0.0), (1.0, 1.0)]
        other_times = [0.0, 5.0]
        planner = MultiAgentSequentialPlanner(
            _DummyGrid(), [other], [other_times], path=ego, v=1.0
        )
        pairs, num_z = planner.find_collision_pairs()
        self.assertGreater(num_z, 0)
        self.assertGreaterEqual(len(pairs[0]), 4)
        _leading, i, j, _z, _mode = _parse_collision_entry(pairs[0])
        _other_idx = _leading[0]
        self.assertEqual(i, 0)
        self.assertGreaterEqual(j, 0)


class TestIntervalOverlapRegression(unittest.TestCase):
    """Occupancy [10,50] on a shared edge must block the other agent during interior times."""

    def test_shared_edge_detected_for_interval_constraints(self):
        path1 = [(0.0, 0.0), (10.0, 0.0)]
        path2 = [(10.0, 0.0), (10.0, 5.0)]
        planner = MultiAgentSimultaneousPlanner(_DummyGrid(), paths=[path1, path2], v=[1.0, 1.0])
        pairs, num_z = planner.find_collision_pairs()
        self.assertGreater(num_z, 0)
        self.assertTrue(
            any(
                i == 0 and j == 0
                for entry in pairs
                for _leading, i, j, _z, _mode in [_parse_collision_entry(entry)]
            )
        )

    def test_opposite_corridor_window_uses_boundary_constraints(self):
        path_a = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        path_b = [(3.0, 0.0), (2.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
        pairs = []
        _collect_segment_collision_pairs(pairs, path_a, path_b, ROBOT_DIAMETER)
        boundary = [
            _parse_collision_entry(p)
            for p in pairs
            if _parse_collision_entry(p)[4] in ("a_first", "b_first")
        ]
        self.assertEqual(len(boundary), 2)
        self.assertEqual(boundary[0][3], boundary[1][3])
        self.assertEqual({row[4] for row in boundary}, {"a_first", "b_first"})

    def test_interval_constraints_exclude_overlapping_occupancy(self):
        try:
            import cvxpy as cp
            if isinstance(cp, types.ModuleType) and isinstance(cp.Variable, mock.MagicMock):
                self.skipTest("cvxpy not installed")
        except ImportError:
            self.skipTest("cvxpy not installed")

        t1_0, t1_1 = 10.0, 50.0
        t2_0, t2_1 = 20.0, 30.0
        z = cp.Variable(1, boolean=True)
        t2 = cp.Variable(2)
        constraints = [
            t2[0] == t2_0,
            t2[1] == t2_1,
            t2[1] <= t1_0 - DELTA + 1e6 * z[0],
            t1_1 <= t2[0] - DELTA - 1e6 * (1 - z[0]),
        ]
        mid = (t2[0] + t2[1]) / 2
        constraints.append(mid >= 25)
        constraints.append(mid <= 25)
        prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve(solver=cp.GLPK_MI)
        self.assertIn(prob.status, ("infeasible", "infeasible_inaccurate"))


if __name__ == "__main__":
    unittest.main()
