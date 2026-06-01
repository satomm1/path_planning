"""Solver parity: centralized vs distributed-detect simultaneous planning."""

import sys
import types
import unittest
from unittest import mock

if "cvxpy" not in sys.modules:
    _cp_mock = types.ModuleType("cvxpy")
    _cp_mock.Variable = mock.MagicMock()
    _cp_mock.Minimize = mock.MagicMock()
    _cp_mock.Problem = mock.MagicMock()
    _cp_mock.hstack = mock.MagicMock()
    _cp_mock.norm = mock.MagicMock()
    sys.modules["cvxpy"] = _cp_mock

from social_path_planning.distributed_pair_assignment import expected_pair_partition
from social_path_planning.multi_planning import (
    MultiAgentSimultaneousPlanner,
    detect_collision_pairs_for_agent_pair,
    merge_collision_reports,
)


class _DummyGrid:
    pass


class TestDistributedSolverParity(unittest.TestCase):
    def setUp(self):
        try:
            import cvxpy as cp
            if isinstance(cp, types.ModuleType) and isinstance(cp.Variable, mock.MagicMock):
                self.skipTest("cvxpy not installed")
        except ImportError:
            self.skipTest("cvxpy not installed")

    def test_centralized_and_distributed_detect_produce_same_times(self):
        paths = [
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            [(1.0, 0.0), (1.0, 1.0), (1.0, 2.0)],
            [(2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        ]
        fleet = [0, 1, 2]
        v_list = [0.7, 0.7, 0.7]
        occ = _DummyGrid()

        centralized = MultiAgentSimultaneousPlanner(occ, paths=paths, norm=1, v=v_list)
        centralized_times = centralized.plan(verbose=False)

        reports = []
        partition = expected_pair_partition(fleet)
        for _robot_id, assigned in partition.items():
            for a1, a2 in assigned:
                _a1, _a2, seg_i, seg_j = detect_collision_pairs_for_agent_pair(
                    paths[a1], paths[a2], a1, a2, threshold=0.5
                )
                reports.append(
                    {"a1": a1, "a2": a2, "segment_i": seg_i, "segment_j": seg_j}
                )
        collision_pairs, max_z = merge_collision_reports(paths, reports, threshold=0.5)
        distributed = MultiAgentSimultaneousPlanner(occ, paths=paths, norm=1, v=v_list)
        distributed_times = distributed.plan_from_collision_pairs(collision_pairs, max_z, verbose=False)

        self.assertEqual(len(centralized_times), len(distributed_times))
        for c_row, d_row in zip(centralized_times, distributed_times):
            self.assertEqual(len(c_row), len(d_row))
            for c_t, d_t in zip(c_row, d_row):
                self.assertAlmostEqual(c_t, d_t, places=5)


if __name__ == "__main__":
    unittest.main()
