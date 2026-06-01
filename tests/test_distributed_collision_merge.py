"""Tests for distributed collision detection merge vs centralized detection."""

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


class TestDistributedCollisionMerge(unittest.TestCase):
    def _sample_paths(self):
        return [
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            [(1.0, 0.0), (1.0, 1.0), (1.0, 2.0)],
            [(2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        ]

    def test_merge_matches_centralized_collision_pairs(self):
        paths = self._sample_paths()
        planner = MultiAgentSimultaneousPlanner(_DummyGrid(), paths=paths, v=[1.0, 1.0, 1.0])
        centralized_pairs, centralized_max_z = planner.find_collision_pairs()

        reports = []
        for a1 in range(len(paths)):
            for a2 in range(a1 + 1, len(paths)):
                _a1, _a2, seg_i, seg_j = detect_collision_pairs_for_agent_pair(
                    paths[a1], paths[a2], a1, a2, threshold=0.5
                )
                reports.append(
                    {"a1": _a1, "a2": _a2, "segment_i": seg_i, "segment_j": seg_j}
                )

        merged_pairs, merged_max_z = merge_collision_reports(paths, reports, threshold=0.5)
        self.assertEqual(centralized_max_z, merged_max_z)
        self.assertEqual(centralized_pairs, merged_pairs)

    def test_partitioned_detect_merge_matches_centralized(self):
        paths = self._sample_paths()
        fleet = [0, 1, 2]
        partition = expected_pair_partition(fleet)

        reports = []
        for _robot_id, assigned in partition.items():
            for a1, a2 in assigned:
                _a1, _a2, seg_i, seg_j = detect_collision_pairs_for_agent_pair(
                    paths[a1], paths[a2], a1, a2, threshold=0.5
                )
                reports.append(
                    {"a1": _a1, "a2": _a2, "segment_i": seg_i, "segment_j": seg_j}
                )

        planner = MultiAgentSimultaneousPlanner(_DummyGrid(), paths=paths, v=[1.0, 1.0, 1.0])
        centralized_pairs, centralized_max_z = planner.find_collision_pairs()
        merged_pairs, merged_max_z = merge_collision_reports(paths, reports, threshold=0.5)
        self.assertEqual(centralized_max_z, merged_max_z)
        self.assertEqual(centralized_pairs, merged_pairs)


if __name__ == "__main__":
    unittest.main()
