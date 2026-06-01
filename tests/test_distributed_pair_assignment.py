"""Tests for ring-based distributed pair assignment."""

import unittest

from social_path_planning.distributed_pair_assignment import (
    assigned_pairs_for_fleet_index,
    assigned_pairs_for_robot_id,
    expected_pair_count,
    expected_pair_partition,
    supports_distributed_assignment,
    validate_full_coverage,
)


class TestDistributedPairAssignment(unittest.TestCase):
    def test_expected_pair_count(self):
        self.assertEqual(expected_pair_count(2), 1)
        self.assertEqual(expected_pair_count(4), 6)
        self.assertEqual(expected_pair_count(5), 10)

    def test_even_n_matches_benchmark_eval(self):
        """N=4 assignment matches evaluate_distributed_constraint_timing.py."""
        self.assertEqual(assigned_pairs_for_fleet_index(0, 4), [(0, 1), (0, 2)])
        self.assertEqual(assigned_pairs_for_fleet_index(1, 4), [(1, 2), (1, 3)])
        self.assertEqual(assigned_pairs_for_fleet_index(2, 4), [(2, 3)])
        self.assertEqual(assigned_pairs_for_fleet_index(3, 4), [(0, 3)])

    def test_validate_full_coverage_even(self):
        for n in (2, 4, 6, 8):
            fleet = list(range(n))
            validate_full_coverage(fleet)

    def test_validate_full_coverage_odd(self):
        for n in (3, 5, 7):
            fleet = list(range(n))
            validate_full_coverage(fleet)

    def test_assigned_pairs_for_robot_id(self):
        fleet = [10, 20, 30, 40]
        pairs = assigned_pairs_for_robot_id(20, fleet)
        self.assertEqual(pairs, [(1, 2), (1, 3)])

    def test_expected_pair_partition(self):
        fleet = [5, 7, 9]
        partition = expected_pair_partition(fleet)
        self.assertEqual(set(partition.keys()), {5, 7, 9})
        all_pairs = [p for pairs in partition.values() for p in pairs]
        self.assertEqual(len(set(all_pairs)), expected_pair_count(3))

    def test_supports_distributed_assignment(self):
        self.assertTrue(supports_distributed_assignment([1, 2, 3, 4]))
        self.assertFalse(supports_distributed_assignment([1, 2, 3], allow_odd_fleet=False))
        self.assertTrue(supports_distributed_assignment([1, 2, 3], allow_odd_fleet=True))


if __name__ == "__main__":
    unittest.main()
