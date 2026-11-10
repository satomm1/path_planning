"""Parity tests: vectorized segment-conflict detection and linear-time encounter
clustering vs the original scalar implementations."""

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Planner modules are imported inside the tests (not at collection time): importing them
# here would load the real cvxpy before other test modules install their cvxpy stub,
# changing which tests in the suite run.


def _mp():
    import social_path_planning.multi_planning as mp

    return mp


def _em():
    import social_path_planning.event_multi_planning as em

    return em


def reference_pairs(path_a, path_b, threshold):
    """Original nested loop with per-endpoint np.linalg.norm checks."""
    out = []
    if len(path_a) < 2 or len(path_b) < 2:
        return out
    for i in range(len(path_a) - 1):
        for j in range(len(path_b) - 1):
            hit = False
            for pa in (path_a[i], path_a[i + 1]):
                for qb in (path_b[j], path_b[j + 1]):
                    if np.linalg.norm(np.array(pa) - np.array(qb)) <= threshold:
                        hit = True
            if hit:
                out.append((i, j))
    return out


def reference_clusters(segment_pairs):
    """Original O(P^2) union-find."""
    pairs = [(int(i), int(j)) for i, j in segment_pairs]
    if not pairs:
        return []
    if len(pairs) == 1:
        return [pairs]
    parent = list(range(len(pairs)))

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(pairs)):
        i_left, j_left = pairs[left]
        for right in range(left + 1, len(pairs)):
            i_right, j_right = pairs[right]
            if i_left == i_right or j_left == j_right:
                union(left, right)
            elif abs(i_left - i_right) == 1 or abs(j_left - j_right) == 1:
                union(left, right)
    grouped = {}
    for idx, pair in enumerate(pairs):
        grouped.setdefault(find(idx), []).append(pair)
    return list(grouped.values())


def lattice_path(rng, n, res=0.2, start=(0, 0)):
    """Random 8-connected walk on a res-lattice (many exact 0.5 m / 0.2 m distances)."""
    k = np.array(start)
    pts = [tuple(res * k)]
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for _ in range(n - 1):
        k = k + np.array(steps[int(rng.integers(0, 8))])
        pts.append((res * int(k[0]), res * int(k[1])))
    return pts


class TestConflictDetection(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_mp()._vectorized_norm_is_exact(), "platform check should pass here")

    def _check(self, pa, pb, thr):
        got = list(_mp()._iter_conflicting_segment_pairs(pa, pb, thr))
        self.assertEqual(got, reference_pairs(pa, pb, thr))
        self.assertTrue(all(type(i) is int and type(j) is int for i, j in got))
        return got

    def test_lattice_walks_and_offsets(self):
        rng = np.random.default_rng(0)
        total = 0
        for trial in range(40):
            pa = lattice_path(rng, int(rng.integers(2, 120)))
            pb = lattice_path(rng, int(rng.integers(2, 120)), start=(int(rng.integers(-3, 4)), int(rng.integers(-3, 4))))
            if trial % 3 == 0:  # lateral lane offset like social paths
                pb = [(x, y + 0.3) for x, y in pb]
            for thr in (0.5, 0.2, 0.0):
                total += len(self._check(pa, pb, thr))
        self.assertGreater(total, 0)

    def test_random_float_paths_and_tiny_inputs(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            pa = [tuple(p) for p in rng.uniform(-3, 3, (int(rng.integers(2, 60)), 2))]
            pb = [tuple(p) for p in rng.uniform(-3, 3, (int(rng.integers(2, 60)), 2))]
            self._check(pa, pb, float(rng.uniform(0.1, 1.5)))
        self._check([(0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)], 0.5)
        self._check([], [(0.0, 0.0), (1.0, 0.0)], 0.5)
        self._check([(0, 0), (1, 0)], [(0, 0), (0, 1)], 0.5)  # integer points

    def test_blocking_matches_single_block(self):
        rng = np.random.default_rng(2)
        pa = lattice_path(rng, 300)
        pb = lattice_path(rng, 250)
        expected = reference_pairs(pa, pb, 0.5)
        mp = _mp()
        old = mp._CONFLICT_BLOCK_ELEMENTS
        try:
            for elements in (1, 251, 700, 10 ** 9):
                mp._CONFLICT_BLOCK_ELEMENTS = elements
                self.assertEqual(list(mp._iter_conflicting_segment_pairs(pa, pb, 0.5)), expected, elements)
        finally:
            mp._CONFLICT_BLOCK_ELEMENTS = old


class TestEncounterClustering(unittest.TestCase):
    def test_matches_reference_on_random_and_real_pair_sets(self):
        em = _em()
        rng = np.random.default_rng(3)
        cases = [[], [(3, 4)]]
        for _ in range(200):
            n = int(rng.integers(2, 80))
            cases.append([(int(i), int(j)) for i, j in rng.integers(0, 40, (n, 2))])
        for _ in range(30):
            pa = lattice_path(rng, int(rng.integers(2, 150)))
            pb = lattice_path(rng, int(rng.integers(2, 150)))
            cases.append(reference_pairs(pa, pb, 0.5))
        for pairs in cases:
            self.assertEqual(em._cluster_segment_pairs_into_encounters(pairs), reference_clusters(pairs))


if __name__ == "__main__":
    unittest.main()
