"""Parity tests: fast ``is_free`` lookup and vectorized ray-cast vs the original scalar code."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from social_path_planning.occupancy_grid import StochOccupancyGrid2D  # noqa: E402


def reference_is_free(grid, state):
    """Original per-call configuration-space window check."""
    x, y = grid.snap_to_grid(state)
    grid_x = int((x - grid.origin_x) / grid.resolution)
    grid_y = int((y - grid.origin_y) / grid.resolution)
    half = int(round(grid.robot_d / 2 / grid.resolution))
    x_lo = max(0, grid_x - half)
    y_lo = max(0, grid_y - half)
    x_hi = min(grid.width, grid_x + half)
    y_hi = min(grid.height, grid_y + half)
    window = grid.probs[y_lo:y_hi, x_lo:x_hi]
    if np.sum(window > 0.85):
        return False
    if np.sum(window < 0):
        return False
    return True


def reference_raycast(grid, x, travel_dir, dist_thresh=15.0):
    """Original scalar ray-march (copied verbatim apart from ``self`` -> ``grid``)."""
    right = np.array([travel_dir[1], -travel_dir[0]], dtype=float)
    rn = np.linalg.norm(right)
    if rn < 1e-12:
        return 100.0
    right = right / rn
    x_min = grid.origin_x
    x_max = grid.origin_x + grid.width * grid.resolution
    y_min = grid.origin_y
    y_max = grid.origin_y + grid.height * grid.resolution
    step = max(grid.resolution * 0.25, 1e-4)
    t_max = 0.0
    if right[0] > 0:
        t_max = max(t_max, (x_max - x[0]) / right[0])
    elif right[0] < 0:
        t_max = max(t_max, (x_min - x[0]) / right[0])
    if right[1] > 0:
        t_max = max(t_max, (y_max - x[1]) / right[1])
    elif right[1] < 0:
        t_max = max(t_max, (y_min - x[1]) / right[1])
    if t_max <= 0:
        return 100
    if t_max > dist_thresh:
        t_max = dist_thresh
    n_steps = int(np.ceil(t_max / step))
    px = float(x[0])
    py = float(x[1])
    for i in range(1, n_steps + 1):
        t = i * step
        sx = px + right[0] * t
        sy = py + right[1] * t
        if sx < x_min or sx >= x_max or sy < y_min or sy >= y_max:
            return 100
        col = int(np.floor((sx - grid.origin_x) / grid.resolution))
        row = int(np.floor((sy - grid.origin_y) / grid.resolution))
        if row < 0 or row >= grid.height or col < 0 or col >= grid.width:
            return 100
        p = grid.probs[row, col]
        if p >= grid.thresh or p < 0:
            return t
    return 100


def make_grid(seed, width=60, height=45, resolution=0.2, origin=(0.0, 0.0), robot_d=0.5):
    rng = np.random.default_rng(seed)
    probs = np.zeros((height, width))
    for _ in range(12):  # rectangular walls and blobs
        r0, c0 = rng.integers(0, height), rng.integers(0, width)
        probs[r0:r0 + rng.integers(1, 8), c0:c0 + rng.integers(1, 15)] = 1.0
    probs[rng.random((height, width)) < 0.02] = -1.0  # unknown cells
    probs[rng.random((height, width)) < 0.02] = 0.9  # occupied but < thresh-free mismatch zone
    probs[rng.random((height, width)) < 0.02] = 0.6  # >= thresh for raycast, free for is_free
    probs[0, :] = 1.0
    return StochOccupancyGrid2D(resolution, width, height, origin[0], origin[1], 10, probs, robot_d=robot_d)


def lattice_states(grid, margin=3):
    res = grid.resolution
    for row in range(-margin, grid.height + margin):
        for col in range(-margin, grid.width + margin):
            yield grid.snap_to_grid((grid.origin_x + col * res, grid.origin_y + row * res))


class TestFreeLookup(unittest.TestCase):
    def _assert_matches(self, grid, extra_points=()):
        for state in list(lattice_states(grid)) + list(extra_points):
            got = grid.is_free(state)
            self.assertIs(type(got), bool)
            self.assertEqual(got, reference_is_free(grid, state), state)

    def test_matches_reference_on_lattice_and_random_points(self):
        for seed, kwargs in [
            (0, {}),
            (1, {"origin": (-3.3, 1.7)}),
            (2, {"robot_d": 0.0}),  # empty window
            (3, {"robot_d": 1.3, "resolution": 0.05}),
            (4, {"resolution": 0.25, "origin": (2.0, -1.0)}),
        ]:
            grid = make_grid(seed, **kwargs)
            rng = np.random.default_rng(100 + seed)
            lo = np.array([grid.origin_x, grid.origin_y]) - 1.0
            hi = lo + np.array([grid.width, grid.height]) * grid.resolution + 2.0
            pts = [tuple(rng.uniform(lo, hi)) for _ in range(500)]
            with self.subTest(seed=seed, **kwargs):
                self._assert_matches(grid, pts)

    def test_rebuilds_after_probs_reassignment(self):
        grid = make_grid(5)
        self._assert_matches(grid)
        new_probs = grid.probs.copy()
        new_probs[10:20, 10:30] = 1.0
        grid.probs = new_probs
        self._assert_matches(grid)

    def test_invalidate_after_in_place_edit(self):
        grid = make_grid(6)
        grid.is_free((1.0, 1.0))
        grid.probs[5:9, 5:9] = 1.0
        grid.invalidate_free_lookup()
        self._assert_matches(grid)


class TestVectorizedRaycast(unittest.TestCase):
    def _queries(self, grid, rng, n_random=400):
        res = grid.resolution
        queries = []
        # A*-style queries: lattice state x2 reached from neighbor x1, unit travel direction.
        for _ in range(n_random):
            col = int(rng.integers(0, grid.width))
            row = int(rng.integers(0, grid.height))
            x2 = grid.snap_to_grid((grid.origin_x + col * res, grid.origin_y + row * res))
            ii, jj = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)][int(rng.integers(0, 8))]
            x1 = grid.snap_to_grid((x2[0] - ii * res, x2[1] - jj * res))
            d = np.array(x2, dtype=float) - np.array(x1, dtype=float)
            d /= np.linalg.norm(d)
            queries.append((x2, d))
            queries.append((x1, [-d[0], -d[1]]))
        # Arbitrary points and directions, including off-map points and a zero direction.
        lo = np.array([grid.origin_x, grid.origin_y]) - 0.5
        hi = lo + np.array([grid.width, grid.height]) * res + 1.0
        for _ in range(n_random):
            ang = rng.uniform(0, 2 * math.pi)
            queries.append((tuple(rng.uniform(lo, hi)), (math.cos(ang), math.sin(ang))))
        queries.append(((1.0, 1.0), (0.0, 0.0)))
        return queries

    def test_matches_reference_bit_for_bit(self):
        for seed, kwargs in [
            (0, {}),
            (1, {"origin": (-3.3, 1.7)}),
            (3, {"resolution": 0.05}),
            (4, {"resolution": 0.25, "origin": (2.0, -1.0)}),
        ]:
            grid = make_grid(seed, **kwargs)
            rng = np.random.default_rng(200 + seed)
            for thresh in (15.0, 2.0, 0.03):
                for x, d in self._queries(grid, rng):
                    expected = reference_raycast(grid, x, d, dist_thresh=thresh)
                    got = grid._dist_to_wall_right_raycast(x, d, dist_thresh=thresh)
                    with self.subTest(seed=seed, x=x, d=tuple(d), thresh=thresh):
                        self.assertEqual(type(got), type(expected))
                        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
