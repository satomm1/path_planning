"""Regression tests for precomputed directional wall-distance cache."""

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.wall_distance_cache import (
    NEIGHBOR_OFFSETS,
    opposite_dir_idx,
    precompute_d_right,
    save_wall_distance_cache,
    travel_dir_to_dir_idx,
    try_load_wall_distance_cache,
    world_xy_for_cell_indices,
)


def _tiny_free_grid():
    """Small map with a wall strip for nontrivial ray distances."""
    h, w = 24, 32
    probs = np.zeros((h, w), dtype=np.float32)
    probs[:, 20:] = 1.0
    probs[0, :] = 1.0
    probs[-1, :] = 1.0
    probs[:, 0] = 1.0
    return probs, 0.1, w, h


class TestWallDistanceCache(unittest.TestCase):
    def test_precompute_matches_raycast(self):
        probs, res, width, height = _tiny_free_grid()
        grid = StochOccupancyGrid2D(res, width, height, 0.0, 0.0, 10, probs)
        d_right = precompute_d_right(grid)

        rng = np.random.default_rng(0)
        for _ in range(80):
            row = int(rng.integers(0, height))
            col = int(rng.integers(0, width))
            k = int(rng.integers(0, 8))
            ii, jj = NEIGHBOR_OFFSETS[k]
            tnorm = np.sqrt(ii * ii + jj * jj)
            travel_dir = (ii / tnorm, jj / tnorm)
            wx, wy = world_xy_for_cell_indices(0.0, 0.0, res, col, row)
            ray = grid._dist_to_wall_right_raycast((wx, wy), travel_dir, dist_thresh=15.0)
            self.assertAlmostEqual(d_right[row, col, k], ray, places=5)

    def test_left_uses_opposite_direction(self):
        probs, res, width, height = _tiny_free_grid()
        grid = StochOccupancyGrid2D(res, width, height, 0.0, 0.0, 10, probs)
        d_right = precompute_d_right(grid)
        grid._d_right = d_right

        rng = np.random.default_rng(1)
        for _ in range(40):
            row = int(rng.integers(1, height - 1))
            col = int(rng.integers(1, width - 1))
            k = int(rng.integers(0, 8))
            ii, jj = NEIGHBOR_OFFSETS[k]
            tnorm = np.sqrt(ii * ii + jj * jj)
            travel_dir = (ii / tnorm, jj / tnorm)
            wx, wy = world_xy_for_cell_indices(0.0, 0.0, res, col, row)
            x = (wx, wy)
            left = grid.dist_to_wall_left(x, travel_dir)
            opp = opposite_dir_idx(k)
            self.assertAlmostEqual(left, d_right[row, col, opp], places=5)

    def test_npz_roundtrip(self):
        probs, res, width, height = _tiny_free_grid()
        grid = StochOccupancyGrid2D(res, width, height, 0.0, 0.0, 10, probs)
        d_right = precompute_d_right(grid)
        path = Path(self.id().split(".")[-1] + "_cache.npz")
        # use temp dir via unittest - simpler: use tempfile
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            save_wall_distance_cache(path, grid, d_right)

            grid2 = StochOccupancyGrid2D(res, width, height, 0.0, 0.0, 10, probs)
            loaded = try_load_wall_distance_cache(path, grid2)
            self.assertIsNotNone(loaded)
            np.testing.assert_allclose(loaded, d_right, rtol=0, atol=1e-4)

    def test_travel_dir_maps_to_neighbor_index(self):
        for k, (ii, jj) in enumerate(NEIGHBOR_OFFSETS):
            tnorm = np.sqrt(ii * ii + jj * jj)
            td = (ii / tnorm, jj / tnorm)
            self.assertEqual(travel_dir_to_dir_idx(td), k)

    def test_stoch_grid_loads_cache_path(self):
        import tempfile

        probs, res, width, height = _tiny_free_grid()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.npz"
            g0 = StochOccupancyGrid2D(res, width, height, 0.0, 0.0, 10, probs)
            save_wall_distance_cache(path, g0, precompute_d_right(g0))

            g1 = StochOccupancyGrid2D(
                res,
                width,
                height,
                0.0,
                0.0,
                10,
                probs,
                wall_distance_cache_path=path,
                auto_build_wall_distance_cache=False,
            )
            self.assertIsNotNone(g1._d_right)
            wx, wy = world_xy_for_cell_indices(0.0, 0.0, res, 5, 5)
            v = g1.dist_to_wall_right((wx, wy), (1.0, 0.0))
            v2 = g1._dist_to_wall_right_raycast((wx, wy), (1.0, 0.0))
            self.assertAlmostEqual(v, v2, places=4)


if __name__ == "__main__":
    unittest.main()
