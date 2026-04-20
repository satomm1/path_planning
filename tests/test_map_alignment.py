"""Tests for optional map raster alignment (``map_align_deg``).

How to run:
    set PYTHONPATH=src
    python -m social_path_planning.tune_map_alignment --scenario y2e2
"""

import io
import sys
from pathlib import Path
from unittest import TestCase

import numpy as np
from scipy.ndimage import rotate as ndimage_rotate

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from social_path_planning.a_star import AStar
from social_path_planning.compare_astar import build_occ_grid
from social_path_planning.grid_loader import align_occ_map_raster, load_grid_scenario
from social_path_planning.utils import snap_to_grid


def _horizontal_free_peak_score(occ):
    """Share of free cells lying in the single row that has the most free cells."""
    free = occ == 0
    row_counts = free.sum(axis=1)
    total = row_counts.sum()
    if total == 0:
        return 0.0
    return float(row_counts.max() / total)


class TestAlignOccMapRaster(TestCase):
    def test_zero_deg_returns_same_shape(self):
        n = 10
        occ = np.full((n, n), -1.0, dtype=np.float32)
        out, map_size = align_occ_map_raster(occ, 0.0, 0.05, crop_known=False)
        self.assertEqual(out.shape, (n, n))
        self.assertEqual(map_size, [n * 0.05, n * 0.05])

    def test_compensating_rotation_concentrates_horizontal_corridor(self):
        """A horizontal strip, rotated 45 deg then aligned by -45 deg, is row-peaked again."""
        n = 40
        res = 0.1
        occ = np.ones((n, n), dtype=np.float32)
        occ[17:22, 8:32] = 0.0
        tilted = ndimage_rotate(
            occ, 45.0, axes=(0, 1), reshape=True, order=0, cval=1.0
        )
        s_tilted = _horizontal_free_peak_score(tilted)
        aligned, _ = align_occ_map_raster(tilted, -45.0, res, crop_known=True)
        s_aligned = _horizontal_free_peak_score(aligned)
        self.assertGreater(s_aligned, s_tilted)


class TestY2E2LoadRegression(TestCase):
    def test_y2e2_loads_with_map_align_zero(self):
        occ, map_size, res = load_grid_scenario("y2e2", plot=False)
        self.assertEqual(occ.ndim, 2)
        self.assertEqual(len(map_size), 2)
        self.assertGreater(occ.shape[0], 10)
        self.assertGreater(occ.shape[1], 10)
        nx, ny = occ.shape
        self.assertAlmostEqual(map_size[0], nx * res, places=5)
        self.assertAlmostEqual(map_size[1], ny * res, places=5)

    def test_y2e2_one_social_plan(self):
        occ_grid, map_size, res, statespace_hi = build_occ_grid("y2e2")
        rng = np.random.default_rng(42)
        for _ in range(200):
            x_init = (
                float(rng.uniform(occ_grid.origin_x, occ_grid.origin_x + occ_grid.width * res)),
                float(rng.uniform(occ_grid.origin_y, occ_grid.origin_y + occ_grid.height * res)),
            )
            x_init = snap_to_grid(x_init, res)
            x_goal = (
                float(rng.uniform(occ_grid.origin_x, occ_grid.origin_x + occ_grid.width * res)),
                float(rng.uniform(occ_grid.origin_y, occ_grid.origin_y + occ_grid.height * res)),
            )
            x_goal = snap_to_grid(x_goal, res)
            while np.linalg.norm(np.array(x_goal) - np.array(x_init)) > 30:
                x_goal = (
                    float(rng.uniform(occ_grid.origin_x, occ_grid.origin_x + occ_grid.width * res)),
                    float(rng.uniform(occ_grid.origin_y, occ_grid.origin_y + occ_grid.height * res)),
                )
                x_goal = snap_to_grid(x_goal, res)
            if not occ_grid.is_free(x_init) or not occ_grid.is_free(x_goal):
                continue
            if np.linalg.norm(np.array(x_goal) - np.array(x_init)) < 5.0:
                continue
            planner = AStar(
                [0, 0],
                statespace_hi,
                x_init,
                x_goal,
                occ_grid,
                resolution=res,
            )
            buf = io.StringIO()
            old_out = sys.stdout
            try:
                sys.stdout = buf
                ok, _ = planner.solve(mode="modified", return_timing=True)
            finally:
                sys.stdout = old_out
            if not ok:
                continue
            return
        self.fail("could not find a feasible random start/goal on y2e2 after 200 tries")
