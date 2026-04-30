"""Tests for optional map raster alignment (``map_align_deg``).

How to run:
    set PYTHONPATH=src
    python -m social_path_planning.tune_map_alignment --scenario y2e2
"""

import io
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
from scipy.ndimage import rotate as ndimage_rotate

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from social_path_planning.a_star import AStar
from social_path_planning.compare_astar import build_occ_grid
from social_path_planning.grid_loader import (
    _load_occ_from_map_yaml,
    align_occ_map_raster,
    load_grid_scenario,
    write_ros_map_pgm_yaml,
)
from social_path_planning.heat_map import heatmap_array_to_ros_pgm_layout
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.utils import snap_to_grid
from social_path_planning.wall_distance_cache import (
    NPZ_D_RIGHT_ROS,
    save_wall_distance_cache,
    wall_distance_d_right_to_ros_pgm_layout,
)


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


class TestWriteRosMapRoundTrip(TestCase):
    def test_write_then_load_matches_occ(self):
        nx, ny = 24, 16
        occ = np.full((nx, ny), -1.0, dtype=np.float32)
        occ[4:10, 3:12] = 0.0
        occ[15:20, 8:14] = 1.0
        occ[8, 8] = 0.37
        res = 0.05
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "roundtrip.yaml"
            write_ros_map_pgm_yaml(occ, res, out, occupancy_encoding="ros_int8")
            loaded, _map_size, res2 = _load_occ_from_map_yaml(out, crop_unknown=False)
        self.assertEqual(res2, res)
        np.testing.assert_array_equal(loaded, occ)

    def test_probability_encoding_roundtrip(self):
        nx, ny = 12, 10
        occ = np.full((nx, ny), -1.0, dtype=np.float32)
        occ[2:8, 2:7] = 0.0
        occ[9:11, 4:6] = 1.0
        res = 0.05
        env_dir = Path(__file__).resolve().parents[1] / "src" / "social_path_planning" / "environments"
        src_yaml = env_dir / "y2e2.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prob.yaml"
            write_ros_map_pgm_yaml(
                occ,
                res,
                out,
                source_map_yaml_path=src_yaml,
                occupancy_encoding="probability",
            )
            loaded, _map_size, res2 = _load_occ_from_map_yaml(out, crop_unknown=False)
        self.assertEqual(res2, res)
        np.testing.assert_array_equal(loaded, occ)


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


_HEAT_INV = (7, 6, 5, 4, 3, 2, 1, 0)
_WALL_INV = (2, 1, 0, 4, 3, 7, 6, 5)


class TestRosLayoutTransforms(TestCase):
    def test_heatmap_ros_matches_flipud_fliplr_gather(self):
        rng = np.random.default_rng(0)
        h, w = 4, 5
        H = rng.standard_normal((h, w, 8)).astype(np.float32)
        inv = np.asarray(_HEAT_INV, dtype=np.intp)
        expected = np.fliplr(np.flipud(H))[..., inv]
        got = heatmap_array_to_ros_pgm_layout(H)
        np.testing.assert_array_equal(got, expected)

    def test_wall_dist_ros_matches_flipud_gather(self):
        rng = np.random.default_rng(1)
        h, w = 4, 5
        D = rng.standard_normal((h, w, 8)).astype(np.float32)
        inv = np.asarray(_WALL_INV, dtype=np.intp)
        expected = np.flipud(D)[..., inv]
        got = wall_distance_d_right_to_ros_pgm_layout(D)
        np.testing.assert_array_equal(got, expected)

    def test_scalar_heatmap_ros_is_flipud_fliplr(self):
        rng = np.random.default_rng(2)
        s = rng.standard_normal((3, 7)).astype(np.float32)
        np.testing.assert_array_equal(
            heatmap_array_to_ros_pgm_layout(s), np.fliplr(np.flipud(s))
        )

    def test_vector_field_ros_flips_xy_component_signs(self):
        rng = np.random.default_rng(3)
        v = rng.standard_normal((2, 3, 2)).astype(np.float32)
        got = heatmap_array_to_ros_pgm_layout(v)
        fu = np.fliplr(np.flipud(v))
        np.testing.assert_array_equal(got[..., 0], -fu[..., 0])
        np.testing.assert_array_equal(got[..., 1], -fu[..., 1])

    def test_save_wall_distance_cache_includes_d_right_ros(self):
        height, width = 6, 8
        probs = np.zeros((height, width), dtype=np.float32)
        grid = StochOccupancyGrid2D(0.1, width, height, 0.0, 0.0, 10, probs)
        d_right = np.arange(height * width * 8, dtype=np.float64).reshape(
            height, width, 8
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.npz"
            save_wall_distance_cache(path, grid, d_right)
            with np.load(path, allow_pickle=False) as data:
                self.assertIn(NPZ_D_RIGHT_ROS, data)
                self.assertEqual(data[NPZ_D_RIGHT_ROS].shape, (height, width, 8))
                np.testing.assert_allclose(
                    data[NPZ_D_RIGHT_ROS],
                    wall_distance_d_right_to_ros_pgm_layout(d_right),
                    rtol=0,
                    atol=1e-5,
                )
