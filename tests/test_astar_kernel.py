"""Parity tests: numba A* kernel (astar_kernel) vs the pure-Python loop in a_star.py."""

import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from social_path_planning import astar_kernel  # noqa: E402
from social_path_planning.a_star import AStar, AStar_With_Graph  # noqa: E402
from social_path_planning.common import generate_random_free_point  # noqa: E402
from social_path_planning.heat_map import HeatMap2DVector  # noqa: E402
from social_path_planning.occupancy_grid import StochOccupancyGrid2D  # noqa: E402
from social_path_planning.planning_grid import PlanningGrid  # noqa: E402
from social_path_planning.sparse_graph import FrequentSubgraph  # noqa: E402
from social_path_planning.wall_distance_cache import precompute_d_right  # noqa: E402

KERNEL = astar_kernel.kernel_ready()
SKIP_REASON = f"A* kernel unavailable: {astar_kernel.UNAVAILABLE_REASON}"


def make_grid(seed, width=70, height=55, resolution=0.2, origin=(0.0, 0.0), robot_d=0.5):
    """Corridor-like map: outer wall, interior walls with gaps, some blobs and unknown cells."""
    rng = np.random.default_rng(seed)
    probs = np.zeros((height, width))
    probs[0, :] = probs[-1, :] = probs[:, 0] = probs[:, -1] = 1.0
    for _ in range(5):
        if rng.random() < 0.5:
            r = int(rng.integers(5, height - 5))
            probs[r, 1:-1] = 1.0
            gap = int(rng.integers(2, width - 12))
            probs[r, gap:gap + 8] = 0.0
        else:
            c = int(rng.integers(5, width - 5))
            probs[1:-1, c] = 1.0
            gap = int(rng.integers(2, height - 12))
            probs[gap:gap + 8, c] = 0.0
    for _ in range(6):
        r0, c0 = int(rng.integers(1, height - 4)), int(rng.integers(1, width - 4))
        probs[r0:r0 + 3, c0:c0 + 3] = 1.0
    probs[rng.random((height, width)) < 0.01] = -1.0
    return StochOccupancyGrid2D(resolution, width, height, origin[0], origin[1], 10, probs, robot_d=robot_d)


def snap(v, res):
    return res * round(v / res)


def state_hi(grid):
    return (grid.origin_x + grid.width * grid.resolution, grid.origin_y + grid.height * grid.resolution)


def solve_signature(problem, mode):
    with contextlib.redirect_stdout(io.StringIO()):
        ok, telemetry = problem.solve(mode=mode, return_telemetry=True)
    telemetry = {k: v for k, v in telemetry.items() if k != "solve_time_s"}
    return {
        "ok": ok,
        "path": problem.path,
        "closed_set": problem.closed_set,
        "came_from": problem.came_from,
        "cost_to_arrive": problem.cost_to_arrive,
        "dist_to_right": problem.dist_to_right if mode != "vanilla" else None,
        "telemetry": telemetry,
    }


def social_graph(grid, routes, astar_res):
    heat = HeatMap2DVector(grid)
    lo = (grid.origin_x, grid.origin_y)
    for a, b in routes:
        p = AStar(lo, state_hi(grid), a, b, grid, resolution=astar_res)
        p.use_compiled = False
        with contextlib.redirect_stdout(io.StringIO()):
            if p.solve(mode="vanilla"):
                for _ in range(2):
                    heat.add_path(p.path)
    frequent = FrequentSubgraph(grid)
    frequent.set_heat_map(heat.heatmap)
    frequent.build_graph(threshold=2.0)
    frequent.prune_graph(min_component_size=3)
    return frequent.graph


@unittest.skipUnless(KERNEL, SKIP_REASON)
class TestFloatHelpers(unittest.TestCase):
    def test_hypot_matches_cpython(self):
        rng = np.random.default_rng(1)
        values = np.concatenate([
            rng.uniform(-100, 100, 20000),
            rng.uniform(-1, 1, 20000) * 10.0 ** rng.integers(-300, 300, 20000),
            0.2 * rng.integers(-3000, 3000, 20000) - 0.2 * rng.integers(-3000, 3000, 20000),
        ]).tolist()
        for x, y in zip(values, values[1:] + values[:1]):
            self.assertEqual(astar_kernel.py_hypot(x, y), math.hypot(x, y), (x, y))
        for x, y in [(0.0, 0.0), (-0.0, 3.0), (math.inf, math.nan), (math.nan, 1.0), (5e-324, 5e-324)]:
            expected = math.hypot(x, y)
            got = astar_kernel.py_hypot(x, y)
            self.assertTrue(got == expected or (math.isnan(got) and math.isnan(expected)), (x, y))

    def test_norm2_matches_numpy(self):
        rng = np.random.default_rng(2)
        for x, y in rng.uniform(-50, 50, (20000, 2)).tolist():
            self.assertEqual(astar_kernel.norm2(x, y), float(np.linalg.norm(np.array([x, y]))))


@unittest.skipUnless(KERNEL, SKIP_REASON)
class TestKernelParity(unittest.TestCase):
    def _compare(self, grid, routes, *, astar_res=None, graph=None, lo=None, hi=None, modes=None):
        astar_res = grid.resolution if astar_res is None else astar_res
        lo = (grid.origin_x, grid.origin_y) if lo is None else lo
        hi = state_hi(grid) if hi is None else hi
        cases = modes or [(AStar, "vanilla"), (AStar, "modified")]
        if graph is not None and modes is None:
            cases = cases + [(AStar_With_Graph, "vanilla"), (AStar_With_Graph, "modified")]
        for cls, mode in cases:
            for a, b in routes:
                results = []
                for compiled in (False, True):
                    if cls is AStar_With_Graph:
                        p = cls(lo, hi, a, b, grid, graph, resolution=astar_res)
                    else:
                        p = cls(lo, hi, a, b, grid, resolution=astar_res)
                    p.use_compiled = compiled
                    if compiled:
                        self.assertIsNotNone(p._compiled_inputs(mode), "kernel should handle this problem")
                    results.append(solve_signature(p, mode))
                with self.subTest(cls=cls.__name__, mode=mode, start=a, goal=b):
                    for key in results[0]:
                        self.assertEqual(results[0][key], results[1][key], key)

    def _routes(self, grid, n, seed, res=None):
        rng = np.random.default_rng(seed)
        res = grid.resolution if res is None else res
        routes = []
        for _ in range(n):
            a = generate_random_free_point(grid, rng)
            b = generate_random_free_point(grid, rng)
            routes.append(((snap(a[0], res), snap(a[1], res)), (snap(b[0], res), snap(b[1], res))))
        return routes

    def test_raycast_grids(self):
        for seed, kwargs in [(0, {}), (1, {"origin": (-3.3, 1.7)}), (2, {"resolution": 0.25, "robot_d": 0.9})]:
            grid = make_grid(seed, **kwargs)
            routes = self._routes(grid, 8, 10 + seed)
            graph = social_graph(grid, routes[:5], grid.resolution)
            self.assertGreater(graph.number_of_edges(), 0)
            with self.subTest(seed=seed, **kwargs):
                self._compare(grid, routes, graph=graph)

    def test_wall_distance_cache_grid(self):
        grid = make_grid(3)
        grid._d_right = precompute_d_right(grid, show_progress=False)
        routes = self._routes(grid, 8, 13)
        self._compare(grid, routes, graph=social_graph(grid, routes[:5], grid.resolution))

    def test_astar_resolution_differs_from_grid(self):
        grid = make_grid(4)
        routes = self._routes(grid, 6, 14, res=0.4)
        self._compare(grid, routes, astar_res=0.4)

    def test_statespace_smaller_than_map_and_unreachable_goal(self):
        grid = make_grid(5)
        hi = (grid.width * grid.resolution * 0.6, grid.height * grid.resolution)
        routes = self._routes(grid, 6, 15)
        self._compare(grid, routes, hi=hi)
        # goal inside a wall: search exhausts in both implementations
        blocked = np.argwhere(grid.probs > 0.85)[len(np.argwhere(grid.probs > 0.85)) // 2]
        goal = (snap(blocked[1] * grid.resolution, grid.resolution), snap(blocked[0] * grid.resolution, grid.resolution))
        self._compare(grid, [(routes[0][0], goal)])

    def test_falls_back_for_unsupported_problems(self):
        grid = make_grid(6)
        (a, b), = self._routes(grid, 1, 16)
        overlay_problem = AStar((0, 0), state_hi(grid), a, b, PlanningGrid(grid), resolution=grid.resolution)
        self.assertIsNone(overlay_problem._compiled_inputs("modified"))
        odd_graph = nx.DiGraph()
        odd_graph.add_edge((0, 0), (5, 5), weight=1.0)  # not an 8-neighbor edge
        graph_problem = AStar_With_Graph((0, 0), state_hi(grid), a, b, grid, odd_graph, resolution=grid.resolution)
        self.assertIsNone(graph_problem._compiled_inputs("modified"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(graph_problem.solve(mode="modified"))  # Python loop still works


if __name__ == "__main__":
    unittest.main()
