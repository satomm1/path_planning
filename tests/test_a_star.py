from unittest import TestCase
from path_planning.a_star import AStar
from path_planning.occupancy_grid import StochOccupancyGrid2D
from path_planning.grid_loader import load_grid_scenario
from path_planning.utils import *


class TestAStar(TestCase):

    def setup(self):
        self.map_size = [100, 100]
        self.map_resolution = 0.2
        self.map_dim = [round(self.map_size[i] / self.map_resolution) for i in range(len(self.map_size))]

        self.occ1, _, _ = load_grid_scenario("sample2_default", plot=False)
        self.occ2, _, _ = load_grid_scenario("sample4_default", plot=False)

class TestGrid1(TestAStar):

    def setUp(self):
        super().setup()
        self.occ_grid = StochOccupancyGrid2D(self.map_resolution, self.map_dim[0], self.map_dim[1], 0, 0, 10, self.occ1.T)

    def test_path_planning_1(self):
        x_init = snap_to_grid([3, 3], self.map_resolution)
        x_goal = snap_to_grid([97, 97], self.map_resolution)
        problem = AStar([0,0], snap_to_grid(self.map_size, self.map_resolution), x_init, x_goal, self.occ_grid, resolution=self.map_resolution)
        problem_status = problem.solve()
        self.assertTrue(problem_status)

    def test_path_planning_2(self):
        x_init = snap_to_grid([3, 3], self.map_resolution)
        x_goal = snap_to_grid([50, 50], self.map_resolution)
        problem = AStar([0, 0], snap_to_grid(self.map_size, self.map_resolution), x_init, x_goal, self.occ_grid,
                        resolution=self.map_resolution)
        problem_status = problem.solve()
        self.assertTrue(problem_status)

    def test_path_planning_vanilla_mode(self):
        x_init = snap_to_grid([3, 3], self.map_resolution)
        x_goal = snap_to_grid([50, 50], self.map_resolution)
        problem = AStar([0, 0], snap_to_grid(self.map_size, self.map_resolution), x_init, x_goal, self.occ_grid,
                        resolution=self.map_resolution)
        problem_status = problem.solve(mode="vanilla")
        self.assertTrue(problem_status)

    def test_invalid_solver_mode_raises(self):
        x_init = snap_to_grid([3, 3], self.map_resolution)
        x_goal = snap_to_grid([50, 50], self.map_resolution)
        problem = AStar([0, 0], snap_to_grid(self.map_size, self.map_resolution), x_init, x_goal, self.occ_grid,
                        resolution=self.map_resolution)
        with self.assertRaises(ValueError):
            problem.solve(mode="invalid-mode")
