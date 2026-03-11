import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import networkx as nx
import cvxpy as cp

from simulate import Simulator, Visualizer
from grid_loader import load_grid_scenario
from occupancy_grid import StochOccupancyGrid2D
from a_star import AStar
from utils import *

V_PREV_THRES = 0.0001
EPS = 0.1

class TrajectoryTracker:
    """ Trajectory tracking controller using differential flatness """

    def __init__(self, kpx, kpy, kdx, kdy,
                 V_max=0.6, om_max=1):
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

        self.coeffs = np.zeros(8)  # Polynomial coefficients for x(t) and y(t) as
        # returned by the differential flatness code

    def reset(self):
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def load_traj(self, times, traj):
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def get_desired_state(self, t):
        """
        Input:
            t: Current time
        Output:
            x_d, xd_d, xdd_d, y_d, yd_d, ydd_d: Desired state and derivatives
                at time t according to self.coeffs
        """
        x_d = np.interp(t, self.traj_times, self.traj[:, 0])
        y_d = np.interp(t, self.traj_times, self.traj[:, 1])
        xd_d = np.interp(t, self.traj_times, self.traj[:, 3])
        yd_d = np.interp(t, self.traj_times, self.traj[:, 4])
        xdd_d = np.interp(t, self.traj_times, self.traj[:, 5])
        ydd_d = np.interp(t, self.traj_times, self.traj[:, 6])

        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """

        dt = t - self.t_prev
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)

        ########## Code starts here ##########
        if self.V_prev < V_PREV_THRES:
            self.V_prev = np.sqrt(xd_d ** 2 + yd_d ** 2)

        x_dot = self.V_prev * np.cos(th)
        y_dot = self.V_prev * np.sin(th)

        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - x_dot)
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - y_dot)

        a = u1 * np.cos(th) + u2 * np.sin(th)
        om = -u1 * np.sin(th) / self.V_prev + u2 * np.cos(th) / self.V_prev

        V = self.V_prev + a * dt
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # If near the end of the trajectory, slow down so we don't stop abruptly
        x_goal, _, _, y_goal, _, _ = self.get_desired_state(self.traj_times[-1])
        dist = np.sqrt((x - x_goal) ** 2 + (y - y_goal) ** 2)
        if dist < 1:
            new_V_max = 0.25+0.25*dist # Slow down linearly starting at 0.5V_max to 0.25V_max
            V = np.clip(V, -new_V_max, new_V_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om


if __name__ == "__main__":
    # Create an occupancy grid
    scenario_name = "sample2_default"
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    map_dim = [round(map_size[i] / map_resolution) for i in range(len(map_size))]
    grid = StochOccupancyGrid2D(map_resolution, map_dim[0], map_dim[1], 0, 0, 10, occ.T)

    # Create simulator and visualizer
    sim = Simulator(grid)


    # Create local planner and global planner

    # This is the current local planner, using as placeholder until teb is integrated
    local_planner = TrajectoryTracker(kpx=2.5, kpy=2.5, kdx=1.5, kdy=1.5, V_max=1, om_max=1.0)

    x_init = snap_to_grid([2, 2], map_resolution)
    x_goal = snap_to_grid([50, 15], map_resolution)
    problem = AStar([0, 0], snap_to_grid(map_size, map_resolution), x_init, x_goal, grid, resolution=map_resolution)

    problem_status = problem.solve(plot=False)
    path = problem.path

    probs = grid.probs
    # add a square obstacle
    x_min = int(3.5/grid.resolution)
    x_max = int(5.5/grid.resolution)
    y_min = int(0.5 / grid.resolution)
    y_max = int(2.5 / grid.resolution)
    probs[y_min:y_max, x_min:x_max] = 1.0
    grid.probs = probs
    grid.compute_distance_map()
    viz = Visualizer(grid)
    viz.add_global_path(path)

    traj_times, traj = compute_smoothed_traj(path, 0.9, 3, 0.15, 0.1)
    local_planner.load_traj(traj_times, traj)

    sim.set_pose(x_init[0], x_init[1], 0)
    dt = 0.1  # time step
    actual_path = [sim.pose]
    while True:
        current_pose = sim.pose
        x, y, th = current_pose

        t = sim.time

        V, om = local_planner.compute_control(x, y, th, t)

        next_pose = sim.step(V, om, dt)
        actual_path.append(next_pose)

        viz.update(next_pose, actual_path)
        input("Press Enter to continue to next step...")

        if np.linalg.norm(np.array([x_goal[0], x_goal[1]]) - np.array([x, y])) < 0.5:
            print("Reached the goal!")
            break