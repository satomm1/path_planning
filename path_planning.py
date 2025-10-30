import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import time

from occupancy_grid import StochOccupancyGrid2D
from a_star import AStar
from utils import *


def generate_sample_grid(map_size=[50,100], map_resolution=0.2, plot=False):
    map_dim = [round(map_size[i]/map_resolution) for i in range(len(map_size))]

    # Create empty map
    occ = np.zeros(map_dim)

    # Create outer boundary
    occ[0:3,:] = 1
    occ[:,0:3] = 1
    occ[-3:,:] = 1
    occ[:,-3:] = 1

    def point_to_grid(x):
        return round(x/map_resolution)

    # Create inner barriers
    occ[point_to_grid(5):point_to_grid(45), point_to_grid(5):point_to_grid(45)] = 1
    occ[point_to_grid(5.5):point_to_grid(44.5), point_to_grid(5.5):point_to_grid(44.5)] = -1

    occ[point_to_grid(5):point_to_grid(45), point_to_grid(55):point_to_grid(95)] = 1
    occ[point_to_grid(5.5):point_to_grid(44.5), point_to_grid(55.5):point_to_grid(94.5)] = -1

    if plot:
        cmap = ListedColormap(['gray', '#9DC6F2', 'black'])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)

        plt.imshow(occ.T, cmap=cmap, norm=norm, interpolation='nearest', origin='lower', extent=[0, map_size[0], 0, map_size[1]], aspect='equal')
        plt.title("Sample Occupancy Grid")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

    return occ

def generate_sample_grid2(map_size=[100,100], map_resolution=0.2, plot=False):
    map_dim = [round(map_size[i]/map_resolution) for i in range(len(map_size))]

    # Create empty map
    occ = np.zeros(map_dim)

    # Create outer boundary
    occ[0:3,:] = 1
    occ[:,0:3] = 1
    occ[-3:,:] = 1
    occ[:,-3:] = 1

    def point_to_grid(x):
        return round(x/map_resolution)

    # Create inner barriers
    occ[point_to_grid(5):point_to_grid(45), point_to_grid(5):point_to_grid(45)] = 1
    occ[point_to_grid(5.5):point_to_grid(44.5), point_to_grid(5.5):point_to_grid(44.5)] = -1

    occ[point_to_grid(5):point_to_grid(45), point_to_grid(55):point_to_grid(95)] = 1
    occ[point_to_grid(5.5):point_to_grid(44.5), point_to_grid(55.5):point_to_grid(94.5)] = -1

    occ[point_to_grid(55):point_to_grid(95), point_to_grid(5):point_to_grid(45)] = 1
    occ[point_to_grid(55.5):point_to_grid(94.5), point_to_grid(5.5):point_to_grid(44.5)] = -1

    occ[point_to_grid(55):point_to_grid(95), point_to_grid(55):point_to_grid(95)] = 1
    occ[point_to_grid(55.5):point_to_grid(94.5), point_to_grid(55.5):point_to_grid(94.5)] = -1

    if plot:
        cmap = ListedColormap(['gray', '#9DC6F2', 'black'])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)

        plt.imshow(occ.T, cmap=cmap, norm=norm, interpolation='nearest', origin='lower', extent=[0, map_size[0], 0, map_size[1]], aspect='equal')
        plt.title("Sample Occupancy Grid")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

    return occ

if __name__ == "__main__":
    map_size=[100,100]
    map_resolution = 0.2
    map_dim = [round(map_size[i]/map_resolution) for i in range(len(map_size))]

    occ = generate_sample_grid2(map_size, map_resolution, plot=False)
    occ_grid = StochOccupancyGrid2D(map_resolution, map_dim[0], map_dim[1], 0, 0, 10, occ.T)

    x_init = snap_to_grid([3,50], map_resolution)
    x_goal = snap_to_grid([97,75], map_resolution)
    problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)

    problem_status = problem.solve(plot=False)
    if problem_status:
        print("Path found!")

        plt.figure(2)
        occ_grid.plot_grid_and_path(problem.path)
        plt.scatter(x_init[0], x_init[1], c='green', s=100, label='Start')
        plt.scatter(x_goal[0], x_goal[1], c='gold', marker="*", s=100, label='Goal')
        plt.show()
    else:
        print("No path found.")

    plt.figure(3)
    occ_grid.plot_grid()
    plt.scatter(x_init[0], x_init[1], c='green', s=100, label='Start', zorder=5)
    plt.scatter(x_goal[0], x_goal[1], c='gold', marker="*", s=100, label='Goal')

    closed_set = problem.closed_set
    xs, ys = zip(*closed_set)  # unzip into two sequences
    plt.scatter(xs, ys, c='red', marker="o", s=0.25, label='Explored Nodes')
    plt.show()