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

def generate_sample_grid3(map_size=[100,100], map_resolution=0.2, plot=False):
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
    occ[point_to_grid(5.5):point_to_grid(44.5), point_to_grid(5.5):point_to_grid(44.5)] = 1
    occ[point_to_grid(22.5):point_to_grid(27.5), point_to_grid(5):point_to_grid(7.5)] = 0

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

def generate_sample_grid4(map_size=[100,100], map_resolution=0.2, plot=False):
    map_dim = [round(map_size[i]/map_resolution) for i in range(len(map_size))]

    # Create empty map
    occ = np.zeros(map_dim)

    # Create outer boundary
    occ[0:3,:] = 1
    occ[:,0:3] = 1
    occ[-3:,:] = 1
    occ[:,-3:] = 1

    for center in [np.array([25.0, 25.0]), np.array([50.0, 50.0]), np.array([75.0, 75.0]), np.array([75, 25.0]), np.array([25, 75.0]), np.array([50, 0.0]), np.array([100, 0.0]), np.array([0, 0.0]),
                   np.array([50, 100.0]), np.array([100, 50.0]), np.array([100, 100.0]), np.array([0, 50.0]), np.array([0, 100.0])]:
        side = 24.0
        half = side / 2.0
        theta = np.deg2rad(45.0)
        c, s = np.cos(theta), np.sin(theta)

        # grid coordinates in meters (match occ indexing: first dim -> x, second -> y)
        xs = np.arange(map_dim[0]) * map_resolution
        ys = np.arange(map_dim[1]) * map_resolution
        X, Y = np.meshgrid(xs, ys, indexing='ij')  # shape (nx, ny)

        Xc = X - center[0]
        Yc = Y - center[1]

        # rotate points by -theta to align square axis-aligned in rotated frame
        xr = c * Xc + s * Yc
        yr = -s * Xc + c * Yc

        mask = (np.abs(xr) <= half) & (np.abs(yr) <= half)
        occ[mask] = 1


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

def generate_sample_grid5(map_size=[100,100], map_resolution=0.2, plot=False):
    map_dim = [round(map_size[i]/map_resolution) for i in range(len(map_size))]

    # Create empty map
    occ = np.zeros(map_dim)

    # Create outer boundary
    occ[0:3,:] = 1
    occ[:,0:3] = 1
    occ[-3:,:] = 1
    occ[:,-3:] = 1

    for center in [np.array([25.0, 25.0]), np.array([50.0, 50.0]), np.array([75.0, 75.0]), np.array([75, 25.0]), np.array([25, 75.0]), np.array([50, 0.0]), np.array([100, 0.0]), np.array([0, 0.0]),
                   np.array([50, 100.0]), np.array([100, 50.0]), np.array([100, 100.0]), np.array([0, 50.0]), np.array([0, 100.0])]:
        side = 24.0
        half = side / 2.0
        theta = np.deg2rad(22.5)
        c, s = np.cos(theta), np.sin(theta)

        # grid coordinates in meters (match occ indexing: first dim -> x, second -> y)
        xs = np.arange(map_dim[0]) * map_resolution
        ys = np.arange(map_dim[1]) * map_resolution
        X, Y = np.meshgrid(xs, ys, indexing='ij')  # shape (nx, ny)

        Xc = X - center[0]
        Yc = Y - center[1]

        # rotate points by -theta to align square axis-aligned in rotated frame
        xr = c * Xc + s * Yc
        yr = -s * Xc + c * Yc

        mask = (np.abs(xr) <= half) & (np.abs(yr) <= half)
        occ[mask] = 1


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

    occ = generate_sample_grid5(map_size, map_resolution, plot=False)

    occ_grid = StochOccupancyGrid2D(map_resolution, map_dim[0], map_dim[1], 0, 0, 10, occ.T)

    x_init = snap_to_grid([25,3], map_resolution)
    x_goal = snap_to_grid([50,70], map_resolution)
    problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)

    problem_status = problem.solve(plot=False)
    if problem_status:
        print("Path found!")

        plt.figure(2)
        occ_grid.plot_grid_and_path(problem.path)
        occ_grid.plot_smoothed_path(problem.smoothed_path)
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