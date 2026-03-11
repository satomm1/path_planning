import matplotlib.pyplot as plt

from occupancy_grid import StochOccupancyGrid2D
from a_star import AStar
from grid_loader import load_grid_scenario
from utils import *

if __name__ == "__main__":
    scenario_name = "hall_with_room_default"
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    map_dim = [round(map_size[i]/map_resolution) for i in range(len(map_size))]

    occ_grid = StochOccupancyGrid2D(map_resolution, map_dim[0], map_dim[1], 0, 0, 10, occ.T)

    x_init = snap_to_grid([60, 45], map_resolution)
    x_goal = snap_to_grid([15, 3], map_resolution)
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

    fig, ax = plt.subplots(figsize=(6,6))
    occ_grid.plot_grid_and_path(problem.path, ax)
    ax.scatter(x_init[0], x_init[1], c='green', s=100, label='Start', zorder=5)
    ax.scatter(x_goal[0], x_goal[1], c='gold', marker="*", s=100, label='Goal')

    closed_set = problem.closed_set
    xs, ys = zip(*closed_set)  # unzip into two sequences
    ax.scatter(xs, ys, c='red', marker="o", s=2, label='Explored Nodes', zorder=3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.subplots_adjust(right=0.65)
    plt.show()