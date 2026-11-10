"""Demo: dense-waypoint MILP coordination on grid2 (``sample2_default``).

Formerly the ``__main__`` block of ``multi_planning``. Plans (or loads cached
``path1.pkl`` ... ``path4.pkl`` from the working directory) four A* paths and
writes ``space_time_simultaneous.png`` / ``map_context_simultaneous.png``.
"""

import pickle

import matplotlib.pyplot as plt

from social_path_planning.a_star import AStar
from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.multi_planning import (
    MAX_VELOCITY,
    NOMINAL_VELOCITY,
    MultiAgentCombinedPlanner,
    MultiAgentSequentialPlanner,
    MultiAgentSimultaneousPlanner,
)
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.trajectory_viz import create_map_context_plot, create_space_time_plot
from social_path_planning.utils import snap_to_grid


def main():

    # Test the MultiAgentPlanner with dummy data
    scenario_name = "sample2_default"
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    occ_grid = StochOccupancyGrid2D(map_resolution, round(map_size[0]/map_resolution), round(map_size[1]/map_resolution), 0, 0, 10, occ.T)

    # Load path1.pkl if it exists
    try:
        with open("path1.pkl", "rb") as f:
            path1 = pickle.load(f)
    except FileNotFoundError:
        # Generate a path for the agent 1
        x_init = snap_to_grid([2, 25], map_resolution)
        x_goal = snap_to_grid([97, 50], map_resolution)
        problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)
        problem_status = problem.solve()
        path1 = problem.path if problem_status else None
        with open("path1.pkl", "wb") as f:
            pickle.dump(path1, f)
        # occ_grid.plot_grid_and_path(path1)
        # plt.show()

    # Load path2.pkl if it exists
    try:
        with open("path2.pkl", "rb") as f:
            path2 = pickle.load(f)
    except FileNotFoundError:
        # Agent 2 paths and times
        # x_init = snap_to_grid([25, 2], map_resolution)
        # x_goal = snap_to_grid([50, 97], map_resolution)
        x_init = snap_to_grid([2, 40], map_resolution)
        x_goal = snap_to_grid([97, 50], map_resolution)
        problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)
        problem_status = problem.solve()
        path2 = problem.path if problem_status else None
        with open("path2.pkl", "wb") as f:
            pickle.dump(path2, f)
        # occ_grid.plot_grid_and_path(path2)
        # plt.show()

    # Load path3.pkl if it exists
    try:
        with open("path3.pkl", "rb") as f:
            path3 = pickle.load(f)
    except FileNotFoundError:
        # Agent 3 paths and times
        x_init = snap_to_grid([50, 80], map_resolution)
        x_goal = snap_to_grid([97, 20], map_resolution)
        problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)
        problem_status = problem.solve()
        path3 = problem.path if problem_status else None
        with open("path3.pkl", "wb") as f:
            pickle.dump(path3, f)
        occ_grid.plot_grid_and_path(path3)
        plt.show()

    # Load path4.pkl if it exists
    try:
        with open("path4.pkl", "rb") as f:
            path4 = pickle.load(f)
    except FileNotFoundError:
        # Agent 4 paths and times
        x_init = snap_to_grid([50, 20], map_resolution)
        x_goal = snap_to_grid([97, 75], map_resolution)
        problem = AStar([0, 0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid,
                        resolution=map_resolution)
        problem_status = problem.solve()
        path4 = problem.path if problem_status else None
        with open("path4.pkl", "wb") as f:
            pickle.dump(path4, f)
        occ_grid.plot_grid_and_path(path4)
        plt.show()

    ############## Sequential Path Planning Example ##############
    # Assign uniform time steps for the other agent (path2)
    path2_times = [i * map_resolution * (1 / NOMINAL_VELOCITY) for i in range(len(path2))]
    path3_times = [i * map_resolution * (1 / (NOMINAL_VELOCITY/1.5)) for i in range(len(path3))]
    path4_times = [i * map_resolution * (1 / (NOMINAL_VELOCITY)) for i in range(len(path4))]

    # Create the planner
    planner = MultiAgentSequentialPlanner(occ_grid, [path2, path3, path4], [path2_times, path3_times, path4_times], path=path1)

    # Plan and visualize
    # times = planner.plan()
    # create_video([path1, path2, path3, path4], [times, path2_times, path3_times, path4_times], occ_grid=occ_grid, output_file="video_sequential.gif")

    ############## Simultaneous Path Planning Example ##############
    # Reduce granularity of paths for faster solving
    # path1 = path1[::5]
    # path2 = path2[::5]

    # Make the planner
    planner = MultiAgentSimultaneousPlanner(occ_grid, paths=[path1, path2, path3, path4])
    planner.assign_velocities([MAX_VELOCITY, MAX_VELOCITY/1.2, MAX_VELOCITY/1.5, MAX_VELOCITY/1.65])

    # Plan and visualize
    times = planner.plan()
    create_space_time_plot(
        [path1, path2, path3, path4],
        times,
        output_file="space_time_simultaneous.png",
        title="Simultaneous Planning Space-Time Plot"
    )
    snapshot_time = 0.345 * max(t_seq[-1] for t_seq in times)
    create_map_context_plot(
        [path1, path2, path3, path4],
        occ_grid=occ_grid,
        times=times,
        snapshot_time=snapshot_time,
        output_file="map_context_simultaneous.png",
        title="Simultaneous Planning Paths on Map"
    )
    # create_video([path1, path2, path3, path4], times, occ_grid=occ_grid, output_file="video_simultaneous.gif")

    ############## Combined Path Planning Example ##############
    # Assign uniform time steps for the already planned paths (path2/path3)
    path2_times = [i * map_resolution * (1 / (1.3*NOMINAL_VELOCITY)) for i in range(len(path2))]
    path3_times = [i * map_resolution * (1 / (NOMINAL_VELOCITY / 1.2)) for i in range(len(path3))]

    # Create the planner
    planner = MultiAgentCombinedPlanner(occ_grid, [path2, path3], [path2_times, path3_times], paths=[path1, path4])
    planner.assign_velocities([MAX_VELOCITY, MAX_VELOCITY / 2])

    # Plan and visualize
    # times = planner.plan()
    # create_video([path1, path2, path3, path4], [times[0], path2_times, path3_times, times[1]], occ_grid=occ_grid, output_file="video_combined.gif")


if __name__ == "__main__":
    main()
