import cvxpy as cp
import numpy as np

from occupancy_grid import StochOccupancyGrid2D
from a_star import AStar
from path_planning import generate_sample_grid2
from utils import *

NOMINAL_VELOCITY = 0.5  # m/s
TIME_STEP = 5  # seconds
MAX_VELOCITY = 0.7 # m/s
ROBOT_DIAMETER = 1  # meters
M = 1e6  # Big-M constant for constraints
DELTA = 1  # Safety margin in seconds

class MultiAgentPlanner:

    def __init__(self, occupancy_grid: StochOccupancyGrid2D, other_agent_paths, other_agent_times, path=None):
        """
        Initialize the multi-agent planner.

        Args:
            occupancy_grid (StochOccupancyGrid2D): The occupancy grid of the environment.
            other_agent_paths (list of list of tuples): Paths of other agents in the environment.
            other_agent_times (list of list of floats): Time steps corresponding to other agent paths.
            path (list of tuples, optional): The path for the agent to follow. Defaults to None.
        """
        
        self.occupancy_grid = occupancy_grid
        self.other_agent_paths = other_agent_paths
        self.other_agent_times = other_agent_times

        self.path = path

        self.time_steps = None
        self.collision_set = []

    def assign_path(self, path):
        self.path = path

    def find_collision_points(self):
        """
        Identify potential collision points with other agents along the assigned path.
        Adds the corresponding (i,j) indices for alpha variables to self.collision_set.
        """
        self.collision_set = []
        all_collision_indices = dict()
        index = 0
        for other_path, other_times in zip(self.other_agent_paths, self.other_agent_times):
            collision_indices = set()
            for i, waypoint in enumerate(self.path):
                for j, other_waypoint in enumerate(other_path):
                    if np.linalg.norm(np.array(waypoint) - np.array(other_waypoint)) <= ROBOT_DIAMETER:
                        collision_indices.add((j, i))
                        time_at_collision = other_times[j]
                        # Find the closest time step index
                        time_index = np.argmin(np.abs(self.time_steps - time_at_collision))
                        self.collision_set.append((i, time_index))
            all_collision_indices[index] = collision_indices
            index += 1

    def find_collision_intervals(self):
        """
        Identify potential collision intervals with other agents along the assigned path.
        returns a list of tuples indicating the (i, start_time, end_time) for collision intervals.
        
        i = index along self.path
        """
        collision_intervals = []
        for other_path, other_times in zip(self.other_agent_paths, self.other_agent_times):
            for i, waypoint in enumerate(self.path):
                collision_times = []
                for j, other_waypoint in enumerate(other_path):
                    if np.linalg.norm(np.array(waypoint) - np.array(other_waypoint)) <= ROBOT_DIAMETER:
                        collision_times.append(other_times[j])
                if collision_times:
                    start_time = min(collision_times)
                    end_time = max(collision_times)
                    collision_intervals.append((i, start_time, end_time))
        return collision_intervals
        
    def plan(self):
        if self.path is None:
            raise ValueError("Path not assigned. Please assign a path before planning.")
        
        t = cp.Variable(len(self.path))
        constraints = []
        constraints += [t[0] == 0]  # Start at time 0

        # Max velocity constraints (also enforces t_i+1 >= t_i)
        for i in range(len(self.path) - 1):
            delta_pos = np.linalg.norm(np.array(self.path[i+1]) - np.array(self.path[i]))
            constraints += [t[i+1] - t[i] >= delta_pos / MAX_VELOCITY]  

        # Collision Avoiding Constraints using Big-M method
        collision_intervals = self.find_collision_intervals()
        z = cp.Variable(len(collision_intervals), boolean=True)
        z_index = 0
        for (i, start_time, end_time) in collision_intervals:
            constraints += [t[i] <= start_time - DELTA + M * z[z_index]]
            constraints += [t[i] >= end_time + DELTA - M * (1 - z[z_index])]
            z_index += 1

        objective = cp.Minimize(t[-1])  # Minimize time to reach final point
        prob = cp.Problem(objective, constraints)
        print("Starting to solve multi-agent planning problem...")
        prob.solve(verbose=True)
        optimized_times = t.value.tolist()
        return optimized_times


    def plan1(self):
        if self.path is None:
            raise ValueError("Path not assigned. Please assign a path before planning.")
        
        # Discretize time steps based on other agent times and nominal velocity
        diag_distance = np.linalg.norm(np.array([self.occupancy_grid.resolution, self.occupancy_grid.resolution]))
        max_time = max([max(times) for times in self.other_agent_times]) + np.ceil(len(self.path) * diag_distance * (1 / NOMINAL_VELOCITY))
        self.time_steps = np.arange(0, max_time, TIME_STEP)
        
        # Generate the optimization variables
        alpha = cp.Variable((len(self.path), len(self.time_steps)))
        constraints = [alpha >= 0]  # Non-negativity constraints
        constraints += [cp.sum(alpha, axis=1) == 1]  # Convex combination constraints
        
        # Add collision avoiding constraints
        self.find_collision_points()
        for (i, j) in self.collision_set:
            constraints += [alpha[i, j] == 0]

        # Each t_i+1 >= t_i
        for i in range(len(self.path) - 1):
            constraints += [cp.sum(cp.multiply(alpha[i+1, :], self.time_steps)) >= 
                            cp.sum(cp.multiply(alpha[i, :], self.time_steps))]
        
        # Max Velocity constraints
        for i in range(len(self.path) - 1):
            delta_pos = np.linalg.norm(np.array(self.path[i+1]) - np.array(self.path[i]))
            time_i = cp.sum(cp.multiply(alpha[i, :], self.time_steps))
            time_next = cp.sum(cp.multiply(alpha[i+1, :], self.time_steps))
            constraints += [delta_pos <= MAX_VELOCITY * (time_next - time_i)]

        # Make sure each time step is used at least once
        for j in range(len(self.time_steps)):
            constraints += [cp.sum(alpha[:, j]) >= 0.0001]

        # Objective: Minimize time to reach final point
        objective = cp.Minimize(cp.sum(cp.multiply(alpha[-1, :], self.time_steps)))
        prob = cp.Problem(objective, constraints)

        print("Starting to solve multi-agent planning problem...")
        # Solve the problem
        prob.solve(verbose=True, solver=cp.CLARABEL)

        # Extract the optimized time steps for the path
        optimized_times = [cp.sum(cp.multiply(alpha[i, :], self.time_steps)).value for i in range(len(self.path))]
        return optimized_times

    
if __name__ == "__main__":

    # Test the MultiAgentPlanner with dummy data
    map_size = [100, 100]
    map_resolution = 0.2

    occ = generate_sample_grid2(map_size, map_resolution, plot=False)
    occ_grid = StochOccupancyGrid2D(map_resolution, round(map_size[0]/map_resolution), round(map_size[1]/map_resolution), 0, 0, 10, occ.T)

    # Generate a path for the main agent
    x_init = snap_to_grid([2, 25], map_resolution)
    x_goal = snap_to_grid([97, 50], map_resolution)
    problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)
    problem_status = problem.solve(plot=False) 
    path = problem.path if problem_status else None
    # occ_grid.plot_grid_and_path(path)
    # plt.show()

    # Dummy other agent paths and times
    # x_init = snap_to_grid([25, 2], map_resolution)
    # x_goal = snap_to_grid([50, 97], map_resolution)
    x_init = snap_to_grid([2, 40], map_resolution)
    x_goal = snap_to_grid([97, 50], map_resolution)
    other_problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)
    other_problem_status = other_problem.solve(plot=False)
    other_path = other_problem.path if other_problem_status else None
    # occ_grid.plot_grid_and_path(other_path)
    # plt.show()

    # Assign uniform time steps for the other agent
    other_times = [i * map_resolution * (1 / NOMINAL_VELOCITY) for i in range(len(other_path))]

    planner = MultiAgentPlanner(occ_grid, [other_path], [other_times], path=path)
    times = planner.plan()
    # print(times)
    plt.figure()
    plt.plot(times, label="Planned Times for Main Agent")
    plt.plot(other_times, label="Other Agent Times")
    plt.legend()
    plt.xlabel("Path Index")
    plt.ylabel("Time (s)")
    plt.title("Multi-Agent Path Planning")
    plt.show()
    
    # Plot x/y positions over time
    main_agent_positions = np.array(planner.path)
    other_agent_positions = np.array(other_path)    
    plt.figure()
    plt.plot(times, main_agent_positions[:,0], label="Main Agent X Position")
    plt.plot(other_times, other_agent_positions[:,0], label="Other Agent X Position")
    plt.plot(times, main_agent_positions[:,1], label="Main Agent Y Position")
    plt.plot(other_times, other_agent_positions[:,1], label="Other Agent Y Position")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("Agent Positions Over Time")
    plt.show()