import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import time
from itertools import cycle

from path_planning import generate_sample_grid2
from occupancy_grid import StochOccupancyGrid2D


class Simulator:

    def __init__(self, occ_grid: StochOccupancyGrid2D, v_std=0.3, w_std=0.3):
        self.occ_grid = occ_grid

        self.v_std = v_std  # Standard deviation for linear velocity noise
        self.w_std = w_std  # Standard deviation for angular velocity noise

        self.pose = (0.0, 0.0, 0.0)  # (x, y, theta)

        self.time = 0.0


    def set_pose(self, x, y, theta):
        self.pose = (x, y, theta)

    def is_collision(self, pose):
        return not self.occ_grid.is_free(pose)

    def step(self, v, w, dt, visualize=False):
        """
        Update the robot's pose based on velocity commands.
        v: linear velocity (m/s)
        w: angular velocity (rad/s)
        dt: time step (s)
        """
        self.time += dt

        x, y, theta = self.pose

        # Update pose
        x_new = x + (v + np.random.normal(0, self.v_std)) * np.cos(theta) * dt
        y_new = y + (v + np.random.normal(0, self.v_std)) * np.sin(theta) * dt
        theta_new = theta + (w + np.random.normal(0, self.w_std)) * dt

        new_pose = (x_new, y_new, theta_new)

        if self.is_collision((x_new, y_new)):
            print("Collision detected! Stopping movement.")
            return self.pose  # No movement if collision

        self.pose = new_pose
        return self.pose

class Visualizer:
    def __init__(self, occ_grid):
        self.occ_grid = occ_grid
        self.fig, self.ax = plt.subplots(1, 1)

        self.global_path = None

        # Use a list to store plot objects that need to be updated
        self.dynamic_artists = []

        # Turn on interactive mode
        plt.ion()
        # self.setup_plot()

        # A color cycle for plotting multiple paths
        self.color_cycle = cycle(['b', 'g', 'c', 'm', 'y', 'k'])

        self.viz_setup = False

    def setup_plot(self):
        # Display the occupancy grid
        # origin='lower' is important to match robot coordinates
        # grid_display = self.occ_grid.probs.T # Transpose for imshow
        self.ax.imshow(self.occ_grid.probs, cmap=ListedColormap(['gray', '#9DC6F2', 'black']),
                       extent=[self.occ_grid.origin_x,
                               self.occ_grid.origin_x + self.occ_grid.width * self.occ_grid.resolution,
                               self.occ_grid.origin_y,
                               self.occ_grid.origin_y + self.occ_grid.height * self.occ_grid.resolution],
                       origin='lower')

        if self.global_path is not None:
            path_x, path_y = zip(*[(p[0], p[1]) for p in self.global_path])
            self.ax.plot(path_x, path_y, 'r--', label="Global Path")

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Robot Simulator")
        self.ax.grid(True)
        self.fig.show()

    def add_global_path(self, path):
        # Display the original global path (dashed line)
        self.global_path = path
        self.setup_plot()
        self.viz_setup = True

    def update(self, pose, path_history, local_path=None):
        if not self.viz_setup:
            self.setup_plot()
            self.viz_setup = True

        # Remove old dynamic artists
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists = []

        # Plot the path history
        if path_history:
            path_x, path_y = zip(*[(p[0], p[1]) for p in path_history])
            path_line, = self.ax.plot(path_x, path_y, 'b-', label="Trajectory")
            self.dynamic_artists.append(path_line)

        # Plot the robot's current pose as an arrow
        x, y, theta = pose
        robot_arrow = patches.Arrow(x, y,
                                    2*np.cos(theta),  # Arrow length scaled for visibility
                                    2*np.sin(theta),
                                    width=0.5, color='r', label="Robot")
        self.ax.add_patch(robot_arrow)
        self.dynamic_artists.append(robot_arrow)

        # Plot the local path if provided
        if local_path is not None:
            local_path_x, local_path_y = zip(*[(p[0], p[1]) for p in local_path])
            local_path_line, = self.ax.plot(local_path_x, local_path_y, 'g--', label="Local Path")
            self.dynamic_artists.append(local_path_line)

        # Force a redraw of the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_paths(self, paths):
        """
        Method for BULK plotting. Plots a list of paths after simulations are complete.
        `paths`: A list of paths, where each path is a list of poses.
        """
        if not self.viz_setup:
            self.setup_plot()
            self.viz_setup = True

        # Clear any artists from previous animations
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists = []

        for path in paths:
            if not path:
                continue

            color = next(self.color_cycle)

            # Plot the trajectory
            path_x, path_y = zip(*[(p[0], p[1]) for p in path])
            path_line, = self.ax.plot(path_x, path_y, '-', linewidth=1.5, color=color, alpha=0.8)
            self.dynamic_artists.append(path_line)

            # Plot the start/final robot pose for this path
            x, y, theta = path[0]
            robot_start = patches.Circle(xy=(x,y), radius=1, color='green')
            self.ax.add_patch(robot_start)
            self.dynamic_artists.append(robot_start)
            x, y, theta = path[-1]
            robot_end = patches.Circle(xy=(x,y), radius=1, color='red')
            self.ax.add_patch(robot_end)
            self.dynamic_artists.append(robot_end)

        # Redraw the canvas once with all the new paths
        self.fig.canvas.draw()

    def show(self):
        """
        Finalizes the plot. Turns off interactive mode and shows the plot
        window until the user closes it.
        """
        plt.ioff()
        print("Displaying final plot. Close the plot window to exit.")
        plt.show()


if __name__ == "__main__":
    # --- Setup ---
    # Create an occupancy grid
    map_size = [100, 100]
    map_resolution = 0.2
    map_dim = [round(map_size[i] / map_resolution) for i in range(len(map_size))]
    occ = generate_sample_grid2(map_size, map_resolution, plot=False)
    grid = StochOccupancyGrid2D(map_resolution, map_dim[0], map_dim[1], 0, 0, 10, occ.T)

    # Create simulator and visualizer
    sim = Simulator(grid)
    viz = Visualizer(grid)

    # --- Simulation ---
    sim.set_pose(2.0, 2.0, 0)
    path = [sim.pose]
    dt = 0.1 # time step


    v = 1.0  # linear velocity (m/s)
    w = 0  # angular velocity (rad/s)

    for i in range(250):
        # Step the simulation
        current_pose = sim.step(v, w, dt)
        path.append(current_pose)

        # Update the visualization
        # viz.update(current_pose, path)

        # Small delay to make the animation viewable
        # time.sleep(0.001)

    viz.draw_paths([path])

    print("Simulation finished. Close the plot window to exit.")
    plt.ioff() # Turn interactive mode OFF
    plt.show() # Keep the final plot open