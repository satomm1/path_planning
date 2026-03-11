import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

import os

from occupancy_grid import StochOccupancyGrid2D
from a_star import AStar
from grid_loader import load_grid_scenario
from utils import *

class HeatMap2D(object):
    def __init__(self, occ_grid: StochOccupancyGrid2D):
        self.occ_grid = occ_grid
        self.resolution = occ_grid.resolution
        self.width = occ_grid.width
        self.height = occ_grid.height
        self.origin_x = occ_grid.origin_x
        self.origin_y = occ_grid.origin_y
        self.extent = occ_grid.extent

        self.heatmap = np.zeros((self.height, self.width))

    def snap_to_grid(self, x):
        return self.resolution * round(x[0] / self.resolution), self.resolution * round(x[1] / self.resolution)

    def get_index(self, x):
        return int(np.round((x[0] - self.origin_x) / self.resolution)), int(np.round((x[1] - self.origin_y) / self.resolution))

    def add_path(self, path, increment=5.0):
        for state in path:
            x_idx, y_idx = self.get_index(state)
            if 0 <= x_idx < self.width and 0 <= y_idx < self.height:
                self.heatmap[x_idx, y_idx] += increment

    def plot_heatmap(self, ax=None, show=True):
        if ax is None:
            _, ax = plt.subplots()
        self.occ_grid.plot_grid(ax=ax)
        im = ax.imshow(
            self.heatmap.T,
            cmap='hot',
            origin='lower',
            extent=self.extent,
            aspect='equal',
            alpha=0.4,
            zorder=2
        )
        plt.colorbar(im, ax=ax, label='Heat Intensity')
        ax.set_title("Heat Map")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if show:
            plt.show()
        return ax

    def load_heatmap(self, filename_prefix):
        filepath = f"{filename_prefix}_heatmap.npy"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Heatmap file not found: {filepath}")
        self.heatmap = np.load(filepath)
        print("Successfully loaded heatmap from ", filepath)

    def save_heatmap(self, filename_prefix):
        np.save(f"{filename_prefix}_heatmap.npy", self.heatmap)
        print("Successfully saved heatmap to ", f"{filename_prefix}_heatmap.npy")
        
class HeatMap2DVector(HeatMap2D):
    def __init__(self, occ_grid: StochOccupancyGrid2D):
        super().__init__(occ_grid)

        self.heatmap = np.zeros((self.height, self.width, 8))  # 8 for each possible direction
        # Directions are indexed as follows:
        # 0: NW, 1: N, 2: NE, 3: E, 4: W, 5: SW, 6: S, 7: SE

    def add_path(self, path, increment=1.0):
        direction_map = {
            (-1, 1): 0,  # NW
            (0, 1): 1,   # N
            (1, 1): 2,   # NE
            (1, 0): 3,   # E
            (-1, 0): 4,  # W
            (-1, -1): 5, # SW
            (0, -1): 6,  # S
            (1, -1): 7   # SE
        }

        for ii in range(len(path)-1):
            state = path[ii]
            next_state = path[ii+1]
            x_idx, y_idx = self.get_index(state)
            if 0 <= x_idx < self.width and 0 <= y_idx < self.height:
                direction = (np.sign(next_state[0] - state[0]), np.sign(next_state[1] - state[1]))
                if direction in direction_map:
                    dir_idx = direction_map[direction]
                    self.heatmap[x_idx, y_idx, dir_idx] += increment

    def plot_heatmap(self, ax=None, show=True, min_visible_intensity=5.0, add_legend=True):
        if ax is None:
            _, ax = plt.subplots()
        self.occ_grid.plot_grid(ax=ax)
        total_heatmap = np.sum(self.heatmap, axis=2).T
        # Zero out low values for better visibility
        total_heatmap = np.where(total_heatmap < min_visible_intensity, 0.0, total_heatmap)

        dominant_direction = np.argmax(self.heatmap, axis=2).T
        # Map direction index to angle (degrees) for hue
        dir_angles = np.array([135, 90, 45, 0, 180, 225, 270, 315])
        H = (dir_angles[dominant_direction] % 360) / 360.0
        # Intensity -> value channel (V). Normalize to [0,1]
        vmax = np.nanmax(total_heatmap)
        if vmax <= 0:
            vmax = 1.0
        V = np.clip(total_heatmap / vmax, 0.0, 1.0)

        # Saturation: zero where no heat, otherwise full saturation
        S = np.where(total_heatmap > 0, 1.0, 0.0)
        # Build HSV image and convert to RGB
        hsv = np.zeros((H.shape[0], H.shape[1], 3))
        hsv[..., 0] = H
        hsv[..., 1] = S
        hsv[..., 2] = V
        rgb = hsv_to_rgb(hsv)
        # Create an alpha channel: fully transparent where total_heatmap == 0
        alpha = np.where(total_heatmap > 0, 1, 0.0)  # 0.8 can be adjusted
        rgba = np.dstack((rgb, alpha))

        ax.imshow(rgba, origin='lower', extent=self.extent, aspect='equal', zorder=2)

        # Add legend mapping direction -> hue
        if add_legend:
            from matplotlib.lines import Line2D
            dir_labels = ['↖', '↑', '↗', '→', '←', '↙', '↓', '↘']
            # Build representative RGB colors for each direction (full saturation/value)
            legend_hsv = np.zeros((len(dir_angles), 3))
            legend_hsv[:, 0] = (dir_angles % 360) / 360.0
            legend_hsv[:, 1] = 1.0
            legend_hsv[:, 2] = 1.0
            legend_rgbs = hsv_to_rgb(legend_hsv)
            handles = [Line2D([0], [0], marker='s', color='none', markerfacecolor=tuple(c), markersize=10, linestyle='') for
                       c in legend_rgbs]
            ax.legend(handles, dir_labels, title='Direction (hue)', bbox_to_anchor=(1.35, 1.0))

        ax.set_title("Heat Map (HSV: hue=direction, value=intensity)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if show:
            plt.show()
        return ax


class HeatMap2DVectorField(HeatMap2D):
    def __init__(self, occ_grid: StochOccupancyGrid2D):
        super().__init__(occ_grid)

        self.heatmap = np.zeros((self.height, self.width, 2))  # 2 for vector field (x and y components)

    def add_path(self, path, increment=1.0):
        for ii in range(len(path)-2):
            prev_state = path[ii]
            state = path[ii+1]
            next_state = path[ii+2]
            x_idx, y_idx = self.get_index(state)
            if 0 <= x_idx < self.width and 0 <= y_idx < self.height:
                direction = np.array(next_state) - np.array(prev_state)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm  # Normalize
                    self.heatmap[x_idx, y_idx, 0] += direction[0] * increment
                    self.heatmap[x_idx, y_idx, 1] += direction[1] * increment

    def plot_heatmap(self, ax=None, show=True):
        if ax is None:
            _, ax = plt.subplots()
        self.occ_grid.plot_grid(ax=ax)
        X, Y = np.meshgrid(np.arange(self.origin_x, self.origin_x + self.width * self.resolution, self.resolution),
                           np.arange(self.origin_y, self.origin_y + self.height * self.resolution, self.resolution))
        U = self.heatmap[:, :, 0].T
        V = self.heatmap[:, :, 1].T
        M = np.sqrt(U**2 + V**2)
        C = M.copy()
        zero_mask = M < 4
        C[zero_mask] = np.nan
        quiver = ax.quiver(X, Y, U, V, C, cmap='viridis', zorder=2, scale=200)
        ax.set_title("Heat Map Vector Field")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.colorbar(quiver, ax=ax, label='Vector Magnitude')
        if show:
            plt.show()
        return ax

    def plot_heatmap_no_vectors(self, ax=None, show=True):
        if ax is None:
            _, ax = plt.subplots()
        self.occ_grid.plot_grid(ax=ax)
        magnitude = np.sqrt(self.heatmap[:, :, 0]**2 + self.heatmap[:, :, 1]**2).T
        im = ax.imshow(magnitude, cmap='hot', origin='lower', extent=self.extent, aspect='equal', alpha=0.6, zorder=2)
        plt.colorbar(im, ax=ax, label='Heat Intensity')
        ax.set_title("Heat Map Magnitude")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if show:
            plt.show()
        return ax



def generate_random_free_point(occ_grid: StochOccupancyGrid2D):
    while True:
        xy = np.random.uniform(low=[occ_grid.origin_x, occ_grid.origin_y], high=[(occ_grid.origin_x + occ_grid.width)*occ_grid.resolution, (occ_grid.origin_y + occ_grid.height)*occ_grid.resolution])
        x, y = occ_grid.snap_to_grid(xy)
        if occ_grid.is_free((x, y)):
            return [x, y]

if __name__ == "__main__":
    # Example usage
    scenario_name = "sample2_default"
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    occ_grid = StochOccupancyGrid2D(map_resolution, round(map_size[0]/map_resolution), round(map_size[1]/map_resolution), 0, 0, 10, occ.T)

    heatmap = HeatMap2DVector(occ_grid)
    heatmap.load_heatmap("vector_incomplete")

    # Simulate adding paths
    for _ in range(10):
        x_init = generate_random_free_point(occ_grid)
        x_goal = generate_random_free_point(occ_grid)

        problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)
        if problem.solve():
            heatmap.add_path(problem.path, increment=1.0)

    heatmap.plot_heatmap()
    # heatmap.plot_heatmap_no_vectors()
    heatmap.save_heatmap("vector_incomplete")