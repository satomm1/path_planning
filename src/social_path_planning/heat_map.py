"""
Directional heatmaps from accumulated A* paths (`HeatMap2DVector`).

CLI example (Y2E2, social / modified A*):

    python -m social_path_planning.heat_map --scenario y2e2 --num-paths 20 \\
        --heatmap-prefix y2e2_routes --max-start-goal-distance 30

For faster planning on large YAML maps, precompute wall distances once:

    python -m social_path_planning.precompute_wall_distances --scenario y2e2

While running, the heatmap is saved to ``<prefix>_heatmap.npy`` every 10 successful
paths (crash recovery), then again at the end.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from social_path_planning.a_star import AStar
from social_path_planning.compare_astar import build_occ_grid, generate_random_free_point
from social_path_planning.occupancy_grid import StochOccupancyGrid2D

_CHECKPOINT_EVERY_N = 10  # save heatmap after this many successful paths (same file as final save)

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
                # heatmap is (height, width) = (row, col) = (y_idx, x_idx)
                self.heatmap[y_idx, x_idx] += increment

    def plot_heatmap(self, ax=None, show=True):
        if ax is None:
            _, ax = plt.subplots()
        self.occ_grid.plot_grid(ax=ax)
        im = ax.imshow(
            self.heatmap,
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
                    # heatmap is (height, width, 8) = (row, col, dir) = (y_idx, x_idx, dir_idx)
                    self.heatmap[y_idx, x_idx, dir_idx] += increment

    def plot_heatmap(self, ax=None, show=True, min_visible_intensity=5.0, add_legend=True):
        if ax is None:
            _, ax = plt.subplots()
        self.occ_grid.plot_grid(ax=ax)
        total_heatmap = np.sum(self.heatmap, axis=2)
        # Zero out low values for better visibility
        total_heatmap = np.where(total_heatmap < min_visible_intensity, 0.0, total_heatmap)

        dominant_direction = np.argmax(self.heatmap, axis=2)
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
                    self.heatmap[y_idx, x_idx, 0] += direction[0] * increment
                    self.heatmap[y_idx, x_idx, 1] += direction[1] * increment

    def plot_heatmap(self, ax=None, show=True):
        if ax is None:
            _, ax = plt.subplots()
        self.occ_grid.plot_grid(ax=ax)
        X, Y = np.meshgrid(np.arange(self.origin_x, self.origin_x + self.width * self.resolution, self.resolution),
                           np.arange(self.origin_y, self.origin_y + self.height * self.resolution, self.resolution))
        U = self.heatmap[:, :, 0]
        V = self.heatmap[:, :, 1]
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
        magnitude = np.sqrt(self.heatmap[:, :, 0] ** 2 + self.heatmap[:, :, 1] ** 2)
        im = ax.imshow(magnitude, cmap='hot', origin='lower', extent=self.extent, aspect='equal', alpha=0.6, zorder=2)
        plt.colorbar(im, ax=ax, label='Heat Intensity')
        ax.set_title("Heat Map Magnitude")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if show:
            plt.show()
        return ax


def parse_args():
    parser = argparse.ArgumentParser(
        description="Accumulate social (modified) A* paths into a HeatMap2DVector and save or plot."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="sample2_default",
        help="Scenario name in environments/grid_scenarios.json (default: sample2_default).",
    )
    parser.add_argument(
        "-n",
        "--num-paths",
        type=int,
        default=10,
        metavar="N",
        help="Number of successful planner runs whose paths are merged into the heatmap (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="RNG seed for start/goal sampling. If omitted, a random seed is used (printed so you can reproduce with --seed S).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max planner trials (default: max(100, 50 * num_paths)).",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=0.0,
        metavar="M",
        help="Minimum start–goal distance (m); 0 disables (default: 0).",
    )
    parser.add_argument(
        "--max-start-goal-distance",
        type=float,
        default=None,
        metavar="D",
        help="If set, resample goal until start–goal distance is at most D m (e.g. 30 for Y2E2).",
    )
    parser.add_argument(
        "--heatmap-prefix",
        type=str,
        default="heatmap_vector",
        help="Prefix for <prefix>_heatmap.npy save/load (default: heatmap_vector).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing <prefix>_heatmap.npy before adding new paths.",
    )
    parser.add_argument(
        "--increment",
        type=float,
        default=1.0,
        help="Per-edge increment passed to HeatMap2DVector.add_path (default: 1.0).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not display the matplotlib figure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_paths < 1:
        print("error: --num-paths must be >= 1", file=sys.stderr)
        return 2

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(100, 50 * args.num_paths)
    if max_attempts < 1:
        print("error: --max-attempts must be >= 1", file=sys.stderr)
        return 2

    seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
    rng = np.random.default_rng(seed)
    print(f"RNG seed: {seed}")
    occ_grid, _, map_resolution, statespace_hi = build_occ_grid(args.scenario)

    heatmap = HeatMap2DVector(occ_grid)
    if args.resume:
        heatmap.load_heatmap(args.heatmap_prefix)

    successes = 0
    attempts = 0
    while successes < args.num_paths and attempts < max_attempts:
        attempts += 1
        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        if args.max_start_goal_distance is not None:
            max_d = float(args.max_start_goal_distance)
            while (
                np.linalg.norm(np.array(x_goal) - np.array(x_init)) > max_d
            ):
                x_goal = generate_random_free_point(occ_grid, rng)
        if x_init == x_goal:
            continue
        if args.min_separation > 0.0:
            if np.linalg.norm(np.array(x_goal) - np.array(x_init)) < args.min_separation:
                continue

        problem = AStar(
            [0, 0],
            statespace_hi,
            x_init,
            x_goal,
            occ_grid,
            resolution=map_resolution,
            desired_dist_right_extra=0.25,
        )
        if not problem.solve(mode="modified"):
            continue

        heatmap.add_path(problem.path, increment=args.increment)
        successes += 1
        if successes % _CHECKPOINT_EVERY_N == 0:
            heatmap.save_heatmap(args.heatmap_prefix)
            print(f"Checkpoint: {successes} successful path(s) merged — heatmap saved.", flush=True)

    print(
        f"Scenario: {args.scenario} | collected {successes}/{args.num_paths} paths "
        f"in {attempts} attempts (max {max_attempts})."
    )

    heatmap.save_heatmap(args.heatmap_prefix)

    if not args.no_plot:
        heatmap.plot_heatmap()

    if successes < args.num_paths:
        print(
            f"error: only {successes} successes before hitting --max-attempts ({max_attempts}).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())