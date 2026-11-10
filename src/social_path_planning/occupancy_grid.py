import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import distance_transform_edt

from social_path_planning.wall_distance_cache import (
    attach_wall_distance_cache,
    DEFAULT_DIST_THRESH,
    travel_dir_to_dir_idx,
)

# Ray-cast samples per vectorized batch: a small first batch for walls close by
# (typical in corridors), then larger batches for open areas.
_RAYCAST_FIRST_CHUNK = 16
_RAYCAST_CHUNK = 128


class StochOccupancyGrid2D(object):
    def __init__(
        self,
        resolution,
        width,
        height,
        origin_x,
        origin_y,
        window_size,
        probs,
        thresh=0.5,
        robot_d=0.25,
        wall_distance_cache_path=None,
        auto_build_wall_distance_cache=False,
    ):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.probs = np.reshape(np.asarray(probs), (height, width))
        self.window_size = 10 # window_size
        self.thresh = thresh
        self.robot_d = robot_d
        self._d_right = None
        self._free_lookup_cache = None

        self.extent = [self.origin_x, self.origin_x + self.width * self.resolution,
                       self.origin_y, self.origin_y + self.height * self.resolution]
        self.cmap = ListedColormap(['gray', '#9DC6F2', 'black'])
        self.bounds = [-1.5, -0.5, 0.5, 1.5]
        self.norm = BoundaryNorm(self.bounds, self.cmap.N)

        # Precompute the distance map
        self.distance_map = None
        self.compute_distance_map()

        attach_wall_distance_cache(
            self,
            wall_distance_cache_path,
            auto_build=auto_build_wall_distance_cache,
        )

    def snap_to_grid(self, x):
        return self.resolution * round(x[0] / self.resolution), self.resolution * round(x[1] / self.resolution)

    def snap_to_grid1(self, x):
        return self.resolution * np.round(x[0] / self.resolution), self.resolution * np.round(x[1] / self.resolution)

    def get_index(self, x):
        return np.round((x[0] - self.origin_x) / self.resolution), np.round((x[1] - self.origin_y) / self.resolution)

    def compute_distance_map(self):
        binary_map = (self.probs >= 0.5) | (self.probs < 0)
        reverse_map = 1 - binary_map

        self.distance_map = distance_transform_edt(reverse_map) * self.resolution

    def get_distance_to_obstacle(self, pose):
        x, y, _ = pose
        x_idx, y_idx = self.get_index((x, y))
        x_idx = int(np.clip(x_idx, 0, self.width - 1))
        y_idx = int(np.clip(y_idx, 0, self.height - 1))
        return self.distance_map[y_idx, x_idx]

    def _free_lookup(self):
        """
        Row-major nested list ``free[grid_y][grid_x]`` with the configuration-space
        result of :meth:`_is_free_window` for every in-bounds cell index.

        Built lazily from a 2-D prefix sum over blocked cells and rebuilt when
        ``probs`` is reassigned or ``robot_d`` / ``resolution`` change. Call
        :meth:`invalidate_free_lookup` after mutating ``probs`` in place.
        """
        cache = self._free_lookup_cache
        if (
            cache is not None
            and cache[0] is self.probs
            and cache[1] == self.robot_d
            and cache[2] == self.resolution
        ):
            return cache[3]

        probs = self.probs
        height, width = self.height, self.width
        if np.shape(probs) != (height, width):
            # Unusual layout: let is_free use the per-call window check.
            self._free_lookup_cache = (probs, self.robot_d, self.resolution, None)
            return None
        half = int(round(self.robot_d / 2 / self.resolution))
        blocked = (probs > 0.85) | (probs < 0)
        prefix = np.zeros((height + 1, width + 1), dtype=np.int64)
        prefix[1:, 1:] = np.cumsum(np.cumsum(blocked, axis=0, dtype=np.int64), axis=1)
        # Same window as _is_free_window: rows [gy - half, gy + half), cols likewise, clipped.
        rows = np.arange(height)
        cols = np.arange(width)
        y0 = np.maximum(0, rows - half)
        y1 = np.minimum(height, rows + half)
        x0 = np.maximum(0, cols - half)
        x1 = np.minimum(width, cols + half)
        counts = (
            prefix[np.ix_(y1, x1)]
            - prefix[np.ix_(y0, x1)]
            - prefix[np.ix_(y1, x0)]
            + prefix[np.ix_(y0, x0)]
        )
        free = (counts == 0).tolist()
        self._free_lookup_cache = (probs, self.robot_d, self.resolution, free)
        return free

    def invalidate_free_lookup(self):
        """Discard the cached free-space lookup (needed only after in-place edits to ``probs``)."""
        self._free_lookup_cache = None

    def is_free(self, state):
        # combine the probabilities of each cell by assuming independence
        # of each estimation
        x, y = self.snap_to_grid(state)
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            free = self._free_lookup()
            if free is not None:
                return free[grid_y][grid_x]
        return self._is_free_window(grid_x, grid_y)

    def _is_free_window(self, grid_x, grid_y):
        """Reference configuration-space check for one cell index (any index, even off-map)."""
        # Check if in configuration space
        half_config_size = int(round(self.robot_d/2/self.resolution))
        config_x_lower = max(0, grid_x - half_config_size)
        config_y_lower = max(0, grid_y - half_config_size)
        config_x_upper = min(self.width, grid_x + half_config_size)
        config_y_upper = min(self.height, grid_y + half_config_size)
        
        if np.sum(self.probs[config_y_lower:config_y_upper, config_x_lower:config_x_upper]>0.85):
            return False  # Not free according to configuration space
        elif np.sum(self.probs[config_y_lower:config_y_upper, config_x_lower:config_x_upper]< 0):
            return False
        else:
            return True

    def is_segment_free(self, p0, p1, step=None):
        """Return True if every sampled point along segment p0->p1 is collision-free."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        if step is None:
            step = max(self.resolution / 2.0, 1e-4)
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < 1e-9:
            return self.is_free(tuple(p0))
        n_steps = max(int(np.ceil(seg_len / step)), 1)
        for i in range(n_steps + 1):
            t = i / n_steps
            pt = p0 + t * (p1 - p0)
            if not self.is_free((float(pt[0]), float(pt[1]))):
                return False
        return True

    def dist_to_wall_left(self, x, travel_dir, dist_thresh=15.0):
        return self.dist_to_wall_right(x, [-travel_dir[0], -travel_dir[1]], dist_thresh=dist_thresh)

    def dist_to_wall_right(self, x, travel_dir, dist_thresh=15.0):
        """
        Return distance (meters) from world position x=(x,y) to the first occupied or
        unknown cell found to the right of the travel_dir. If no wall is found inside
        the map bounds, returns 0.
        """


        if (
            self._d_right is not None
            and abs(float(dist_thresh) - DEFAULT_DIST_THRESH) < 1e-9
        ):
            col = int(np.round((x[0] - self.origin_x) / self.resolution))
            row = int(np.round((x[1] - self.origin_y) / self.resolution))
            col = int(np.clip(col, 0, self.width - 1))
            row = int(np.clip(row, 0, self.height - 1))
            k = travel_dir_to_dir_idx(travel_dir)
            return float(self._d_right[row, col, k])

        return self._dist_to_wall_right_raycast(x, travel_dir, dist_thresh=dist_thresh)

    def _dist_to_wall_right_raycast(self, x, travel_dir, dist_thresh=15.0):
        """Ray-march implementation used for planning when no cache and for precompute."""
        right = np.array([travel_dir[1], -travel_dir[0]], dtype=float)
        rn = np.linalg.norm(right)
        if rn < 1e-12:
            return 100.0
        right = right / rn

        # map bounds in meters
        x_min = self.origin_x
        x_max = self.origin_x + self.width * self.resolution
        y_min = self.origin_y
        y_max = self.origin_y + self.height * self.resolution

        # sampling parameters
        step = max(self.resolution * 0.25, 1e-4)   # quarter cell steps (or small eps)
        # maximum distance to search: distance to map border along right direction
        # compute intersection of ray x + t*right with map bounding box to get a safe upper bound
        t_max = 0.0
        # compute candidate distances to each vertical boundary
        if right[0] > 0:
            t_max = max(t_max, (x_max - x[0]) / right[0])
        elif right[0] < 0:
            t_max = max(t_max, (x_min - x[0]) / right[0])
        if right[1] > 0:
            t_max = max(t_max, (y_max - x[1]) / right[1])
        elif right[1] < 0:
            t_max = max(t_max, (y_min - x[1]) / right[1])
        # if right component is 0 that boundary doesn't constrain; t_max may remain 0 if both comps lead outward.
        # ensure positive t_max
        if t_max <= 0:
            # ray immediately points out of bounds — no wall reachable to the right inside the map
            return 100
        if t_max > dist_thresh:
            t_max = dist_thresh

        # Sample along the ray at t = i * step, i = 1..n_steps, stopping at the first sample
        # that leaves the map (-> 100) or lands in an occupied/unknown cell (-> t).
        # Samples are evaluated in vectorized chunks; each element uses exactly the same
        # floating-point operations as a scalar loop would, so results are bit-identical.
        n_steps = int(np.ceil(t_max / step))
        px = float(x[0])
        py = float(x[1])
        rx = right[0]
        ry = right[1]
        start = 1
        chunk = _RAYCAST_FIRST_CHUNK
        while start <= n_steps:
            stop = min(n_steps, start + chunk - 1)
            t = np.arange(start, stop + 1) * step
            sx = px + rx * t
            sy = py + ry * t
            # map bounds, then grid indices (self.probs is indexed [row(y), col(x)])
            outside = (sx < x_min) | (sx >= x_max) | (sy < y_min) | (sy >= y_max)
            col = np.floor((sx - self.origin_x) / self.resolution)
            row = np.floor((sy - self.origin_y) / self.resolution)
            outside |= (row < 0) | (row >= self.height) | (col < 0) | (col >= self.width)
            stop_here = outside.copy()
            inside = ~outside
            if inside.any():
                p = self.probs[row[inside].astype(np.intp), col[inside].astype(np.intp)]
                stop_here[inside] = (p >= self.thresh) | (p < 0)
            if stop_here.any():
                k = int(np.argmax(stop_here))
                if outside[k]:
                    return 100
                return float(t[k])
            start = stop + 1
            chunk = _RAYCAST_CHUNK
        return 100

    def plot_grid(self, ax=None):
        if ax is not None:
            ax.imshow(self.probs, cmap=self.cmap, norm=self.norm, interpolation='nearest', origin='lower', extent=self.extent, aspect='equal')
        else:
            plt.imshow(self.probs, cmap=self.cmap, norm=self.norm, interpolation='nearest', origin='lower', extent=self.extent, aspect='equal')

    def plot_path(self, path, ax=None):
        xs, ys = zip(*path)
        if ax is not None:
            ax.plot(xs, ys, color='navy', linewidth=2, marker=',', zorder=5, alpha=0.75, label='Planned Path')
        else:
            plt.plot(xs, ys, color='navy', linewidth=2, marker=',', zorder=5, alpha=0.75, label='Planned Path')

    def plot_smoothed_path(self, path):
        xs, ys = zip(*path)
        plt.plot(xs, ys, color='green', linewidth=1, marker=',', zorder=6)
    
    def plot_grid_and_path(self, path, ax=None):
        self.plot_grid(ax=ax)
        self.plot_path(path, ax=ax)