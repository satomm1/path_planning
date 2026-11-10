"""Small shared helpers used across planners, benchmarks and demos.

Kept deliberately light on imports so that importing it does not pull in the
MAPF comparison stack, cvxpy or plotting code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.mapf_comparison.grid_traversability import ROBOT_DIAMETER_M
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.utils import snap_to_grid

MODULE_DIR = Path(__file__).resolve().parent

# Directional heatmap channel order (``HeatMap2DVector``, social graph, route entropy):
# 0 NW, 1 N, 2 NE, 3 E, 4 W, 5 SW, 6 S, 7 SE, as (dx, dy) grid steps.
HEATMAP_DIRECTIONS = (
    (-1, 1),   # NW
    (0, 1),    # N
    (1, 1),    # NE
    (1, 0),    # E
    (-1, 0),   # W
    (-1, -1),  # SW
    (0, -1),   # S
    (1, -1),   # SE
)
HEATMAP_DIRECTION_INDEX = {direction: idx for idx, direction in enumerate(HEATMAP_DIRECTIONS)}

WALL_CACHE_POLICIES = ("auto_build", "if_present", "none")


def wall_distance_cache_path(scenario_name: str) -> Path:
    """Default location of the precomputed wall-distance cache for a scenario."""
    return MODULE_DIR / "environments" / f"{scenario_name}_wall_dist.npz"


def build_occ_grid(scenario_name, *, wall_cache="auto_build"):
    """
    Load a scenario and wrap it in a ``StochOccupancyGrid2D``.

    ``wall_cache`` selects how the precomputed wall-distance cache is used:

    - ``"auto_build"``: use ``<scenario>_wall_dist.npz`` when the file exists, and
      rebuild it if it is present but incompatible.
    - ``"if_present"``: use the file when it exists, never build one, and print a
      notice when it is missing (runtime raycast).
    - ``"none"``: ignore any cache file (runtime raycast).

    Returns ``(occ_grid, map_size, map_resolution, statespace_hi)``.
    """
    if wall_cache not in WALL_CACHE_POLICIES:
        raise ValueError(f"wall_cache must be one of {WALL_CACHE_POLICIES}, got {wall_cache!r}")
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    map_dim = [round(map_size[i] / map_resolution) for i in range(2)]
    cache_kwargs = {}
    if wall_cache != "none":
        cache_path = wall_distance_cache_path(scenario_name)
        present = cache_path.is_file()
        if wall_cache == "if_present" and not present:
            print(
                f"No wall-distance cache at {cache_path}; "
                "modified A* will use runtime raycast (slower)."
            )
        cache_kwargs = {
            "wall_distance_cache_path": cache_path if present else None,
            "auto_build_wall_distance_cache": wall_cache == "auto_build",
        }
    occ_grid = StochOccupancyGrid2D(
        map_resolution,
        map_dim[0],
        map_dim[1],
        0,
        0,
        10,
        occ.T,
        robot_d=ROBOT_DIAMETER_M,
        **cache_kwargs,
    )
    statespace_hi = snap_to_grid(map_size, map_resolution)
    return occ_grid, map_size, map_resolution, statespace_hi


def generate_random_free_point(occ_grid, rng):
    """Sample a uniformly random snapped free state from ``occ_grid`` using ``rng.uniform``."""
    low = np.array([occ_grid.origin_x, occ_grid.origin_y], dtype=float)
    high = np.array(
        [
            occ_grid.origin_x + occ_grid.width * occ_grid.resolution,
            occ_grid.origin_y + occ_grid.height * occ_grid.resolution,
        ],
        dtype=float,
    )
    while True:
        xy = rng.uniform(low=low, high=high)
        x, y = occ_grid.snap_to_grid(xy)
        if occ_grid.is_free((x, y)):
            return (x, y)


def compute_path_length(path):
    """Sum of Euclidean segment lengths of a polyline (0 for fewer than two points)."""
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        p0 = np.array(path[i], dtype=float)
        p1 = np.array(path[i + 1], dtype=float)
        total += float(np.linalg.norm(p1 - p0))
    return total


def route_endpoint_key(x_init, x_goal):
    """Hashable ``(x0, y0, x1, y1)`` key identifying a start/goal pair."""
    return (
        float(x_init[0]),
        float(x_init[1]),
        float(x_goal[0]),
        float(x_goal[1]),
    )


def path_to_serializable(path):
    """Convert a polyline to JSON-friendly ``[[x, y], ...]``."""
    return [[float(p[0]), float(p[1])] for p in path]


def path_from_serializable(path_data):
    """Inverse of :func:`path_to_serializable` (keeps every coordinate of each point)."""
    return [tuple(float(v) for v in point) for point in path_data]


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def modified_solver_config(heatmap_prefix, heatmap_file, sparse_graph_threshold, sparse_min_component_size):
    """Describe how the "modified" (social) solver is configured, for cache/resume metadata."""
    if heatmap_prefix or heatmap_file:
        return {
            "modified_solver": "heatmap_graph",
            "heatmap_prefix": heatmap_prefix,
            "heatmap_file": heatmap_file,
            "sparse_graph_threshold": sparse_graph_threshold,
            "sparse_min_component_size": sparse_min_component_size,
        }
    return {"modified_solver": "rightness_penalty"}
