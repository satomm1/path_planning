"""
Precomputed directional distances to the first wall on the geometric "right" of motion.

Tensor shape (height, width, 8): for each grid cell and each neighbor direction (same
ordering as AStar.get_neighbors), stores dist_to_wall_right in meters.

dist_to_wall_left(x, T) equals dist_to_wall_right(x, -T); direction index for -T is
opposite_dir_idx(k) where k is the index for T.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, TypeVar

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

T = TypeVar("T")


def _maybe_tqdm(
    iterable: Iterable[T],
    *,
    show_progress: bool,
    **kwargs,
) -> Iterable[T]:
    if not show_progress or tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)

if TYPE_CHECKING:
    from social_path_planning.occupancy_grid import StochOccupancyGrid2D

# Same nested order as AStar.get_neighbors: for ii in [1,0,-1]: for jj in [1,0,-1]
NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = tuple(
    (ii, jj)
    for ii in (1, 0, -1)
    for jj in (1, 0, -1)
    if ii != 0 or jj != 0
)
assert len(NEIGHBOR_OFFSETS) == 8

# Unit travel directions (for dot-product matching)
_UNIT_TRAVEL_DIRS = np.array(
    [
        [ii / np.sqrt(ii * ii + jj * jj), jj / np.sqrt(ii * ii + jj * jj)]
        for ii, jj in NEIGHBOR_OFFSETS
    ],
    dtype=np.float64,
)

CACHE_VERSION = 1
DEFAULT_DIST_THRESH = 15.0
NPZ_D_RIGHT = "D_right"
NPZ_META_VERSION = "cache_version"
NPZ_PROBS_SHA256 = "probs_sha256"
NPZ_DIST_THRESH = "dist_thresh"


def opposite_dir_idx(k: int) -> int:
    """Index for travel direction -T given index k for T (NEIGHBOR_OFFSETS pairing)."""
    return 7 - k


def travel_dir_to_dir_idx(travel_dir) -> int:
    """
    Map a (possibly noisy) unit vector to the nearest NEIGHBOR_OFFSETS direction index.
    """
    t = np.asarray(travel_dir, dtype=np.float64).reshape(2)
    n = np.linalg.norm(t)
    if n < 1e-9:
        return 0
    t = t / n
    dots = _UNIT_TRAVEL_DIRS @ t
    return int(np.argmax(dots))


def fingerprint_probs(probs: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(probs).tobytes()).hexdigest()


def world_xy_for_cell_indices(
    origin_x: float, origin_y: float, resolution: float, col: int, row: int
) -> Tuple[float, float]:
    """World position aligned with snapped grid vertices (same as discrete A* states)."""
    return (
        origin_x + col * resolution,
        origin_y + row * resolution,
    )


def precompute_d_right(
    grid: StochOccupancyGrid2D,
    dist_thresh: float = DEFAULT_DIST_THRESH,
    *,
    show_progress: Optional[bool] = None,
) -> np.ndarray:
    """Fill (height, width, 8) array using the grid's raycast implementation."""
    if show_progress is None:
        show_progress = sys.stderr.isatty()

    h, w = grid.height, grid.width
    out = np.empty((h, w, 8), dtype=np.float64)
    row_iter = _maybe_tqdm(
        range(h),
        show_progress=show_progress,
        desc="Wall distance precompute",
        unit="row",
        total=h,
        file=sys.stderr,
    )
    for row in row_iter:
        for col in range(w):
            wx, wy = world_xy_for_cell_indices(
                grid.origin_x, grid.origin_y, grid.resolution, col, row
            )
            x = (wx, wy)
            for k, (ii, jj) in enumerate(NEIGHBOR_OFFSETS):
                tnorm = np.sqrt(ii * ii + jj * jj)
                travel_dir = (ii / tnorm, jj / tnorm)
                out[row, col, k] = grid._dist_to_wall_right_raycast(
                    x, travel_dir, dist_thresh=dist_thresh
                )
    return out


def save_wall_distance_cache(
    path: Path,
    grid: StochOccupancyGrid2D,
    d_right: np.ndarray,
    dist_thresh: float = DEFAULT_DIST_THRESH,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = fingerprint_probs(grid.probs)
    np.savez_compressed(
        path,
        **{
            NPZ_D_RIGHT: d_right.astype(np.float32),
            NPZ_META_VERSION: np.array(CACHE_VERSION),
            NPZ_PROBS_SHA256: np.array(sha),
            NPZ_DIST_THRESH: np.array(dist_thresh),
            "resolution": np.array(grid.resolution),
            "origin_x": np.array(grid.origin_x),
            "origin_y": np.array(grid.origin_y),
            "width": np.array(grid.width),
            "height": np.array(grid.height),
            "neighbor_offsets_version": np.array(CACHE_VERSION),
        },
    )


def try_load_wall_distance_cache(path: Path, grid: StochOccupancyGrid2D) -> Optional[np.ndarray]:
    """Load cache if file exists and metadata matches ``grid``."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None

    if int(np.asarray(data[NPZ_META_VERSION]).item()) != CACHE_VERSION:
        return None

    expected_sha = fingerprint_probs(grid.probs)
    stored = str(np.asarray(data[NPZ_PROBS_SHA256]).item())
    if stored != expected_sha:
        return None

    if not np.isclose(
        float(np.asarray(data[NPZ_DIST_THRESH]).item()), DEFAULT_DIST_THRESH
    ):
        return None

    if (
        not np.isclose(float(np.asarray(data["resolution"]).item()), grid.resolution)
        or not np.isclose(float(np.asarray(data["origin_x"]).item()), grid.origin_x)
        or not np.isclose(float(np.asarray(data["origin_y"]).item()), grid.origin_y)
        or int(np.asarray(data["width"]).item()) != grid.width
        or int(np.asarray(data["height"]).item()) != grid.height
    ):
        return None

    d_right = np.asarray(data[NPZ_D_RIGHT], dtype=np.float64)
    if d_right.shape != (grid.height, grid.width, 8):
        return None

    return d_right


def attach_wall_distance_cache(
    grid: StochOccupancyGrid2D,
    cache_path: Optional[Path],
    *,
    auto_build: bool = False,
    dist_thresh: float = DEFAULT_DIST_THRESH,
) -> None:
    """
    Set ``grid._d_right`` from file, or build and save when ``auto_build`` is True.
    If no cache is available, ``grid._d_right`` stays None (runtime raycast).
    """
    if cache_path is None:
        return

    cache_path = Path(cache_path)
    loaded = try_load_wall_distance_cache(cache_path, grid)
    if loaded is not None:
        grid._d_right = loaded
        print(
            f"Using precomputed wall-distance cache: {cache_path}",
            file=sys.stderr,
        )
        return

    if not auto_build:
        return

    d_right = precompute_d_right(grid, dist_thresh=dist_thresh)
    save_wall_distance_cache(cache_path, grid, d_right, dist_thresh=dist_thresh)
    grid._d_right = d_right
    print(
        f"Built and saved wall-distance cache (now in use): {cache_path}",
        file=sys.stderr,
    )
