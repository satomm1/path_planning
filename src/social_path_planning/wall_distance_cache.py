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
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, TypeVar, Union

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


class WallDistanceCacheError(RuntimeError):
    """Raised when ``load_wall_distance_cache_into_grid(..., strict=True)`` cannot apply a cache."""


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


def _attempt_load_wall_distance_cache(
    path: Path, grid: "StochOccupancyGrid2D"
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Try to load ``D_right`` from ``path`` for ``grid``.

    Returns
    -------
    (array, None) on success, or (None, error_message) on failure.
    """
    path = Path(path)
    if not path.is_file():
        return None, f"wall-distance cache file not found: {path}"
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        return None, f"cannot read wall-distance cache {path}: {exc}"

    try:
        ver = int(np.asarray(data[NPZ_META_VERSION]).item())
    except (KeyError, ValueError, TypeError):
        return None, f"invalid cache metadata in {path} (missing {NPZ_META_VERSION})"

    if ver != CACHE_VERSION:
        return None, f"cache version mismatch: file has {ver}, expected {CACHE_VERSION}"

    expected_sha = fingerprint_probs(grid.probs)
    try:
        stored = str(np.asarray(data[NPZ_PROBS_SHA256]).item())
    except (KeyError, ValueError, TypeError):
        return None, f"invalid cache metadata in {path} (missing {NPZ_PROBS_SHA256})"

    if stored != expected_sha:
        return None, (
            "occupancy fingerprint mismatch: grid.probs does not match the map used to "
            "build this cache (rebuild with python -m social_path_planning.precompute_wall_distances --force)"
        )

    try:
        dt = float(np.asarray(data[NPZ_DIST_THRESH]).item())
    except (KeyError, ValueError, TypeError):
        return None, f"invalid cache metadata in {path} (missing {NPZ_DIST_THRESH})"

    if not np.isclose(dt, DEFAULT_DIST_THRESH):
        return None, f"cache dist_thresh={dt} but this code expects {DEFAULT_DIST_THRESH}"

    try:
        res = float(np.asarray(data["resolution"]).item())
        ox = float(np.asarray(data["origin_x"]).item())
        oy = float(np.asarray(data["origin_y"]).item())
        w = int(np.asarray(data["width"]).item())
        h = int(np.asarray(data["height"]).item())
    except (KeyError, ValueError, TypeError) as exc:
        return None, f"cache missing grid metadata in {path}: {exc}"

    if not np.isclose(res, grid.resolution):
        return None, f"resolution mismatch: cache={res} grid={grid.resolution}"
    if not np.isclose(ox, grid.origin_x):
        return None, f"origin_x mismatch: cache={ox} grid={grid.origin_x}"
    if not np.isclose(oy, grid.origin_y):
        return None, f"origin_y mismatch: cache={oy} grid={grid.origin_y}"
    if w != grid.width:
        return None, f"width mismatch: cache={w} grid={grid.width}"
    if h != grid.height:
        return None, f"height mismatch: cache={h} grid={grid.height}"

    d_right = np.asarray(data[NPZ_D_RIGHT], dtype=np.float64)
    if d_right.shape != (grid.height, grid.width, 8):
        return None, (
            f"{NPZ_D_RIGHT} shape {d_right.shape} != expected {(grid.height, grid.width, 8)}"
        )

    return d_right, None


def try_load_wall_distance_cache(path: Path, grid: StochOccupancyGrid2D) -> Optional[np.ndarray]:
    """Load cache if file exists and metadata matches ``grid``."""
    d_right, _err = _attempt_load_wall_distance_cache(path, grid)
    return d_right


def diagnose_wall_distance_cache(
    path: Union[str, Path], grid: "StochOccupancyGrid2D"
) -> Optional[str]:
    """
    Return ``None`` if ``path`` is a compatible cache for ``grid``, else a short reason.

    Use this when loading maps outside this repository (e.g. ROS ``map_server``) to log
    why a precomputed ``.npz`` cannot be attached before falling back to raycasting.
    """
    _d, err = _attempt_load_wall_distance_cache(Path(path), grid)
    return err


def load_wall_distance_cache_into_grid(
    grid: "StochOccupancyGrid2D",
    path: Union[str, Path],
    *,
    strict: bool = True,
    verbose: bool = True,
) -> bool:
    """
    Attach precomputed ``D_right`` from an ``.npz`` file to an existing grid.

    For workflows where ``StochOccupancyGrid2D`` is built from an external map
    (ROS occupancy grid, aligned PGM, etc.), call this **after** construction once
    ``probs`` matches the map that was used to build the cache.

    Parameters
    ----------
    grid : StochOccupancyGrid2D
        Grid whose ``probs``, ``resolution``, ``width``, ``height``, ``origin_x``,
        ``origin_y`` must match the cache metadata.
    path : str or Path
        Path to ``*_wall_dist.npz`` (same format as ``save_wall_distance_cache``).
    strict : bool
        If True (default), raise :exc:`WallDistanceCacheError` when the cache cannot
        be used. If False, return ``False`` and leave ``grid._d_right`` unchanged.
    verbose : bool
        If True, print a short message to stderr when the cache is loaded.

    Returns
    -------
    bool
        True if the cache was applied, False if ``strict`` is False and loading failed.

    Raises
    ------
    WallDistanceCacheError
        If ``strict`` is True and the cache is missing or incompatible.
    """
    d_right, err = _attempt_load_wall_distance_cache(Path(path), grid)
    if d_right is not None:
        grid._d_right = d_right
        if verbose:
            print(
                f"Using precomputed wall-distance cache: {Path(path)}",
                file=sys.stderr,
            )
        return True
    if strict:
        raise WallDistanceCacheError(err or "wall-distance cache could not be loaded")
    return False


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
