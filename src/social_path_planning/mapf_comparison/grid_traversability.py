"""Single source of truth for occupancy / MAPF traversability."""

from __future__ import annotations

from typing import List, Literal, Sequence, Tuple

import numpy as np

Coord = Tuple[int, int]

CoarseBlockPolicy = Literal["any", "all", "majority", "center", "fine_center"]
COARSE_BLOCK_POLICIES: tuple[CoarseBlockPolicy, ...] = (
    "any",
    "all",
    "majority",
    "center",
    "fine_center",
)
DEFAULT_COARSE_BLOCK_POLICY: CoarseBlockPolicy = "fine_center"
ROBOT_DIAMETER_M = 0.5


def normalize_coarse_block_policy(policy: str) -> CoarseBlockPolicy:
    key = str(policy).strip().lower()
    if key not in COARSE_BLOCK_POLICIES:
        raise ValueError(
            f"coarse_block_policy must be one of {COARSE_BLOCK_POLICIES}, got {policy!r}"
        )
    return key  # type: ignore[return-value]


def fine_cell_center_world(occ_grid, col: int, row: int) -> Tuple[float, float]:
    res = float(occ_grid.resolution)
    x = occ_grid.origin_x + (col + 0.5) * res
    y = occ_grid.origin_y + (row + 0.5) * res
    return (float(x), float(y))


def fine_cell_corner_world(occ_grid, col: int, row: int) -> Tuple[float, float]:
    x = occ_grid.origin_x + col * occ_grid.resolution
    y = occ_grid.origin_y + row * occ_grid.resolution
    return (float(x), float(y))


def world_to_mapf_cell(occ_grid, xy: Sequence[float], downsample: int = 1) -> Coord:
    col = int(np.round((float(xy[0]) - occ_grid.origin_x) / occ_grid.resolution))
    row = int(np.round((float(xy[1]) - occ_grid.origin_y) / occ_grid.resolution))
    ds = max(1, int(downsample))
    return (col // ds, row // ds)


def mapf_cell_to_world(occ_grid, cell: Coord, downsample: int = 1) -> Tuple[float, float]:
    ds = max(1, int(downsample))
    col, row = cell
    x = occ_grid.origin_x + (col * ds + ds / 2.0) * occ_grid.resolution
    y = occ_grid.origin_y + (row * ds + ds / 2.0) * occ_grid.resolution
    return (float(x), float(y))


def _boundary_frame(col_min: int, col_max: int, row_min: int, row_max: int) -> set:
    frame = set()
    for col in range(col_min - 1, col_max + 2):
        frame.add((col, row_min - 1))
        frame.add((col, row_max + 1))
    for row in range(row_min - 1, row_max + 2):
        frame.add((col_min - 1, row))
        frame.add((col_max + 1, row))
    return frame


class TraversabilityModel:
    """
    Unified static passability for path bank A*, MAPF rasterization, and validation.
    """

    def __init__(
        self,
        occ_grid,
        downsample: int = 1,
        coarse_block_policy: str = DEFAULT_COARSE_BLOCK_POLICY,
    ):
        self.occ_grid = occ_grid
        self.downsample = max(1, int(downsample))
        self.coarse_block_policy = normalize_coarse_block_policy(coarse_block_policy)

    def is_world_pose_free(self, x: float, y: float) -> bool:
        return bool(self.occ_grid.is_free((float(x), float(y))))

    def is_fine_center_free(self, col: int, row: int) -> bool:
        if col < 0 or row < 0 or col >= self.occ_grid.width or row >= self.occ_grid.height:
            return False
        return self.is_world_pose_free(*fine_cell_center_world(self.occ_grid, col, row))

    def is_fine_corner_blocked(self, col: int, row: int) -> bool:
        if col < 0 or row < 0 or col >= self.occ_grid.width or row >= self.occ_grid.height:
            return True
        return not self.is_world_pose_free(*fine_cell_corner_world(self.occ_grid, col, row))

    def is_coarse_cell_free(self, coarse_col: int, coarse_row: int) -> bool:
        return not self._coarse_cell_is_blocked(coarse_col, coarse_row)

    def _coarse_cell_is_blocked(self, coarse_col: int, coarse_row: int) -> bool:
        policy = self.coarse_block_policy
        ds = self.downsample
        occ = self.occ_grid

        if policy == "fine_center":
            for dr in range(ds):
                for dc in range(ds):
                    if self.is_fine_center_free(
                        coarse_col * ds + dc, coarse_row * ds + dr
                    ):
                        return False
            return True

        if policy == "center":
            dc = dr = ds // 2
            return self.is_fine_corner_blocked(
                coarse_col * ds + dc, coarse_row * ds + dr
            )

        blocked = 0
        total = ds * ds
        for dr in range(ds):
            for dc in range(ds):
                if self.is_fine_corner_blocked(
                    coarse_col * ds + dc, coarse_row * ds + dr
                ):
                    if policy == "any":
                        return True
                    blocked += 1

        if policy == "all":
            return blocked == total
        if policy == "majority":
            return blocked > total // 2
        return blocked > 0

    def static_obstacle_cells(
        self,
        crop_bounds: Tuple[int, int, int, int] | None = None,
    ) -> List[Coord]:
        occ = self.occ_grid
        if crop_bounds is None:
            col_min, col_max = 0, occ.width - 1
            row_min, row_max = 0, occ.height - 1
        else:
            col_min, col_max, row_min, row_max = crop_bounds

        ds = self.downsample
        coarse_col_min = col_min // ds
        coarse_col_max = col_max // ds
        coarse_row_min = row_min // ds
        coarse_row_max = row_max // ds

        blocked: List[Coord] = []
        for coarse_row in range(coarse_row_min, coarse_row_max + 1):
            for coarse_col in range(coarse_col_min, coarse_col_max + 1):
                if self._coarse_cell_is_blocked(coarse_col, coarse_row):
                    blocked.append((coarse_col, coarse_row))

        all_obs = set(blocked) | _boundary_frame(
            coarse_col_min, coarse_col_max, coarse_row_min, coarse_row_max
        )
        return sorted(all_obs)

    def world_to_cell(self, xy: Sequence[float]) -> Coord:
        return world_to_mapf_cell(self.occ_grid, xy, self.downsample)

    def cell_to_world(self, cell: Coord) -> Tuple[float, float]:
        return mapf_cell_to_world(self.occ_grid, cell, self.downsample)

    def count_static_collisions(
        self,
        world_waypoints: Sequence[Sequence[float]],
        *,
        check_segments: bool = True,
    ) -> int:
        """Count waypoints (and optionally segments) that violate static occupancy."""
        collisions = 0
        pts = [tuple(float(v) for v in p[:2]) for p in world_waypoints if len(p) >= 2]
        for x, y in pts:
            if not self.is_world_pose_free(x, y):
                collisions += 1
        if check_segments and len(pts) >= 2 and hasattr(self.occ_grid, "is_segment_free"):
            for i in range(len(pts) - 1):
                if not self.occ_grid.is_segment_free(pts[i], pts[i + 1]):
                    collisions += 1
        return collisions

    def count_mapf_path_static_collisions(self, path) -> int:
        """
        Validate a STA* / CBS / PP grid path against the same coarse rules used for obstacles.
        """
        if path is None or len(path) == 0:
            return 0
        collisions = 0
        prev = None
        for point in path:
            cell = (int(point[0]), int(point[1]))
            if cell == prev:
                continue
            if not self.is_coarse_cell_free(cell[0], cell[1]):
                collisions += 1
            prev = cell
        return collisions

    def validation_radius_cells(self) -> int:
        import math

        coarse_res = float(self.occ_grid.resolution) * self.downsample
        return max(1, int(math.ceil(ROBOT_DIAMETER_M / (2.0 * coarse_res))))
