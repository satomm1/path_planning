"""Occupancy grid wrapper that composes a base map with a temporary overlay."""

from __future__ import annotations

from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.temporary_overlay import TemporaryObstacleOverlay


class PlanningGrid:
    """
    Drop-in occupancy object for A* that blocks cells covered by a temporary overlay
    while leaving the base ``StochOccupancyGrid2D.probs`` unchanged.
    """

    requires_segment_collision_check = True

    def __init__(
        self,
        base: StochOccupancyGrid2D,
        overlay: TemporaryObstacleOverlay | None = None,
    ):
        self._base = base
        self._overlay = overlay

    @property
    def base(self) -> StochOccupancyGrid2D:
        return self._base

    @property
    def overlay(self) -> TemporaryObstacleOverlay | None:
        return self._overlay

    def is_free(self, state) -> bool:
        if not self._base.is_free(state):
            return False
        if self._overlay is not None and self._overlay.has_active_obstacles():
            if self._overlay.is_state_blocked(state):
                return False
        return True

    def is_segment_free(self, p0, p1, step=None) -> bool:
        import numpy as np

        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        if step is None:
            step = max(self._base.resolution / 2.0, 1e-4)
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

    def __getattr__(self, name: str):
        return getattr(self._base, name)
