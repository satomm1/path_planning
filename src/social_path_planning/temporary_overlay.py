"""Local temporary obstacle overlay for fleet path planning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure, iterate_structure

from social_path_planning.occupancy_grid import StochOccupancyGrid2D

ShapeType = Literal["rect", "circle", "rotated_square"]
DEFAULT_PROXIMITY_RADIUS_M = 2.5


def _world_to_grid_index(value: float, resolution: float, origin: float = 0.0) -> int:
    """Match ``grid_loader._point_to_grid`` with origin offset."""
    return int(round((float(value) - origin) / resolution))


def paint_rect_on_mask(
    mask: np.ndarray,
    x_range: list[float],
    y_range: list[float],
    resolution: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> None:
    x0 = _world_to_grid_index(x_range[0], resolution, origin_x)
    x1 = _world_to_grid_index(x_range[1], resolution, origin_x)
    y0 = _world_to_grid_index(y_range[0], resolution, origin_y)
    y1 = _world_to_grid_index(y_range[1], resolution, origin_y)
    x0, x1 = sorted((max(0, x0), min(mask.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(mask.shape[0], y1)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True


def paint_circle_on_mask(
    mask: np.ndarray,
    center: list[float],
    radius: float,
    resolution: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> None:
    height, width = mask.shape
    xs = origin_x + np.arange(width) * resolution
    ys = origin_y + np.arange(height) * resolution
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask[dist <= radius] = True


def paint_rotated_square_on_mask(
    mask: np.ndarray,
    center: list[float],
    side: float,
    theta_deg: float,
    resolution: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> None:
    height, width = mask.shape
    xs = origin_x + np.arange(width) * resolution
    ys = origin_y + np.arange(height) * resolution
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    half = side / 2.0
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    Xc = X - center[0]
    Yc = Y - center[1]
    xr = c * Xc + s * Yc
    yr = -s * Xc + c * Yc
    local_mask = (np.abs(xr) <= half) & (np.abs(yr) <= half)
    mask[local_mask] = True


def paint_obstacle_spec_on_mask(
    mask: np.ndarray,
    spec: dict[str, Any],
    resolution: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> None:
    shape = spec.get("shape") or spec.get("type")
    if shape == "rect":
        paint_rect_on_mask(mask, spec["x"], spec["y"], resolution, origin_x, origin_y)
    elif shape == "circle":
        paint_circle_on_mask(mask, spec["center"], float(spec["radius"]), resolution, origin_x, origin_y)
    elif shape == "rotated_square":
        paint_rotated_square_on_mask(
            mask,
            spec["center"],
            float(spec["side"]),
            float(spec["theta_deg"]),
            resolution,
            origin_x,
            origin_y,
        )
    else:
        raise ValueError(f"Unsupported obstacle shape '{shape}'. Use rect, circle, or rotated_square.")


@dataclass
class TemporaryObstacle:
    id: str
    shape: ShapeType
    active: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> dict[str, Any]:
        spec = {"shape": self.shape, **self.params}
        return spec


class TemporaryObstacleOverlay:
    """Maintains a boolean occupancy mask for transient obstacles."""

    def __init__(self, occ_grid: StochOccupancyGrid2D):
        self.occ_grid = occ_grid
        self._obstacles: dict[str, TemporaryObstacle] = {}
        self._mask = np.zeros((occ_grid.height, occ_grid.width), dtype=bool)
        self._planning_block_mask = self._mask.copy()
        self._distance_to_obstacle_m: np.ndarray | None = None

    @property
    def distance_to_obstacle_m(self) -> np.ndarray:
        """Per-cell distance (m) to the nearest active temporary obstacle cell."""
        if self._distance_to_obstacle_m is None:
            self._rebuild_distance_map()
        return self._distance_to_obstacle_m

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def planning_block_mask(self) -> np.ndarray:
        """Obstacle cells dilated for configuration-space blocking during planning."""
        return self._planning_block_mask

    def has_active_obstacles(self) -> bool:
        return bool(np.any(self._mask))

    def list_obstacles(self) -> list[TemporaryObstacle]:
        return list(self._obstacles.values())

    def add_obstacle(self, spec: dict[str, Any]) -> str:
        obstacle_id = spec.get("id")
        if not obstacle_id:
            obstacle_id = f"obs_{len(self._obstacles)}"
        shape = spec.get("shape") or spec.get("type")
        if shape not in ("rect", "circle", "rotated_square"):
            raise ValueError(f"Unsupported obstacle shape '{shape}'.")
        params = {k: v for k, v in spec.items() if k not in ("id", "shape", "type", "active")}
        obstacle = TemporaryObstacle(
            id=str(obstacle_id),
            shape=shape,
            active=bool(spec.get("active", True)),
            params=params,
        )
        self._obstacles[obstacle.id] = obstacle
        self._rebuild_mask()
        return obstacle.id

    def set_active(self, obstacle_id: str, active: bool) -> None:
        if obstacle_id not in self._obstacles:
            raise KeyError(f"Unknown obstacle id '{obstacle_id}'.")
        self._obstacles[obstacle_id].active = active
        self._rebuild_mask()

    def remove_obstacle(self, obstacle_id: str) -> None:
        if obstacle_id not in self._obstacles:
            raise KeyError(f"Unknown obstacle id '{obstacle_id}'.")
        del self._obstacles[obstacle_id]
        self._rebuild_mask()

    def clear_all(self) -> None:
        self._obstacles.clear()
        self._mask.fill(False)
        self._planning_block_mask.fill(False)
        self._distance_to_obstacle_m = None

    def _world_to_grid(self, state: tuple[float, float]) -> tuple[int, int]:
        x, y = self.occ_grid.snap_to_grid(state)
        grid_x = _world_to_grid_index(x, self.occ_grid.resolution, self.occ_grid.origin_x)
        grid_y = _world_to_grid_index(y, self.occ_grid.resolution, self.occ_grid.origin_y)
        return grid_x, grid_y

    def is_cell_blocked(self, grid_x: int, grid_y: int) -> bool:
        if grid_x < 0 or grid_y < 0 or grid_x >= self.occ_grid.width or grid_y >= self.occ_grid.height:
            return True
        return bool(self._planning_block_mask[grid_y, grid_x])

    def is_state_blocked(self, state: tuple[float, float]) -> bool:
        grid_x, grid_y = self._world_to_grid(state)
        half = int(round(self.occ_grid.robot_d / 2 / self.occ_grid.resolution))
        x_lo = max(0, grid_x - half)
        y_lo = max(0, grid_y - half)
        x_hi = min(self.occ_grid.width, grid_x + half + 1)
        y_hi = min(self.occ_grid.height, grid_y + half + 1)
        return bool(np.any(self._planning_block_mask[y_lo:y_hi, x_lo:x_hi]))

    def _rebuild_distance_map(self) -> None:
        if not np.any(self._mask):
            self._distance_to_obstacle_m = np.full(
                self._mask.shape, np.inf, dtype=np.float32
            )
            return
        # Distance from each free cell to the nearest obstacle cell (meters).
        free_space = ~self._mask
        self._distance_to_obstacle_m = (
            distance_transform_edt(free_space) * self.occ_grid.resolution
        ).astype(np.float32)

    def distance_at_state(self, state: tuple[float, float]) -> float:
        grid_x, grid_y = self._world_to_grid(state)
        if (
            grid_x < 0
            or grid_y < 0
            or grid_x >= self.occ_grid.width
            or grid_y >= self.occ_grid.height
        ):
            return float("inf")
        return float(self.distance_to_obstacle_m[grid_y, grid_x])

    def path_within_proximity(
        self,
        path: list[tuple[float, float]],
        radius_m: float = DEFAULT_PROXIMITY_RADIUS_M,
        *,
        sample_step: float | None = None,
    ) -> bool:
        """
        Return True if any sampled point along ``path`` is within ``radius_m`` of
        an active temporary obstacle (entire path treated as affected when True).
        """
        if not self.has_active_obstacles() or len(path) < 2:
            return False
        if sample_step is None:
            sample_step = max(self.occ_grid.resolution / 2.0, 0.05)
        for i in range(len(path) - 1):
            p0 = np.asarray(path[i], dtype=float)
            p1 = np.asarray(path[i + 1], dtype=float)
            seg_len = float(np.linalg.norm(p1 - p0))
            n_steps = max(int(np.ceil(seg_len / sample_step)), 1)
            for j in range(n_steps + 1):
                t = j / n_steps
                pt = p0 + t * (p1 - p0)
                if self.distance_at_state((float(pt[0]), float(pt[1]))) <= radius_m:
                    return True
        return False

    def _rebuild_mask(self) -> None:
        self._mask.fill(False)
        for obstacle in self._obstacles.values():
            if not obstacle.active:
                continue
            paint_obstacle_spec_on_mask(
                self._mask,
                obstacle.to_spec(),
                self.occ_grid.resolution,
                self.occ_grid.origin_x,
                self.occ_grid.origin_y,
            )
        self._planning_block_mask = self._build_planning_block_mask(self._mask)
        self._distance_to_obstacle_m = None

    def _build_planning_block_mask(self, obstacle_mask: np.ndarray) -> np.ndarray:
        if not np.any(obstacle_mask):
            return np.zeros_like(obstacle_mask, dtype=bool)
        half = int(round(self.occ_grid.robot_d / 2 / self.occ_grid.resolution))
        if half <= 0:
            return obstacle_mask.copy()
        struct = iterate_structure(generate_binary_structure(2, 2), half)
        return binary_dilation(obstacle_mask, structure=struct)


def _resolve_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        module_dir = Path(__file__).resolve().parent
        default = module_dir / "environments" / "temporary_obstacles_sample2.json"
        if default.is_file():
            return default
        raise FileNotFoundError(
            "No obstacle config provided and default temporary_obstacles_sample2.json not found."
        )
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    module_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd() / path, module_dir / path, module_dir / "environments" / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path.cwd() / path


def load_temporary_obstacle_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = _resolve_config_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Temporary obstacle config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("Temporary obstacle config must be a JSON object.")
    return config


def apply_obstacle_config(overlay: TemporaryObstacleOverlay, config: dict[str, Any]) -> None:
    for spec in config.get("obstacles", []):
        overlay.add_obstacle(spec)
