"""Shared motion limits for MAPF schedule conversion and MILP timing."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_VELOCITY_MPS = 0.7


def cell_size_m(resolution: float, downsample: int = 1) -> float:
    """World meters per coarse MAPF grid step."""
    return float(resolution) * max(1, int(downsample))


@dataclass(frozen=True)
class MotionConfig:
    """Max speed cap used for MAPF post-scheduling and MILP segment constraints."""

    max_velocity_mps: float = DEFAULT_MAX_VELOCITY_MPS
    wait_step_seconds: float | None = None

    def wait_dt(self, resolution: float, downsample: int = 1) -> float:
        """Duration of one STA* wait tick at the speed cap (matches one cardinal move)."""
        if self.wait_step_seconds is not None:
            return float(self.wait_step_seconds)
        cs = cell_size_m(resolution, downsample)
        return cs / max(self.max_velocity_mps, 1e-9)

    def to_manifest_dict(self) -> dict:
        return {
            "max_velocity_mps": float(self.max_velocity_mps),
            "wait_step_seconds": self.wait_step_seconds,
        }
