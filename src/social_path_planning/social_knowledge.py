"""Permanent vs temporary social knowledge (dual heatmaps + social graph H)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from social_path_planning.heat_map import HeatMap2DVector
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.sparse_graph import FrequentSubgraph

PERMANENT_CHECKPOINT = "permanent_heatmap.npy"
TEMPORARY_CHECKPOINT = "temporary_heatmap.npy"
KNOWLEDGE_METADATA = "knowledge_metadata.json"


@dataclass
class PathBroadcast:
    robot_id: int
    path: list[tuple[float, float]]
    temporary: bool
    obstacle_affected: bool = False
    obstacle_id: str | None = None
    step: int = 0


@dataclass
class SocialKnowledgeStore:
    occ_grid: StochOccupancyGrid2D
    permanent_heatmap: HeatMap2DVector = field(init=False)
    temporary_heatmap: HeatMap2DVector = field(init=False)
    social_graph: FrequentSubgraph = field(init=False)
    broadcasts: list[PathBroadcast] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.permanent_heatmap = HeatMap2DVector(self.occ_grid)
        self.temporary_heatmap = HeatMap2DVector(self.occ_grid)
        self.social_graph = FrequentSubgraph(self.occ_grid)
        self.social_graph.set_heat_map(self.permanent_heatmap.heatmap)

    def set_permanent_heatmap(self, heatmap_array: np.ndarray) -> None:
        if heatmap_array.shape != self.permanent_heatmap.heatmap.shape:
            raise ValueError(
                f"Permanent heatmap shape mismatch: expected {self.permanent_heatmap.heatmap.shape}, "
                f"got {heatmap_array.shape}"
            )
        self.permanent_heatmap.heatmap = np.asarray(heatmap_array, dtype=float)
        self.social_graph.set_heat_map(self.permanent_heatmap.heatmap)

    def add_path(
        self,
        path: list[tuple[float, float]],
        *,
        temporary: bool = False,
        increment: float = 1.0,
    ) -> None:
        if temporary:
            self.temporary_heatmap.add_path(path, increment=increment)
        else:
            self.permanent_heatmap.add_path(path, increment=increment)
            self.social_graph.set_heat_map(self.permanent_heatmap.heatmap)

    def broadcast_path(
        self,
        robot_id: int,
        path: list[tuple[float, float]],
        *,
        temporary: bool,
        obstacle_affected: bool = False,
        add_to_permanent: bool = False,
        obstacle_id: str | None = None,
        step: int = 0,
    ) -> None:
        record = PathBroadcast(
            robot_id=robot_id,
            path=list(path),
            temporary=temporary,
            obstacle_affected=obstacle_affected,
            obstacle_id=obstacle_id,
            step=step,
        )
        self.broadcasts.append(record)
        if temporary and obstacle_affected:
            self.add_path(path, temporary=True)
        elif add_to_permanent:
            self.add_path(path, temporary=False)

    def rebuild_social_graph(
        self,
        threshold: float = 1.0,
        min_component_size: int = 15,
    ) -> None:
        self.social_graph.set_heat_map(self.permanent_heatmap.heatmap)
        self.social_graph.build_graph(threshold=threshold, reset_graph=True)
        self.social_graph.prune_graph(min_component_size=min_component_size)

    def build_adapted_graph(
        self,
        threshold: float = 1.0,
        min_component_size: int = 15,
        *,
        source: str = "temporary",
    ) -> FrequentSubgraph:
        """
        Build a preview social graph from temporary (or combined) heatmap data.

        This graph is for visualization only; the fleet's primary H uses permanent data.
        """
        adapted = FrequentSubgraph(self.occ_grid)
        if source == "temporary":
            heat = self.temporary_heatmap.heatmap
        elif source == "combined":
            heat = self.permanent_heatmap.heatmap + self.temporary_heatmap.heatmap
        else:
            raise ValueError("source must be 'temporary' or 'combined'.")
        adapted.set_heat_map(heat)
        adapted.build_graph(threshold=threshold, reset_graph=True)
        adapted.prune_graph(min_component_size=min_component_size)
        return adapted

    def promotion_candidates(self, threshold: int) -> list[tuple[int, int, int]]:
        """Return (row, col, dir_idx) cells in the temporary heatmap at or above threshold."""
        hits = np.argwhere(self.temporary_heatmap.heatmap >= threshold)
        return [(int(r), int(c), int(d)) for r, c, d in hits]

    def promote_temporary(self, threshold: int) -> int:
        """
        Merge qualifying temporary heat counts into the permanent heatmap.

        Returns the number of directed cells promoted.
        """
        candidates = self.promotion_candidates(threshold)
        if not candidates:
            return 0
        for row, col, dir_idx in candidates:
            count = float(self.temporary_heatmap.heatmap[row, col, dir_idx])
            self.permanent_heatmap.heatmap[row, col, dir_idx] += count
            self.temporary_heatmap.heatmap[row, col, dir_idx] = 0.0
        self.social_graph.set_heat_map(self.permanent_heatmap.heatmap)
        return len(candidates)

    def save_checkpoint(self, output_dir: str) -> dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        perm_path = os.path.join(output_dir, PERMANENT_CHECKPOINT)
        temp_path = os.path.join(output_dir, TEMPORARY_CHECKPOINT)
        np.save(perm_path, self.permanent_heatmap.heatmap)
        np.save(temp_path, self.temporary_heatmap.heatmap)
        return {"permanent": perm_path, "temporary": temp_path}

    def load_permanent_checkpoint(self, path: str) -> None:
        arr = np.load(path)
        self.set_permanent_heatmap(arr)

    def load_temporary_checkpoint(self, path: str) -> None:
        arr = np.load(path)
        if arr.shape != self.temporary_heatmap.heatmap.shape:
            raise ValueError(f"Temporary heatmap shape mismatch: got {arr.shape}")
        self.temporary_heatmap.heatmap = np.asarray(arr, dtype=float)

    def summary(self) -> dict[str, Any]:
        perm_total = float(np.sum(self.permanent_heatmap.heatmap))
        temp_total = float(np.sum(self.temporary_heatmap.heatmap))
        temp_broadcasts = [b for b in self.broadcasts if b.temporary]
        return {
            "permanent_path_count_proxy": perm_total,
            "temporary_path_count_proxy": temp_total,
            "permanent_broadcasts": sum(1 for b in self.broadcasts if not b.temporary),
            "temporary_broadcasts": sum(1 for b in temp_broadcasts if b.obstacle_affected),
            "temporary_replanned_total": len(temp_broadcasts),
            "temporary_unaffected_replans": sum(
                1 for b in temp_broadcasts if not b.obstacle_affected
            ),
            "social_graph_nodes": self.social_graph.graph.number_of_nodes(),
            "social_graph_edges": self.social_graph.graph.number_of_edges(),
        }

    def save_metadata(self, output_dir: str, extra: dict[str, Any] | None = None) -> str:
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, KNOWLEDGE_METADATA)
        payload = self.summary()
        if extra:
            payload.update(extra)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return meta_path
