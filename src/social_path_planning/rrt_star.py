from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from social_path_planning.social_cost import euclidean_distance, rightness_penalty

if TYPE_CHECKING:
    from social_path_planning.occupancy_grid import StochOccupancyGrid2D


@dataclass
class _RRTNode:
    x: tuple[float, float]
    parent: int | None
    cost: float
    dist_to_right: float


class RRTStar:
    """2D holonomic RRT* on a stochastic occupancy grid."""

    SUPPORTED_SOLVER_MODES = {"modified", "vanilla"}

    def __init__(
        self,
        statespace_lo,
        statespace_hi,
        x_init,
        x_goal,
        occupancy: StochOccupancyGrid2D,
        resolution=1,
        robot_d=0.4,
        desired_dist_right_extra=0.25,
        *,
        max_iterations=5000,
        step_size=None,
        goal_sample_rate=0.10,
        goal_tolerance=None,
        rewire_radius=None,
        timeout_vanilla=150.0,
        timeout_modified=120.0,
    ):
        self.statespace_lo = np.array(statespace_lo, dtype=float)
        self.statespace_hi = np.array(statespace_hi, dtype=float)
        self.occupancy = occupancy
        self.resolution = float(resolution)
        self.robot_d = float(robot_d)
        self.desired_dist_right_extra = float(desired_dist_right_extra)

        self.x_init = tuple(map(float, x_init))
        self.x_goal = tuple(map(float, x_goal))

        self.max_iterations = int(max_iterations)
        self.step_size = float(step_size if step_size is not None else resolution)
        self.goal_sample_rate = float(goal_sample_rate)
        self.goal_tolerance = float(goal_tolerance if goal_tolerance is not None else resolution)
        self.rewire_radius = float(rewire_radius if rewire_radius is not None else 2.0 * self.step_size)
        self.timeout_vanilla = float(timeout_vanilla)
        self.timeout_modified = float(timeout_modified)

        self.path = None
        self.last_solve_time = None
        self.last_tree = None
        self.last_solve_stats = None

    def is_free(self, x):
        x = (float(x[0]), float(x[1]))
        if not (
            self.statespace_lo[0] <= x[0] < self.statespace_hi[0]
            and self.statespace_lo[1] <= x[1] < self.statespace_hi[1]
        ):
            return False
        return self.occupancy.is_free(x)

    def is_segment_free(self, p0, p1):
        return self.occupancy.is_segment_free(p0, p1)

    def distance(self, x1, x2):
        return euclidean_distance(x1, x2)

    def get_index(self, x):
        return (
            int(np.round((x[0] - self.occupancy.origin_x) / self.resolution)),
            int(np.round((x[1] - self.occupancy.origin_y) / self.resolution)),
        )

    def vanilla_edge_cost(self, x1, x2):
        return self.distance(x1, x2)

    def modified_edge_cost(self, x1, x2, dist2right_prev=0):
        dist = self.distance(x1, x2)
        social_cost, dist2right = rightness_penalty(
            self.occupancy,
            self.x_init,
            self.x_goal,
            x1,
            x2,
            dist2right_prev,
            robot_d=self.robot_d,
            desired_dist_right_extra=self.desired_dist_right_extra,
            resolution=self.resolution,
        )
        return dist + social_cost, dist2right

    def edge_cost(self, mode, x1, x2, dist2right_prev=0):
        if mode == "vanilla":
            return self.vanilla_edge_cost(x1, x2), 0.0
        edge_total, dist2right = self.modified_edge_cost(x1, x2, dist2right_prev)
        return edge_total, dist2right

    def _sample_free(self, rng):
        while True:
            if rng.random() < self.goal_sample_rate:
                candidate = self.x_goal
            else:
                candidate = (
                    float(rng.uniform(self.statespace_lo[0], self.statespace_hi[0])),
                    float(rng.uniform(self.statespace_lo[1], self.statespace_hi[1])),
                )
            if self.is_free(candidate):
                return candidate

    def _steer(self, from_xy, to_xy):
        from_xy = np.array(from_xy, dtype=float)
        to_xy = np.array(to_xy, dtype=float)
        delta = to_xy - from_xy
        dist = float(np.linalg.norm(delta))
        if dist <= self.step_size:
            new_xy = to_xy
        else:
            new_xy = from_xy + (self.step_size / dist) * delta
        return (float(new_xy[0]), float(new_xy[1]))

    def _near_indices(self, tree, positions, new_xy):
        if len(tree) <= 1:
            return [0]
        kdtree = cKDTree(positions)
        idxs = kdtree.query_ball_point(new_xy, self.rewire_radius)
        return idxs if idxs else [int(kdtree.query(new_xy)[1])]

    def _choose_parent(self, mode, tree, near_idxs, new_xy):
        best_parent = None
        best_cost = float("inf")
        best_d2r = 0.0
        for idx in near_idxs:
            parent = tree[idx]
            if not self.is_segment_free(parent.x, new_xy):
                continue
            edge_cost, d2r = self.edge_cost(mode, parent.x, new_xy, parent.dist_to_right)
            total = parent.cost + edge_cost
            if total < best_cost:
                best_cost = total
                best_parent = idx
                best_d2r = d2r
        return best_parent, best_cost, best_d2r

    def _rewire(self, mode, tree, near_idxs, new_idx):
        new_node = tree[new_idx]
        for idx in near_idxs:
            if idx == new_idx:
                continue
            node = tree[idx]
            if not self.is_segment_free(new_node.x, node.x):
                continue
            edge_cost, d2r = self.edge_cost(mode, new_node.x, node.x, new_node.dist_to_right)
            new_cost = new_node.cost + edge_cost
            if new_cost + 1e-9 < node.cost:
                node.parent = new_idx
                node.cost = new_cost
                node.dist_to_right = d2r

    def _extract_path(self, tree, goal_idx):
        path = []
        idx = goal_idx
        while idx is not None:
            path.append(tree[idx].x)
            idx = tree[idx].parent
        path.reverse()
        return path

    def _shortcut_path(self, path):
        if path is None or len(path) < 3:
            return path
        shortened = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.is_segment_free(path[i], path[j]):
                    break
                j -= 1
            shortened.append(path[j])
            i = j
        return shortened

    def _connect_to_goal(self, mode, tree, positions):
        kdtree = cKDTree(positions)
        dist, nearest_idx = kdtree.query(self.x_goal)
        nearest_idx = int(nearest_idx)
        nearest = tree[nearest_idx]
        if float(dist) > self.goal_tolerance:
            return None
        if not self.is_segment_free(nearest.x, self.x_goal):
            return None
        edge_cost, d2r = self.edge_cost(mode, nearest.x, self.x_goal, nearest.dist_to_right)
        goal_node = _RRTNode(
            x=self.x_goal,
            parent=nearest_idx,
            cost=nearest.cost + edge_cost,
            dist_to_right=d2r,
        )
        tree.append(goal_node)
        return len(tree) - 1

    def solve(self, mode="modified", return_timing=False, rng=None):
        if mode not in self.SUPPORTED_SOLVER_MODES:
            raise ValueError(f"Unsupported RRT* mode '{mode}'. Expected one of {self.SUPPORTED_SOLVER_MODES}.")

        if not self.is_free(self.x_init) or not self.is_free(self.x_goal):
            self.path = None
            self.last_solve_time = 0.0
            self.last_tree = []
            self.last_solve_stats = {
                "success": False,
                "reason": "start_or_goal_occupied",
                "tree_nodes": 0,
                "min_goal_dist": float("inf"),
                "nearest_goal_node": None,
            }
            return (False, 0.0) if return_timing else False

        rng = np.random.default_rng() if rng is None else rng
        timeout = self.timeout_vanilla if mode == "vanilla" else self.timeout_modified
        t_start = time.time()

        tree = [_RRTNode(x=self.x_init, parent=None, cost=0.0, dist_to_right=0.0)]
        goal_idx = None
        iterations_run = 0
        stop_reason = "max_iterations"

        for _ in range(self.max_iterations):
            iterations_run += 1
            if time.time() - t_start > timeout:
                stop_reason = "timeout"
                break

            sample = self._sample_free(rng)
            positions = np.array([node.x for node in tree], dtype=float)
            nearest_idx = int(cKDTree(positions).query(sample)[1])
            new_xy = self._steer(tree[nearest_idx].x, sample)
            if not self.is_free(new_xy):
                continue
            if not self.is_segment_free(tree[nearest_idx].x, new_xy):
                continue

            near_idxs = self._near_indices(tree, positions, new_xy)
            parent_idx, best_cost, best_d2r = self._choose_parent(mode, tree, near_idxs, new_xy)
            if parent_idx is None:
                continue

            tree.append(
                _RRTNode(
                    x=new_xy,
                    parent=parent_idx,
                    cost=best_cost,
                    dist_to_right=best_d2r,
                )
            )
            new_idx = len(tree) - 1
            positions = np.vstack([positions, new_xy])
            self._rewire(mode, tree, near_idxs, new_idx)

            if self.distance(new_xy, self.x_goal) <= self.goal_tolerance:
                goal_idx = self._connect_to_goal(mode, tree, positions)
                if goal_idx is not None:
                    break

        if goal_idx is None:
            goal_idx = self._connect_to_goal(mode, tree, np.array([node.x for node in tree], dtype=float))

        elapsed = time.time() - t_start
        self.last_solve_time = elapsed
        self.last_tree = list(tree)

        positions = np.array([node.x for node in tree], dtype=float)
        if len(positions) == 0:
            min_goal_dist = float("inf")
            nearest_idx = None
        else:
            dists = np.linalg.norm(positions - np.array(self.x_goal, dtype=float), axis=1)
            nearest_idx = int(np.argmin(dists))
            min_goal_dist = float(dists[nearest_idx])

        if goal_idx is None:
            self.path = None
            self.last_solve_stats = {
                "success": False,
                "reason": stop_reason,
                "tree_nodes": len(tree),
                "iterations_run": iterations_run,
                "min_goal_dist": min_goal_dist,
                "goal_tolerance": float(self.goal_tolerance),
                "nearest_goal_node": None if nearest_idx is None else tuple(tree[nearest_idx].x),
                "nearest_segment_free": None,
            }
            if nearest_idx is not None:
                nearest_xy = tree[nearest_idx].x
                self.last_solve_stats["nearest_segment_free"] = bool(
                    self.is_segment_free(nearest_xy, self.x_goal)
                )
            return (False, elapsed) if return_timing else False

        self.path = self._shortcut_path(self._extract_path(tree, goal_idx))
        self.last_solve_stats = {
            "success": True,
            "reason": "ok",
            "tree_nodes": len(tree),
            "iterations_run": iterations_run,
            "min_goal_dist": min_goal_dist,
            "goal_tolerance": float(self.goal_tolerance),
            "nearest_goal_node": tuple(tree[nearest_idx].x) if nearest_idx is not None else None,
        }
        return (True, elapsed) if return_timing else True

    def iter_tree_edges(self):
        """Yield (parent_xy, child_xy) for each edge in ``last_tree``."""
        if not self.last_tree:
            return
        for idx, node in enumerate(self.last_tree):
            if node.parent is None:
                continue
            parent = self.last_tree[node.parent]
            yield parent.x, node.x


class RRTStar_With_Graph(RRTStar):
    def __init__(
        self,
        statespace_lo,
        statespace_hi,
        x_init,
        x_goal,
        occupancy: StochOccupancyGrid2D,
        graph: nx.DiGraph,
        resolution=1,
        robot_d=0.4,
        desired_dist_right_extra=0.25,
        **rrt_kwargs,
    ):
        super().__init__(
            statespace_lo,
            statespace_hi,
            x_init,
            x_goal,
            occupancy,
            resolution,
            robot_d,
            desired_dist_right_extra,
            **rrt_kwargs,
        )
        self.graph = graph

    def modified_edge_cost(self, x1, x2, dist2right_prev=0):
        dist = self.distance(x1, x2)
        x1x, x1y = self.get_index(x1)
        x2x, x2y = self.get_index(x2)
        if self.graph.has_edge((x1x, x1y), (x2x, x2y)):
            return float(self.graph[(x1x, x1y)][(x2x, x2y)]["weight"]), 0.0
        social_cost, dist2right = rightness_penalty(
            self.occupancy,
            self.x_init,
            self.x_goal,
            x1,
            x2,
            dist2right_prev,
            robot_d=self.robot_d,
            desired_dist_right_extra=self.desired_dist_right_extra,
            resolution=self.resolution,
        )
        return dist + social_cost + 0.01, dist2right
