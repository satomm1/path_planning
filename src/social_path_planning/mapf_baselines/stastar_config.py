"""Configure space-time A* (8-connected, octile costs) for CBS / PP."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Dict, List, Set, Tuple

import numpy as np

# Octile costs (10 cardinal, 14 diagonal) on the stastar integer lattice.
OCTILE_CARDINAL_COST = 10
OCTILE_DIAGONAL_COST = 14

OCTILE_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

_STASTAR_CONFIGURED = False

# Octile A* needs more expansions than unit-cost STA* on the same grid.
STA_LOW_LEVEL_ITER_SCALE = 8


def octile_heuristic(start: np.ndarray, goal: np.ndarray) -> int:
    """Admissible octile distance for 10/14-connected grids."""
    dx = int(abs(int(goal[0]) - int(start[0])))
    dy = int(abs(int(goal[1]) - int(start[1])))
    d = OCTILE_CARDINAL_COST
    d2 = OCTILE_DIAGONAL_COST
    return d * (dx + dy) + (d2 - 2 * d) * min(dx, dy)


def octile_move_cost(current: np.ndarray, neighbour: np.ndarray) -> int:
    """Edge cost from current lattice position to neighbour (wait = one cardinal step)."""
    dx = int(abs(int(neighbour[0]) - int(current[0])))
    dy = int(abs(int(neighbour[1]) - int(current[1])))
    if dx == 0 and dy == 0:
        return OCTILE_CARDINAL_COST
    d = OCTILE_CARDINAL_COST
    d2 = OCTILE_DIAGONAL_COST
    return d * (dx + dy) + (d2 - 2 * d) * min(dx, dy)


def _octile_plan(
    self,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    dynamic_obstacles: Dict[int, Set[Tuple[int, int]]],
    semi_dynamic_obstacles: Dict[int, Set[Tuple[int, int]]] | None = None,
    max_iter: int = 500,
    debug: bool = False,
) -> np.ndarray:
    """Space-time A* with octile move costs (patch for stastar.Planner.plan)."""
    from stastar.state import State

    dynamic_obstacles = dict((k, np.array(list(v))) for k, v in dynamic_obstacles.items())

    def safe_dynamic(grid_pos: np.ndarray, time: int) -> bool:
        return all(
            self.l2(grid_pos, obstacle) > 2 * self.robot_radius
            for obstacle in dynamic_obstacles.setdefault(time, np.array([]))
        )

    if semi_dynamic_obstacles is None:
        semi_dynamic_obstacles = {}
    else:
        semi_dynamic_obstacles = dict(
            (k, np.array(list(v))) for k, v in semi_dynamic_obstacles.items()
        )

    def safe_semi_dynamic(grid_pos: np.ndarray, time: int) -> bool:
        for timestamp, obstacles in semi_dynamic_obstacles.items():
            if time >= timestamp:
                if not all(
                    self.l2(grid_pos, obstacle) > 2 * self.robot_radius
                    for obstacle in obstacles
                ):
                    return False
        return True

    start_pos = self.grid.snap_to_grid(np.array(start))
    goal_pos = self.grid.snap_to_grid(np.array(goal))

    def _st_key(pos: np.ndarray, time: int) -> Tuple[int, int, int]:
        return (int(pos[0]), int(pos[1]), int(time))

    start_h = octile_heuristic(start_pos, goal_pos)
    s = State(start_pos, 0, 0, start_h)
    open_set = [s]
    closed: Set[Tuple[int, int, int]] = set()
    came_from: dict = {}
    best_g: Dict[Tuple[int, int, int], int] = {_st_key(start_pos, 0): 0}
    iter_ = 0

    while open_set and iter_ < max_iter:
        iter_ += 1
        current_state = heappop(open_set)
        cur_key = _st_key(current_state.pos, current_state.time)
        if current_state.g_score > best_g.get(cur_key, current_state.g_score):
            continue
        if cur_key in closed:
            continue
        closed.add(cur_key)

        if current_state.pos_equal_to(goal_pos):
            if debug:
                print(f"STA* (octile): Path found after {iter_} iterations")
            return self.reconstruct_path(came_from, current_state)

        epoch = current_state.time + 1
        neighbours = self.neighbour_table.lookup(current_state.pos)

        def _feasible_at_epoch(grid_pos: np.ndarray) -> bool:
            return (
                self.safe_static(grid_pos)
                and safe_dynamic(grid_pos, epoch)
                and safe_semi_dynamic(grid_pos, epoch)
            )

        def _has_feasible_spatial_move() -> bool:
            for candidate in neighbours:
                if np.array_equal(candidate, current_state.pos):
                    continue
                if _feasible_at_epoch(candidate):
                    return True
            return False

        allow_spatial = _has_feasible_spatial_move()

        for neighbour in neighbours:
            if np.array_equal(neighbour, current_state.pos):
                if allow_spatial:
                    continue
            elif not _feasible_at_epoch(neighbour):
                continue

            step = octile_move_cost(current_state.pos, neighbour)
            g = current_state.g_score + step
            n_key = _st_key(neighbour, epoch)
            if g >= best_g.get(n_key, 1 << 60):
                continue
            best_g[n_key] = g
            if n_key in closed:
                closed.discard(n_key)
            h = octile_heuristic(neighbour, goal_pos)
            neighbour_state = State(neighbour, epoch, g, h)
            came_from[neighbour_state] = current_state
            heappush(open_set, neighbour_state)

    if debug:
        print("STA* (octile): Open set is empty, no path found.")
    return np.array([])


def apply_stastar_connectivity() -> None:
    """Apply 8-connected neighbours and octile 10/14 costs to stastar (idempotent)."""
    global _STASTAR_CONFIGURED
    from stastar.neighbour_table import NeighbourTable
    from stastar.planner import Planner

    NeighbourTable.directions = list(OCTILE_DIRECTIONS)
    Planner.h = staticmethod(octile_heuristic)
    Planner.plan = _octile_plan
    _STASTAR_CONFIGURED = True
