"""Shared social edge-cost utilities for grid and sampling planners."""

from __future__ import annotations

import numpy as np

from social_path_planning.occupancy_grid import StochOccupancyGrid2D


def euclidean_distance(x1, x2):
    return float(np.linalg.norm(np.array(x1, dtype=float) - np.array(x2, dtype=float)))


def rightness_penalty(
    occupancy: StochOccupancyGrid2D,
    x_init,
    x_goal,
    x1,
    x2,
    dist2right_prev=0,
    *,
    robot_d=0.4,
    desired_dist_right_extra=0.25,
    resolution=1.0,
):
    """
    Social penalty for traversing edge x1 -> x2.

    Returns (penalty, dist_to_right) where dist_to_right is stored for chained
    penalties along a path.
    """
    if euclidean_distance(x2, x_goal) < 2:
        return 0, 0
    if euclidean_distance(x2, x_init) < 2:
        return 0, 0

    penalty = 0

    travel_dir = np.array(x2, dtype=float) - np.array(x1, dtype=float)
    travel_dir /= np.linalg.norm(travel_dir)
    dist_to_right = float(occupancy.dist_to_wall_right(x2, travel_dir))

    if dist_to_right > 10:
        penalty += resolution
        dist_to_left = float(occupancy.dist_to_wall_left(x2, travel_dir))

        if dist_to_left > 3:
            dist_to_left_prev = float(occupancy.dist_to_wall_left(x1, travel_dir))
            delta_dist_to_left = dist_to_left - dist_to_left_prev

            if delta_dist_to_left < 0 or delta_dist_to_left > 10:
                delta_dist_to_left = 0
            penalty += 5 * delta_dist_to_left
        else:
            penalty = max(0, (4 - dist_to_left))
    else:
        dist_to_right_prev = dist2right_prev
        delta_raw = dist_to_right - dist_to_right_prev

        desired_dist_right = robot_d / 2 + desired_dist_right_extra
        penalty = abs(dist_to_right - desired_dist_right)

        delta_far = delta_raw
        if delta_far > 15 or delta_far < 0:
            delta_far = 0
        if dist_to_right > desired_dist_right:
            penalty += 2 * delta_far
        elif dist_to_right < desired_dist_right and -15 < delta_raw < 0:
            penalty += 2 * (-delta_raw)
    return penalty, dist_to_right
