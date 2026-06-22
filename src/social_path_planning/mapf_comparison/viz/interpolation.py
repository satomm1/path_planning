"""Time interpolation along scheduled polylines."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def get_position_at_time(
    t: float,
    path: Sequence[Tuple[float, float]],
    time_points: Sequence[float],
) -> Tuple[float, float]:
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    if t < time_points[0]:
        return xs[0], ys[0]
    if t > time_points[-1]:
        return xs[-1], ys[-1]
    x = float(np.interp(t, time_points, xs))
    y = float(np.interp(t, time_points, ys))
    return x, y
