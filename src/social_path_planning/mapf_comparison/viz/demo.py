"""Demo scenario for quick MAPF checks."""
from __future__ import annotations
import numpy as np
from social_path_planning.occupancy_grid import StochOccupancyGrid2D

def build_demo_open_scenario(num_agents: int = 4):
    """
    Small open grid where CBS/PP always succeed — for quick implementation checks.
    """
    import numpy as np
    from social_path_planning.occupancy_grid import StochOccupancyGrid2D

    width, height = 16, 12
    resolution = 1.0
    occ = StochOccupancyGrid2D(
        resolution, width, height, 0.0, 0.0, 10, np.zeros((height, width)), thresh=0.5
    )
    routes = []
    # At least 2*robot_radius + 1 cells apart (robot_radius_cells == 1 on this grid).
    y_slots = np.linspace(2, height - 3, num_agents)
    if num_agents > 1:
        min_sep = 4.0
        y_slots = [2.0 + i * min_sep for i in range(num_agents)]
    for i in range(num_agents):
        y = float(y_slots[i])
        routes.append({
            "route_id": i,
            "x_init": [1.0, y],
            "x_goal": [float(width - 2), y],
            "path": [(1.0, y), (float(width - 2), y)],
        })
    return occ, routes


