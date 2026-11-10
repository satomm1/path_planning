"""Social multi-robot global path planning.

Top-level names are resolved lazily so that importing any submodule (for example
``social_path_planning.common``) does not load every planner, matplotlib and
networkx up front. ``from social_path_planning import AStar`` still works.
"""

import importlib

_EXPORTS = {
    "AStar": "social_path_planning.a_star",
    "AStar_With_Graph": "social_path_planning.a_star",
    "RRTStar": "social_path_planning.rrt_star",
    "RRTStar_With_Graph": "social_path_planning.rrt_star",
    "WallDistanceCacheError": "social_path_planning.wall_distance_cache",
    "diagnose_wall_distance_cache": "social_path_planning.wall_distance_cache",
    "load_wall_distance_cache_into_grid": "social_path_planning.wall_distance_cache",
    "attach_wall_distance_cache": "social_path_planning.wall_distance_cache",
    "DEFAULT_DIST_THRESH": "social_path_planning.wall_distance_cache",
    "travel_dir_to_dir_idx": "social_path_planning.wall_distance_cache",
    "FrequentSubgraph": "social_path_planning.sparse_graph",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value
        return value
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{name}":
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
