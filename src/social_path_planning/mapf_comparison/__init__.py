"""Cross-method MAPF vs MILP comparison: motion, schedules, unified metrics, pipeline.

Names are resolved lazily: importing a light submodule such as
``mapf_comparison.grid_traversability`` does not load the solver pipeline,
CBS/PP baselines or the ensemble benchmark.
"""

import importlib

_EXPORTS = {
    "DEFAULT_COARSE_BLOCK_POLICY": "grid_traversability",
    "ROBOT_DIAMETER_M": "grid_traversability",
    "TraversabilityModel": "grid_traversability",
    "DEFAULT_MAX_VELOCITY_MPS": "motion",
    "MotionConfig": "motion",
    "cell_size_m": "motion",
    "MapfRunConfig": "pipeline",
    "prepare_mapf_problem": "pipeline",
    "run_event_milp_solver": "pipeline",
    "run_mapf_solvers": "pipeline",
    "run_milp_solver": "pipeline",
    "schedule_mapf_paths": "schedule",
    "compute_solution_metrics": "solution_metrics",
    "dedupe_world_path": "solution_metrics",
    "aggregate_mapf_metrics": "metrics",
    "aggregate_milp_metrics": "metrics",
    "mapf_path_timesteps": "metrics",
    "path_bank_soc_meters": "metrics",
    "aggregate_ensemble_metrics": "ensemble",
    "ensure_modified_path_pool": "ensemble",
    "run_ensemble_trial": "ensemble",
    "sample_route_subset": "ensemble",
}

__all__ = [
    "DEFAULT_COARSE_BLOCK_POLICY",
    "DEFAULT_MAX_VELOCITY_MPS",
    "ROBOT_DIAMETER_M",
    "TraversabilityModel",
    "MotionConfig",
    "MapfRunConfig",
    "cell_size_m",
    "schedule_mapf_paths",
    "prepare_mapf_problem",
    "run_mapf_solvers",
    "run_milp_solver",
    "run_event_milp_solver",
    "compute_solution_metrics",
    "dedupe_world_path",
    "aggregate_mapf_metrics",
    "aggregate_milp_metrics",
    "mapf_path_timesteps",
    "path_bank_soc_meters",
    "aggregate_ensemble_metrics",
    "ensure_modified_path_pool",
    "run_ensemble_trial",
    "sample_route_subset",
]


def __getattr__(name):
    submodule = _EXPORTS.get(name)
    if submodule is not None:
        value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
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
