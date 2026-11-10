"""Demo: event-waypoint MICP coordination on grid2 (``sample2_default``).

Formerly the ``__main__`` block of ``event_multi_planning``. Runs the simultaneous,
sequential and combined event planners on four cached/planned paths and writes
figures and GIFs to ``results/event_planning_viz``.
"""

from __future__ import annotations

from pathlib import Path

from social_path_planning.event_multi_planning import (
    MAX_VELOCITY,
    EventMultiAgentCombinedPlanner,
    EventMultiAgentSequentialPlanner,
    EventMultiAgentSimultaneousPlanner,
    _interest_segment_lengths,
    build_constant_velocity_schedules,
)
from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.trajectory_viz import (
    create_event_video,
    create_map_context_plot,
    create_space_time_plot,
    viz_event_waypoints,
)


def _resolve_fixed_event_times(agent_path, reference_times, velocity):
    """
    Build a fixed interest-waypoint schedule for sequential/combined demos.

    Reuse ``reference_times`` when lengths already match; otherwise fall back to
    minimum-time motion at ``velocity`` along interest-waypoint segments.
    """
    expected = len(agent_path.interest_waypoints)
    if reference_times is not None and len(reference_times) == expected:
        return [float(t) for t in reference_times]
    velocity = max(float(velocity), 1e-6)
    times = [0.0]
    for seg_len in _interest_segment_lengths(agent_path):
        times.append(times[-1] + float(seg_len) / velocity)
    return times


_DEFAULT_GRID2_HEATMAP = "../../outputs_temp_obstacles/baseline_permanent_heatmap.npy"
_social_graph_cache: dict[str, object] = {}


def _social_graph_from_heatmap(occ_grid, heatmap_file):
    """Build (or return cached) sparse social graph from a directional heatmap file."""
    key = str(Path(heatmap_file).resolve())
    if key not in _social_graph_cache:
        from social_path_planning.compare_astar import build_sparse_graph_from_heatmap

        _social_graph_cache[key] = build_sparse_graph_from_heatmap(
            occ_grid,
            heatmap_path=key,
        )
    return _social_graph_cache[key]


def _load_or_plan_path(
    occ_grid,
    map_size,
    map_resolution,
    pickle_name,
    x_init,
    x_goal,
    *,
    social_graph=None,
    heatmap_file=_DEFAULT_GRID2_HEATMAP,
):
    """Load a cached A* path or plan and pickle it (grid2 demo helper)."""
    import pickle

    from social_path_planning.a_star import AStar, AStar_With_Graph
    from social_path_planning.utils import snap_to_grid

    if social_graph is None and heatmap_file is not None:
        social_graph = _social_graph_from_heatmap(occ_grid, heatmap_file)

    try:
        with open(pickle_name, "rb") as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        pass

    statespace_hi = snap_to_grid(map_size, map_resolution)
    x_init_snapped = snap_to_grid(x_init, map_resolution)
    x_goal_snapped = snap_to_grid(x_goal, map_resolution)
    if social_graph is not None:
        problem = AStar_With_Graph(
            [0, 0],
            statespace_hi,
            x_init_snapped,
            x_goal_snapped,
            occ_grid,
            social_graph,
            resolution=map_resolution,
        )
    else:
        problem = AStar(
            [0, 0],
            statespace_hi,
            x_init_snapped,
            x_goal_snapped,
            occ_grid,
            resolution=map_resolution,
        )
    if not problem.solve():
        raise RuntimeError(f"A* failed for {pickle_name}")
    path = problem.path
    with open(pickle_name, "wb") as handle:
        pickle.dump(path, handle)
    return path


def main():
    scenario_name = "sample2_default"  # grid2 / four-quadrant benchmark map
    output_dir = Path("results/event_planning_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    occ_grid = StochOccupancyGrid2D(
        map_resolution,
        round(map_size[0] / map_resolution),
        round(map_size[1] / map_resolution),
        0,
        0,
        10,
        occ.T,
    )

    path1 = _load_or_plan_path(
        occ_grid, map_size, map_resolution, "path1.pkl", [2, 25], [97, 40]
    )
    path2 = _load_or_plan_path(
        occ_grid, map_size, map_resolution, "path2.pkl", [2, 40], [97, 50]
    )
    path3 = _load_or_plan_path(
        occ_grid, map_size, map_resolution, "path3.pkl", [85, 50], [50, 80]
    )
    path4 = _load_or_plan_path(
        occ_grid, map_size, map_resolution, "path4.pkl", [50, 20], [50, 65]
    )

    stride = 1
    stage_paths = [path1[::stride], path2[::stride], path3[::stride], path4[::stride]]
    velocities = [
        MAX_VELOCITY,
        MAX_VELOCITY / 1.2,
        MAX_VELOCITY / 3.2,
        MAX_VELOCITY / 3.0,
    ]

    ############## Simultaneous event MILP (all four agents) ##############
    sim_planner = EventMultiAgentSimultaneousPlanner(occ_grid, paths=stage_paths, norm=1)
    sim_planner.assign_velocities(velocities)

    print("Running simultaneous event MILP on grid2 (sample2_default)...")
    try:
        sim_event_times = sim_planner.plan(verbose=True, print_constraints=True)
        for agent_idx, event_time in enumerate(sim_event_times):
            print(f"simultaneous agent {agent_idx}: {event_time}")
    except Exception as exc:
        print(f"Simultaneous event MILP failed: {type(exc).__name__}: {exc}")
        raise

    sim_analysis = sim_planner.analysis
    sim_anim_paths, sim_anim_times = build_constant_velocity_schedules(sim_analysis, sim_event_times)

    viz_event_waypoints(
        sim_analysis,
        occ_grid=occ_grid,
        output_path=output_dir / "grid2_event_waypoints_simultaneous.png",
        also_write_pair_panels=True,
    )
    create_space_time_plot(
        sim_anim_paths,
        sim_anim_times,
        output_file=str(output_dir / "grid2_space_time_event_simultaneous.png"),
        title="Simultaneous Trajectory Planning",
    )
    snapshot_time = 120 # 0.35 * max(t_seq[-1] for t_seq in sim_anim_times)
    create_map_context_plot(
        sim_anim_paths,
        occ_grid=occ_grid,
        times=sim_anim_times,
        snapshot_time=snapshot_time,
        output_file=str(output_dir / "grid2_map_context_event_simultaneous.svg"),
        title="Simultaneous Planning",
        dpi=600
    )
    create_event_video(
        sim_analysis,
        sim_event_times,
        output_file=str(output_dir / "grid2_video_event_simultaneous.gif"),
        occ_grid=occ_grid,
        title="Simultaneous Planning",
    )
    print(
        "Simultaneous event MILP demo complete: "
        f"{sim_planner.num_z} binaries, {sim_planner.mutex_constraint_count} mutex constraints"
    )

    ############## Sequential event MILP (path1 ego; paths 2-4 fixed) ##############
    seq_planner = EventMultiAgentSequentialPlanner(
        occ_grid,
        other_agent_paths=[stage_paths[1], stage_paths[2], stage_paths[3]],
        other_agent_times=[[0.0], [0.0], [0.0]],
        path=stage_paths[0],
        v=velocities[0],
    )
    seq_analysis = seq_planner.analyze()
    seq_planner.other_agent_times = [
        _resolve_fixed_event_times(
            seq_analysis.agents[other_idx],
            sim_event_times[other_idx],
            velocities[other_idx],
        )
        for other_idx in (1, 2, 3)
    ]

    print("Running sequential event MILP (optimize path1; paths 2-4 fixed)...")
    try:
        ego_event_times = seq_planner.plan(verbose=True, print_constraints=True)
        print(f"sequential ego times: {ego_event_times}")
    except Exception as exc:
        print(f"Sequential event MILP failed: {type(exc).__name__}: {exc}")
        raise

    seq_event_times_list = [ego_event_times] + seq_planner.other_agent_times
    seq_anim_paths, seq_anim_times = build_constant_velocity_schedules(
        seq_analysis,
        seq_event_times_list,
    )
    create_space_time_plot(
        seq_anim_paths,
        seq_anim_times,
        output_file=str(output_dir / "grid2_space_time_event_sequential.png"),
        title="Event MILP Space-Time Plot (sequential: path1 active)",
    )
    create_map_context_plot(
        seq_anim_paths,
        occ_grid=occ_grid,
        times=seq_anim_times,
        snapshot_time=0.35 * max(t_seq[-1] for t_seq in seq_anim_times),
        output_file=str(output_dir / "grid2_map_context_event_sequential.png"),
        title="Event MILP Paths on Grid2 (sequential: path1 active)",
    )
    create_event_video(
        seq_analysis,
        seq_event_times_list,
        output_file=str(output_dir / "grid2_video_event_sequential.gif"),
        occ_grid=occ_grid,
        title="Event MILP Paths on Grid2 (sequential: path1 active)",
    )
    print(
        "Sequential event MILP demo complete: "
        f"{seq_planner.num_z} binaries, {seq_planner.mutex_constraint_count} mutex constraints"
    )

    ############## Combined event MILP (paths 1-2 fixed; paths 3 & 4 active) ##############
    combined_planner = EventMultiAgentCombinedPlanner(
        occ_grid,
        other_agent_paths=[stage_paths[0], stage_paths[1]],
        other_agent_times=[[0.0], [0.0]],
        paths=[stage_paths[2], stage_paths[3]],
        norm=1,
        v=[velocities[2], velocities[3]],
    )
    combined_analysis = combined_planner.analyze()
    combined_planner.other_agent_times = [
        _resolve_fixed_event_times(
            combined_analysis.agents[0],
            sim_event_times[0],
            velocities[0],
        ),
        _resolve_fixed_event_times(
            combined_analysis.agents[1],
            sim_event_times[1],
            velocities[1],
        ),
    ]

    print("Running combined event MILP (optimize paths 3 & 4; paths 1-2 fixed)...")
    try:
        combined_event_times = combined_planner.plan(verbose=True, print_constraints=True)
        print(f"combined active times (path1, path4): {combined_event_times}")
    except Exception as exc:
        print(f"Combined event MILP failed: {type(exc).__name__}: {exc}")
        raise

    combined_event_times_list = (
        combined_planner.other_agent_times + combined_event_times
    )
    combined_anim_paths, combined_anim_times = build_constant_velocity_schedules(
        combined_analysis,
        combined_event_times_list,
    )
    create_space_time_plot(
        combined_anim_paths,
        combined_anim_times,
        output_file=str(output_dir / "grid2_space_time_event_combined.png"),
        title="Event MILP Space-Time Plot (combined: paths 1 & 4 active)",
    )
    create_map_context_plot(
        combined_anim_paths,
        occ_grid=occ_grid,
        times=combined_anim_times,
        snapshot_time=0.35 * max(t_seq[-1] for t_seq in combined_anim_times),
        output_file=str(output_dir / "grid2_map_context_event_combined.png"),
        title="Event MILP Paths on Grid2 (combined: paths 1 & 4 active)",
    )
    create_event_video(
        combined_analysis,
        combined_event_times_list,
        output_file=str(output_dir / "grid2_video_event_combined.gif"),
        occ_grid=occ_grid,
        title="Event MILP Paths on Grid2 (combined: paths 1 & 4 active)",
    )
    print(
        "Combined event MILP demo complete: "
        f"{combined_planner.num_z} binaries, {combined_planner.mutex_constraint_count} mutex constraints"
    )

    print(f"All event MILP demos complete; outputs in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
