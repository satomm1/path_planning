"""Plots and animations for multi-robot trajectory coordination.

Moved out of ``multi_planning`` and ``event_multi_planning`` so the solver modules
do not import matplotlib. The old names remain importable from those modules.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from social_path_planning.event_multi_planning import (
    _encounter_kind_counts,
    build_constant_velocity_schedules,
    point_at_arc_length,
)
from social_path_planning.multi_planning import (
    VIZ_MILP_Z_OUTPUT_DIR,
    _group_collision_pairs_by_agent_pair,
    path_to_arc_length,
)

VIZ_EVENT_OUTPUT_DIR = "results/event_planning_viz"

_AGENT_COLORS = ["steelblue", "darkorange", "seagreen", "purple", "brown", "teal"]
_WAYPOINT_MARKERS = {
    "start": ("o", 10),
    "conflict_enter": ("s", 9),
    "conflict_exit": ("^", 9),
    "goal": ("*", 12),
}


# ---------------------------------------------------------------------------
# Dense MILP diagnostics
# ---------------------------------------------------------------------------

def viz_milp_z_assignments(paths, collision_pairs, occ_grid=None, output_dir=VIZ_MILP_Z_OUTPUT_DIR):
    """Save one PNG per agent pair showing conflict chords colored by z index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_collision_pairs_by_agent_pair(collision_pairs)

    for (a1, a2), constraints in sorted(grouped.items()):
        if not constraints:
            continue
        path_a, path_b = paths[a1], paths[a2]
        unique_z = sorted({z for _, _, z, _ in constraints})
        cmap = plt.colormaps["tab10"].resampled(max(1, len(unique_z)))
        z_colors = {z: cmap(idx % 10) for idx, z in enumerate(unique_z)}

        fig, ax = plt.subplots(figsize=(10, 8))
        if occ_grid is not None:
            occ_grid.plot_grid(ax=ax)
        ax.plot(*zip(*path_a), color="steelblue", linewidth=2.5, label=f"agent {a1}")
        ax.plot(*zip(*path_b), color="darkorange", linewidth=2.5, label=f"agent {a2}")

        opp_partial = defaultdict(dict)
        for i, j, z, mode in constraints:
            if mode == "a_first":
                opp_partial[z].update(i_end=i, j_start=j)
            elif mode == "b_first":
                opp_partial[z].update(i_start=i, j_end=j)
        for z, row in opp_partial.items():
            if len(row) != 4:
                continue
            i0, i1, j0, j1 = row["i_start"], row["i_end"], row["j_start"], row["j_end"]
            for seg in range(i0, i1 + 1):
                ax.plot(*zip(path_a[seg], path_a[seg + 1]), color=z_colors[z], linewidth=5, alpha=0.35)
            for seg in range(j0, j1 + 1):
                ax.plot(*zip(path_b[seg], path_b[seg + 1]), color=z_colors[z], linewidth=5, alpha=0.35)

        drawn = set()
        for i, j, z, mode in constraints:
            ma = (np.array(path_a[i]) + np.array(path_a[i + 1])) / 2.0
            mb = (np.array(path_b[j]) + np.array(path_b[j + 1])) / 2.0
            ax.plot(
                [ma[0], mb[0]],
                [ma[1], mb[1]],
                color=z_colors[z],
                linestyle="--" if mode in ("a_first", "b_first") else "-",
                linewidth=1.8,
                label=f"z{z}" if z not in drawn else None,
            )
            drawn.add(z)

        ax.set_title(
            f"MILP z agents ({a1},{a2}): {len(unique_z)} z, {len(constraints)} constraints"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=8)
        out = output_dir / f"z_pair_{a1}_{a2}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved MILP z visualization: {out.resolve()}")


# ---------------------------------------------------------------------------
# Schedules: space-time plots, map context, videos
# ---------------------------------------------------------------------------

def get_position_at_time(t, path, time_points):
    """
    Returns (x, y) at time t using linear interpolation.
    If t is outside the range of time_points, it returns the start or end pos.
    """
    # Extract x and y lists
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    # Check if the path hasn't started or has finished
    # (Optional: return None if you want points to disappear)
    if t < time_points[0]:
        return xs[0], ys[0]
    if t > time_points[-1]:
        return xs[-1], ys[-1]

    # Interpolate
    x = np.interp(t, time_points, xs)
    y = np.interp(t, time_points, ys)
    return x, y


def create_space_time_plot(paths, times, output_file="space_time_sequential.png", title="Space-Time Plot", dpi=300):
    """
    Create and save a static space-time plot for multiple agents.
    x-axis: time [s], y-axis: distance along path [m].
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.brg(np.linspace(0, 0.9, len(paths)))
    title_fs = 22
    label_fs = 20
    tick_fs = 20
    legend_fs = 20

    for i, (path, time_seq, color) in enumerate(zip(paths, times, colors)):
        s_values = path_to_arc_length(path)
        if len(s_values) != len(time_seq):
            raise ValueError(
                f"Length mismatch for path {i + 1}: "
                f"{len(s_values)} arc-length points vs {len(time_seq)} time points."
            )
        ax.plot(time_seq, s_values, color=color, linewidth=4, label=f"Robot {i + 1}")

    ax.set_xlabel("Time [s]", fontsize=label_fs)
    ax.set_ylabel("Distance Along Path [m]", fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=legend_fs)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    print(f"Saved space-time plot to {output_file}")


def _draw_map_context(ax, paths, occ_grid=None, title="Path Context Map"):
    """
    Draw the static map-context layers (grid, paths, start/goal markers, legend).
    Returns the per-path colors so callers can overlay matching robot markers.
    """
    title_fs = 10
    legend_fs = 8

    if occ_grid is not None:
        occ_grid.plot_grid(ax=ax)

    colors = plt.cm.brg(np.linspace(0, 0.7, len(paths)))
    path_handles = []
    path_labels = []
    for i, (path, color) in enumerate(zip(paths, colors)):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        path_line, = ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, label=f"Robot {i + 1}")
        path_handles.append(path_line)
        path_labels.append(f"Robot {i + 1}")

        # Start/goal markers for context in the paper figure.
        ax.scatter(xs[0], ys[0], marker="s", s=50, color=color, edgecolors="black",
                   linewidths=1.4, zorder=9)
        ax.scatter(xs[-1], ys[-1], marker="*", s=80, color=color, edgecolors="black",
                   linewidths=1.4, zorder=9)

    ax.set_title(title, fontsize=title_fs)
    # Turn ticks off
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    path_legend = ax.legend(path_handles, path_labels, loc="upper right", fontsize=legend_fs)
    marker_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
               markeredgecolor="black", markersize=4, label="Robot Position"),
    ]
    marker_legend = ax.legend(handles=marker_handles, loc="upper right", fontsize=legend_fs)
    path_legend.get_title().set_fontsize(legend_fs)
    marker_legend.get_title().set_fontsize(legend_fs)
    ax.add_artist(marker_legend)
    return colors


def create_map_context_plot(
        paths,
        occ_grid=None,
        times=None,
        snapshot_time=None,
        output_file="map_context.png",
        title="Path Context Map",
        dpi=300):
    """
    Create and save a static map plot with robot paths.
    Optionally overlays robot positions at snapshot_time.
    """
    fig, ax = plt.subplots(figsize=(3, 3.1))
    colors = _draw_map_context(ax, paths, occ_grid=occ_grid, title=title)

    if times is not None and snapshot_time is not None:
        for path, time_seq, color in zip(paths, times, colors):
            rx, ry = get_position_at_time(snapshot_time, path, time_seq)
            ax.scatter(rx, ry, marker="o", s=25, color=color, edgecolors="black", linewidths=1.4,
                       zorder=10)

    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    print(f"Saved map context plot to {output_file}")


def create_map_context_video(
        paths,
        times,
        occ_grid=None,
        output_file="map_context.gif",
        title="Path Context Map",
        total_frames=200,
        fps=10,
        dpi=200,
        show_time=True):
    """
    Animated version of create_map_context_plot: same layout and coloring, with
    robot position markers moving along their paths over time.
    Saves .gif via Pillow, or .mp4 if ffmpeg is installed.
    """
    fig, ax = plt.subplots(figsize=(3, 3.1))
    colors = _draw_map_context(ax, paths, occ_grid=occ_grid, title=title)

    robot_markers = []
    for path, time_seq, color in zip(paths, times, colors):
        rx, ry = get_position_at_time(time_seq[0], path, time_seq)
        marker = ax.scatter([rx], [ry], marker="o", s=25, color=color, edgecolors="black",
                            linewidths=1.4, zorder=10)
        robot_markers.append(marker)

    time_text = None
    if show_time:
        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top",
                            fontsize=8, zorder=11,
                            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
    fig.tight_layout()

    all_times = [t for time_seq in times for t in time_seq]
    frames = np.linspace(min(all_times), max(all_times), total_frames)

    def update(current_time):
        for marker, path, time_seq in zip(robot_markers, paths, times):
            marker.set_offsets([get_position_at_time(current_time, path, time_seq)])
        artists = list(robot_markers)
        if time_text is not None:
            time_text.set_text(f"t = {current_time:.1f} s")
            artists.append(time_text)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=1000 / fps)
    writer = "ffmpeg" if str(output_file).lower().endswith(".mp4") else "pillow"
    print(f"Saving map context animation to {output_file}...")
    ani.save(output_file, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Saved map context animation to {output_file}")


def create_video(paths, times, output_file="video.gif", occ_grid=None):
    """
    Create a video visualizing the multi-agent paths over time.
    """
    # --- 1. Determine global time bounds ---
    all_times = [t for sublist in times for t in sublist]
    start_time = min(all_times)
    end_time = max(all_times)

    # --- 2. Setup Figure ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Path Visualization")
    ax.grid(True, linestyle='--', alpha=0.6)
    if occ_grid is not None:
        occ_grid.plot_grid(ax=ax)

    # Determine axis limits automatically based on all coordinates
    all_coords = [p for sublist in paths for p in sublist]
    all_xs = [p[0] for p in all_coords]
    all_ys = [p[1] for p in all_coords]
    pad = 1

    # --- 3. Initialize lines (trails) and points ---
    colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
    lines = []
    points = []

    for i, color in enumerate(colors):
        # The trail line
        line, = ax.plot([], [], color=color, alpha=0.5, linewidth=1)
        lines.append(line)

        # The current head point
        point, = ax.plot([], [], marker='o', color=color, markersize=8, label=f'Path {i + 1}')
        points.append(point)

    ax.legend(loc='upper right')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # --- 4. Define Animation Update Functions ---
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]

    def update(frame):
        current_time = frame

        for i, (path, time_seq) in enumerate(zip(paths, times)):
            # Get current head position
            curr_x, curr_y = get_position_at_time(current_time, path, time_seq)
            points[i].set_data([curr_x], [curr_y])

            # Draw trail (history up to current time)
            history_x = []
            history_y = []

            # Add past fixed waypoints
            for (px, py), pt in zip(path, time_seq):
                if pt <= current_time:
                    history_x.append(px)
                    history_y.append(py)

            # Add current interpolated position to connect the line smoothly
            history_x.append(curr_x)
            history_y.append(curr_y)

            lines[i].set_data(history_x, history_y)

        time_text.set_text(f'Time: {current_time:.2f}')
        return lines + points + [time_text]

    # --- 5. Run Animation ---
    total_frames = 200
    interval_ms = 100
    frames = np.linspace(start_time, end_time, total_frames)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=interval_ms
    )

    if output_file:
        print(f"Saving animation to {output_file}...")
        # Note: Saving as .mp4 requires ffmpeg to be installed.
        # Saving as .gif requires Pillow.
        try:
            ani.save(output_file, fps=1000 / interval_ms)
            print("Save complete.")
        except Exception as e:
            print(f"Could not save file: {e}")
            print("Ensure ffmpeg is installed for video, or try saving as .gif")


# ---------------------------------------------------------------------------
# Event MILP: interest waypoints and encounter panels
# ---------------------------------------------------------------------------

def _plot_conflict_segments(ax, path, intervals, color="crimson"):
    for interval in intervals:
        for seg in range(interval.seg_start, interval.seg_end + 1):
            x0, y0 = path[seg]
            x1, y1 = path[seg + 1]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=color,
                linewidth=5.0,
                alpha=0.9,
                solid_capstyle="round",
                zorder=8,
            )


def _plot_interest_waypoint(ax, waypoint, color):
    marker, size = _WAYPOINT_MARKERS.get(waypoint.kind, ("o", 8))
    x, y = waypoint.point
    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=size**2,
        color=color,
        edgecolors="black",
        linewidths=0.6,
        zorder=10,
        label=waypoint.kind if waypoint.kind not in getattr(ax, "_event_labels", set()) else None,
    )
    labels = getattr(ax, "_event_labels", set())
    labels.add(waypoint.kind)
    ax._event_labels = labels


def viz_event_waypoints(
    analysis,
    *,
    occ_grid=None,
    output_path=None,
    also_write_pair_panels=True,
):
    """Save a multi-agent figure highlighting conflict segments and interest waypoints."""
    output_path = Path(output_path or (VIZ_EVENT_OUTPUT_DIR + "/waypoints.png"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.4, 2.72))
    if occ_grid is not None:
        occ_grid.plot_grid(ax=ax)
    ax._event_labels = set()

    for agent_path in analysis.agents:
        color = _AGENT_COLORS[agent_path.agent % len(_AGENT_COLORS)]
        path = agent_path.original_path
        if len(path) >= 2:
            ax.plot(
                [p[0] for p in path],
                [p[1] for p in path],
                color=color,
                linewidth=2.0,
                alpha=0.55,
                label=f"agent {agent_path.agent}",
                zorder=5,
            )
        _plot_conflict_segments(ax, path, agent_path.conflict_intervals)
        for waypoint in agent_path.interest_waypoints:
            _plot_interest_waypoint(ax, waypoint, color)

    opposite_count, same_count, intersect_count = _encounter_kind_counts(analysis.encounters)
    interest_total = sum(len(agent.interest_waypoints) for agent in analysis.agents)
    ax.set_title(
        "Event interest waypoints: "
        f"{len(analysis.agents)} agents, {interest_total} waypoints, "
        f"{len(analysis.encounters)} encounters "
        f"({opposite_count} opposite, {same_count} same, {intersect_count} intersect)"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved event waypoint visualization: {output_path.resolve()}")

    if also_write_pair_panels and analysis.encounters:
        for encounter in analysis.encounters:
            pair_path = output_path.parent / f"encounter_{encounter.agent_a}_{encounter.agent_b}.png"
            _save_encounter_panel(analysis, encounter, occ_grid=occ_grid, output_path=pair_path)

    return output_path


def _save_encounter_panel(analysis, encounter, *, occ_grid=None, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    if occ_grid is not None:
        occ_grid.plot_grid(ax=ax)

    for agent_idx in (encounter.agent_a, encounter.agent_b):
        agent_path = analysis.agents[agent_idx]
        color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
        path = agent_path.original_path
        ax.plot(
            [p[0] for p in path],
            [p[1] for p in path],
            color=color,
            linewidth=2.0,
            alpha=0.55,
            label=f"agent {agent_idx}",
            zorder=5,
        )
        interval = (
            encounter.interval_a if agent_idx == encounter.agent_a else encounter.interval_b
        )
        enter = point_at_arc_length(path, interval.s_enter)
        exit_pt = point_at_arc_length(path, interval.s_exit)
        ax.plot(
            [enter[0], exit_pt[0]],
            [enter[1], exit_pt[1]],
            color=color,
            linewidth=7.0,
            alpha=0.25,
            solid_capstyle="round",
            zorder=6,
        )
        _plot_conflict_segments(ax, path, [interval], color="crimson")
        for waypoint in agent_path.interest_waypoints:
            _plot_interest_waypoint(ax, waypoint, color)

    ax.set_title(
        f"Encounter ({encounter.agent_a},{encounter.agent_b}) kind={encounter.kind}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved encounter visualization: {Path(output_path).resolve()}")


def create_event_video(
    analysis,
    event_times_list,
    *,
    output_file="video_event.gif",
    occ_grid=None,
    title="Path Context Map",
):
    """Render a multi-agent animation from event MILP schedules (map-context styling)."""
    paths, times = build_constant_velocity_schedules(analysis, event_times_list)
    create_map_context_video(paths, times, occ_grid=occ_grid, output_file=output_file, title=title)
