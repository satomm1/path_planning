"""MAPF schedule animations."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from social_path_planning.mapf_adapter import MapfPath
from social_path_planning.mapf_comparison.viz.conflicts import (
    DEFAULT_CONFLICT_THRESHOLD_M,
    pairwise_conflict_at_time,
    pairwise_conflict_at_time_world,
)
from social_path_planning.mapf_comparison.viz.interpolation import get_position_at_time

def create_mapf_animation(
    occ_grid,
    world_paths: Sequence[Sequence[Tuple[float, float]]],
    time_lists: Sequence[Sequence[float]],
    output_file: Path,
    title: str = "MAPF",
    robot_radius_m: Optional[float] = None,
    grid_paths_for_conflict: Optional[Sequence[MapfPath]] = None,
    robot_radius_cells: int = 1,
    conflict_threshold_m: Optional[float] = None,
    num_frames: int = 200,
    interval_ms: int = 50,
    dpi: int = 120,
    show: bool = False,
) -> None:
    """GIF/MP4 of agents moving along MAPF schedules; flags timestep conflicts in red."""
    if robot_radius_m is None:
        robot_radius_m = 0.5 * occ_grid.resolution * max(robot_radius_cells, 1)

    all_times = [t for seq in time_lists for t in seq]
    if not all_times:
        raise ValueError("No timesteps to animate")
    start_time = min(all_times)
    end_time = max(all_times)

    fig, ax = plt.subplots(figsize=(9, 7))
    if occ_grid is not None:
        occ_grid.plot_grid(ax=ax)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)

    n = len(world_paths)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(n, 1)))
    trails = []
    heads = []
    disks = []
    for i, color in enumerate(colors):
        trail, = ax.plot([], [], color=color, linewidth=2, alpha=0.55, label=f"Agent {i + 1}")
        head, = ax.plot([], [], marker="o", color=color, markersize=9, zorder=5)
        circle = Circle((0, 0), radius=robot_radius_m, fill=False, edgecolor=color, linewidth=1.5, zorder=4)
        ax.add_patch(circle)
        trails.append(trail)
        heads.append(head)
        disks.append(circle)

    ax.legend(loc="upper right", fontsize=9)
    info = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=10,
                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    def _time_step() -> float:
        for seq in time_lists:
            if len(seq) > 1:
                return float(seq[1] - seq[0])
        return float(occ_grid.resolution)

    def _discrete_t(frame_time: float) -> int:
        step = _time_step()
        return int(round(frame_time / step)) if step > 0 else int(round(frame_time))

    def init():
        for trail, head, disk in zip(trails, heads, disks):
            trail.set_data([], [])
            head.set_data([], [])
            disk.set_visible(False)
        info.set_text("")
        return trails + heads + disks + [info]

    def update(frame_time):
        t_disc = _discrete_t(frame_time)
        conflict_pairs = []
        if grid_paths_for_conflict is not None:
            conflict_pairs = pairwise_conflict_at_time(
                grid_paths_for_conflict, t_disc, robot_radius_cells
            )
        elif conflict_threshold_m is not None:
            conflict_pairs = pairwise_conflict_at_time_world(
                world_paths, time_lists, frame_time, conflict_threshold_m
            )

        for i, (path, times) in enumerate(zip(world_paths, time_lists)):
            if not path or not times:
                continue
            cx, cy = get_position_at_time(frame_time, path, times)
            heads[i].set_data([cx], [cy])
            disks[i].center = (cx, cy)
            disks[i].set_edgecolor("red" if any(i in p for p in conflict_pairs) else colors[i])
            disks[i].set_visible(True)

            hx, hy = [], []
            for (px, py), pt in zip(path, times):
                if pt <= frame_time:
                    hx.append(px)
                    hy.append(py)
            hx.append(cx)
            hy.append(cy)
            trails[i].set_data(hx, hy)

        status = "CONFLICT" if conflict_pairs else "ok"
        info.set_text(
            f"t = {frame_time:.2f}  |  step = {t_disc}  |  {status}\n"
            f"conflicts: {conflict_pairs or 'none'}"
        )
        return trails + heads + disks + [info]

    frames = np.linspace(start_time, end_time, num_frames)
    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=False, interval=interval_ms
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving animation: {output_file}")
    try:
        if output_file.suffix.lower() == ".mp4":
            ani.save(str(output_file), writer="ffmpeg", dpi=dpi)
        else:
            ani.save(str(output_file), writer="pillow", dpi=dpi)
        print("Save complete.")
    except Exception as exc:
        print(f"Could not save animation ({exc}). Showing interactively.")
        if show:
            plt.show()
    finally:
        plt.close(fig)


def create_multi_panel_animation(
    occ_grid,
    panels: Sequence[Tuple[dict, str]],
    output_file: Path,
    num_frames: int = 200,
    interval_ms: int = 50,
) -> None:
    """GIF with one column per method (e.g. MILP | CBS | PP), synchronized time."""
    if not panels:
        return
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    all_end = 0.0
    for entry, _label in panels:
        for tseq in entry.get("time_lists") or []:
            if tseq:
                all_end = max(all_end, tseq[-1])
    frames = np.linspace(0.0, all_end, num_frames)

    panel_state = []
    for ax, (entry, label) in zip(axes, panels):
        occ_grid.plot_grid(ax=ax)
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        paths = entry["world_paths"]
        times = entry["time_lists"]
        colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(paths), 1)))
        artists = []
        for color in colors:
            trail, = ax.plot([], [], color=color, linewidth=2, alpha=0.55)
            head, = ax.plot([], [], marker="o", color=color, markersize=8)
            artists.append((trail, head))
        panel_state.append((entry, paths, times, artists))

    def update(ft):
        for entry, paths, times, artists in panel_state:
            grid_paths = entry.get("grid_paths")
            if grid_paths is not None:
                step = times[0][1] - times[0][0] if times and len(times[0]) > 1 else 1.0
                t_disc = int(round(ft / step)) if step > 0 else int(round(ft))
                conflicts = pairwise_conflict_at_time(
                    grid_paths, t_disc, entry["robot_radius_cells"]
                )
            else:
                conflicts = pairwise_conflict_at_time_world(
                    paths,
                    times,
                    ft,
                    entry.get("conflict_threshold_m", DEFAULT_CONFLICT_THRESHOLD_M),
                )
            for i, ((trail, head), path, tseq) in enumerate(zip(artists, paths, times)):
                if not path or not tseq:
                    continue
                cx, cy = get_position_at_time(ft, path, tseq)
                head.set_data([cx], [cy])
                head.set_color("red" if any(i in p for p in conflicts) else "C0")
                hx = [p[0] for p, pt in zip(path, tseq) if pt <= ft] + [cx]
                hy = [p[1] for p, pt in zip(path, tseq) if pt <= ft] + [cy]
                trail.set_data(hx, hy)
        fig.suptitle(f"t = {ft:.2f}")
        artists_out = []
        for _entry, _paths, _times, arts in panel_state:
            for trail, head in arts:
                artists_out.extend([trail, head])
        return artists_out

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving multi-panel animation: {output_file}")
    try:
        ani.save(str(output_file), writer="pillow")
    except Exception as exc:
        print(f"Multi-panel save failed: {exc}")
    plt.close(fig)


def create_side_by_side_animation(
    occ_grid,
    left: dict,
    right: dict,
    output_file: Path,
    left_label: str = "CBS",
    right_label: str = "PP",
    num_frames: int = 200,
    interval_ms: int = 50,
) -> None:
    """Single GIF with CBS (left) and PP (right) panels, synchronized time."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    panels = [(ax_l, left, left_label), (ax_r, right, right_label)]
    all_end = 0.0
    for _ax, entry, _lbl in panels:
        for tseq in entry.get("time_lists") or []:
            if tseq:
                all_end = max(all_end, tseq[-1])
    start_time = 0.0
    frames = np.linspace(start_time, all_end, num_frames)

    artists = []
    for ax, entry, label in panels:
        occ_grid.plot_grid(ax=ax)
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        paths = entry["world_paths"]
        times = entry["time_lists"]
        colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(paths), 1)))
        panel_art = []
        for i, color in enumerate(colors):
            trail, = ax.plot([], [], color=color, linewidth=2, alpha=0.55)
            head, = ax.plot([], [], marker="o", color=color, markersize=8)
            panel_art.append((trail, head))
        artists.append((paths, times, panel_art))

    def update(ft):
        for (paths, times, panel_art), (_ax, entry, _label) in zip(artists, panels):
            grid_paths = entry["grid_paths"]
            r_cells = entry["robot_radius_cells"]
            t_disc = int(round(ft / (times[0][1] - times[0][0] if times and len(times[0]) > 1 else 1.0)))
            conflicts = pairwise_conflict_at_time(grid_paths, t_disc, r_cells)
            for i, ((trail, head), path, tseq) in enumerate(zip(panel_art, paths, times)):
                if not path:
                    continue
                cx, cy = get_position_at_time(ft, path, tseq)
                head.set_data([cx], [cy])
                head.set_color("red" if any(i in p for p in conflicts) else "C0")
                hx = [p[0] for p, pt in zip(path, tseq) if pt <= ft] + [cx]
                hy = [p[1] for p, pt in zip(path, tseq) if pt <= ft] + [cy]
                trail.set_data(hx, hy)
        fig.suptitle(f"t = {ft:.2f}")
        return [a for _p in artists for art in _p[2] for a in art]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving side-by-side animation: {output_file}")
    try:
        ani.save(str(output_file), writer="pillow")
    except Exception as exc:
        print(f"Side-by-side save failed: {exc}")
    plt.close(fig)


