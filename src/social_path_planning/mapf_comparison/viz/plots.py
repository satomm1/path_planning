"""Static route overlays and snapshot grids."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence
import matplotlib.pyplot as plt
import numpy as np
from social_path_planning.mapf_comparison.viz.conflicts import (
    DEFAULT_CONFLICT_THRESHOLD_M,
    pairwise_conflict_at_time,
    pairwise_conflict_at_time_world,
)
from social_path_planning.mapf_comparison.viz.interpolation import get_position_at_time

def plot_routes_overlay(
    occ_grid,
    fixed_paths: Sequence[Sequence],
    results: Dict[str, dict],
    output_path: Path,
    dpi: int = 200,
) -> None:
    """One map: fixed A* (dashed), CBS and PP full spatial routes."""
    fig, ax = plt.subplots(figsize=(10, 8))
    occ_grid.plot_grid(ax=ax)

    if fixed_paths:
        for i, path in enumerate(fixed_paths):
            xs, ys = zip(*path)
            ax.plot(xs, ys, color="steelblue", linestyle="--", linewidth=1.2, alpha=0.7,
                    label="Path bank (A*)" if i == 0 else None)

    styles = {
        "milp": ("MILP (social A* paths)", "darkviolet", "-"),
        "event_milp": ("Event MILP (interest waypoints)", "darkorange", "-"),
        "cbs": ("CBS", "crimson", "-"),
        "pp": ("PP (longest first)", "forestgreen", "-."),
    }
    for key, (label, color, ls) in styles.items():
        entry = results.get(key)
        if not entry or not entry.get("world_polylines"):
            continue
        for i, poly in enumerate(entry["world_polylines"]):
            if not poly:
                continue
            xs, ys = zip(*poly)
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.2,
                    label=label if i == 0 else None)

    for key, color in (
        ("milp", "darkviolet"),
        ("event_milp", "darkorange"),
        ("cbs", "crimson"),
        ("pp", "forestgreen"),
    ):
        entry = results.get(key)
        if not entry:
            continue
        starts = entry.get("starts_world") or []
        goals = entry.get("goals_world") or []
        for (sx, sy), (gx, gy) in zip(starts, goals):
            ax.scatter(sx, sy, marker="o", s=60, color=color, edgecolors="k", linewidths=0.5, zorder=6)
            ax.scatter(gx, gy, marker="*", s=140, color=color, edgecolors="k", linewidths=0.5, zorder=6)

    ax.set_title("Spatial routes: path bank vs MILP / Event MILP / CBS / PP")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved static comparison: {output_path}")


def plot_side_by_side_snapshots(
    occ_grid,
    snapshot_results: Dict[str, dict],
    output_path: Path,
    snapshot_times: Sequence[float],
    dpi: int = 150,
) -> None:
    """Grid of panels: methods (rows) x snapshot times (cols)."""
    methods = [
        k
        for k in ("milp", "event_milp", "cbs", "pp")
        if k in snapshot_results and snapshot_results[k].get("success")
    ]
    if not methods:
        return

    ncols = len(snapshot_times)
    fig, axes = plt.subplots(len(methods), ncols, figsize=(4 * ncols, 4 * len(methods)), squeeze=False)
    for row, method in enumerate(methods):
        entry = snapshot_results[method]
        paths = entry["world_paths"]
        times = entry["time_lists"]
        for col, t_snap in enumerate(snapshot_times):
            ax = axes[row, col]
            occ_grid.plot_grid(ax=ax)
            colors = plt.cm.tab10(np.linspace(0, 0.9, len(paths)))
            grid_paths = entry.get("grid_paths")
            if grid_paths is not None:
                t_disc = int(round(t_snap / (times[0][1] - times[0][0] if len(times[0]) > 1 else 1.0)))
                conflicts = pairwise_conflict_at_time(
                    grid_paths, t_disc, entry["robot_radius_cells"]
                )
            else:
                conflicts = pairwise_conflict_at_time_world(
                    paths,
                    times,
                    t_snap,
                    entry.get("conflict_threshold_m", DEFAULT_CONFLICT_THRESHOLD_M),
                )
            for i, (path, tseq, color) in enumerate(zip(paths, times, colors)):
                xs, ys = zip(*path) if path else ([], [])
                ax.plot(xs, ys, color=color, alpha=0.35, linewidth=1)
                if path and tseq:
                    cx, cy = get_position_at_time(t_snap, path, tseq)
                    edge = "red" if any(i in p for p in conflicts) else color
                    ax.scatter(cx, cy, s=80, color=color, edgecolors=edge, linewidths=2, zorder=5)
            title = {"milp": "MILP", "event_milp": "Event MILP", "cbs": "CBS", "pp": "PP"}.get(method, method.upper())
            ax.set_title(f"{title}  t={t_snap:.1f}")
            ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved snapshots: {output_path}")


