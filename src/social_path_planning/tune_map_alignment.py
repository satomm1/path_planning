"""
Interactive tool to pick ``map_align_deg`` for map_yaml scenarios.

Loads the occupancy grid **without** applying ``map_align_deg`` from JSON, then
lets you sweep the angle with a slider. Copy the printed value into
``environments/grid_scenarios.json`` for that scenario, **or** export a ``.pgm`` +
``.yaml`` for the current alignment (``Save for ROS`` button, or ``--export``).
Exports use ``occupancy_encoding: ros_int8`` (costmap-style PGM bytes ``205`` /
``254`` / ``0`` for unknown / free / occupied). Pass ``occupancy_encoding='probability'``
to ``write_ros_map_pgm_yaml`` if you need the older grayscale probability maps.

Usage (from repository root, with PYTHONPATH=src or after install)::

    python -m social_path_planning.tune_map_alignment --scenario y2e2

    python -m social_path_planning.tune_map_alignment --scenario y2e2 \\
        --export C:/maps/y2e2_aligned.yaml --angle -4.3

Sign convention matches ``align_occ_map_raster``: positive degrees rotate the
raster counter-clockwise (SciPy).
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.widgets import Button, CheckButtons, Slider

from social_path_planning.grid_loader import (
    _load_occ_from_map_yaml,
    _resolve_config_path,
    _resolve_relative_path,
    align_occ_map_raster,
    load_grid_config,
    write_ros_map_pgm_yaml,
)


def _load_raw_occ_for_map_scenario(scenario_name: str, config_path: str | None):
    """Load occ + resolution from a map_yaml scenario, before map_align_deg."""
    config_file = _resolve_config_path(config_path=config_path)
    config = load_grid_config(config_path=config_path)
    if scenario_name not in config:
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. Available: {sorted(config.keys())}"
        )
    scenario = config[scenario_name]
    if not isinstance(scenario, dict) or "map_yaml" not in scenario:
        raise ValueError(
            f"Scenario '{scenario_name}' must be a map_yaml scenario for alignment tuning."
        )
    map_yaml_path = _resolve_relative_path(scenario["map_yaml"], config_file.parent)
    crop_unknown = scenario.get("crop_unknown", True)
    if not isinstance(crop_unknown, bool):
        raise ValueError("crop_unknown must be true or false.")
    occ, map_size, resolution = _load_occ_from_map_yaml(
        map_yaml_path, crop_unknown=crop_unknown
    )
    map_align_crop_default = scenario.get("map_align_crop", True)
    return occ, float(resolution), bool(map_align_crop_default), map_yaml_path


def run_interactive(
    scenario_name: str,
    config_path: str | None,
    deg_min: float,
    deg_max: float,
    initial_deg: float,
    crop_known: bool,
):
    occ, resolution, _, map_yaml_path = _load_raw_occ_for_map_scenario(
        scenario_name, config_path
    )
    crop_state = [crop_known]

    cmap = ListedColormap(["gray", "#9DC6F2", "black"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    aligned, map_size = align_occ_map_raster(
        occ, initial_deg, resolution, crop_known=crop_state[0]
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.18, left=0.08, right=0.98, top=0.94)

    im = ax.imshow(
        aligned.T,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="lower",
        extent=[0, map_size[0], 0, map_size[1]],
        aspect="equal",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    title = ax.set_title(
        f"{scenario_name}  |  map_align_deg = {initial_deg:.2f}°  |  crop = {crop_state[0]}"
    )
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    ax_slider = plt.axes((0.12, 0.08, 0.76, 0.03))
    slider = Slider(
        ax_slider,
        "deg",
        deg_min,
        deg_max,
        valinit=float(initial_deg),
        valstep=0.1,
    )

    ax_check = plt.axes((0.12, 0.02, 0.25, 0.04))
    check = CheckButtons(ax_check, ["crop after rotate (map_align_crop)"], [crop_state[0]])

    ax_save = plt.axes((0.40, 0.02, 0.18, 0.04))
    btn_save = Button(ax_save, "Save for ROS")

    text_ax = plt.axes((0.60, 0.01, 0.38, 0.05))
    text_ax.axis("off")
    snippet = text_ax.text(
        0,
        0.5,
        "",
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="center",
        transform=text_ax.transAxes,
    )

    def _update_snippet(angle: float):
        snippet.set_text(
            f'Put in grid_scenarios.json for "{scenario_name}":\n'
            f'  "map_align_deg": {angle:.4f},\n'
            f'  "map_align_crop": {str(crop_state[0]).lower()},'
        )

    def _redraw(angle: float):
        al, ms = align_occ_map_raster(
            occ, angle, resolution, crop_known=crop_state[0]
        )
        im.set_data(al.T)
        im.set_extent([0, ms[0], 0, ms[1]])
        ax.set_xlim(0, ms[0])
        ax.set_ylim(0, ms[1])
        title.set_text(
            f"{scenario_name}  |  map_align_deg = {angle:.2f}°  |  crop = {crop_state[0]}"
        )
        _update_snippet(angle)
        fig.canvas.draw_idle()

    def on_slider_change(val):
        _redraw(float(val))

    def on_check(_label):
        crop_state[0] = check.get_status()[0]
        _redraw(slider.val)

    def on_save(_event):
        try:
            from tkinter import Tk, filedialog
        except ImportError:
            print("tkinter not available; use --export instead.", file=sys.stderr)
            return
        root = Tk()
        root.withdraw()
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("ROS map yaml", "*.yaml"), ("All files", "*.*")],
            initialfile=f"{scenario_name}_aligned.yaml",
        )
        root.destroy()
        if not path:
            return
        al, _ = align_occ_map_raster(
            occ, float(slider.val), resolution, crop_known=crop_state[0]
        )
        ypath, ppath = write_ros_map_pgm_yaml(
            al,
            resolution,
            path,
            source_map_yaml_path=map_yaml_path,
        )
        print(f"Wrote ROS map: {ypath}  {ppath}")

    btn_save.on_clicked(on_save)

    slider.on_changed(on_slider_change)
    check.on_clicked(on_check)
    _update_snippet(initial_deg)

    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive slider to tune map_align_deg for map_yaml scenarios."
    )
    parser.add_argument(
        "--scenario",
        default="y2e2",
        help="Scenario name from grid_scenarios.json (must use map_yaml).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to grid_scenarios.json.",
    )
    parser.add_argument(
        "--deg-min",
        type=float,
        default=-45.0,
        help="Slider minimum (degrees).",
    )
    parser.add_argument(
        "--deg-max",
        type=float,
        default=45.0,
        help="Slider maximum (degrees).",
    )
    parser.add_argument(
        "--initial-deg",
        type=float,
        default=0.0,
        help="Initial slider value (degrees).",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Start with map_align_crop off (tight bbox crop disabled after rotate).",
    )
    parser.add_argument(
        "--export",
        default=None,
        metavar="MAP.yaml",
        help="Write aligned ROS map .yaml and .pgm to this path and exit (no GUI).",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Alignment angle in degrees for --export (defaults to --initial-deg).",
    )
    args = parser.parse_args(argv)

    try:
        if args.export:
            angle = args.angle if args.angle is not None else args.initial_deg
            occ, resolution, _, map_yaml_path = _load_raw_occ_for_map_scenario(
                args.scenario, args.config
            )
            aligned, _ = align_occ_map_raster(
                occ, float(angle), resolution, crop_known=not args.no_crop
            )
            ypath, ppath = write_ros_map_pgm_yaml(
                aligned,
                resolution,
                args.export,
                source_map_yaml_path=map_yaml_path,
            )
            print(f"Wrote ROS map: {ypath}  {ppath}")
            return 0

        if args.deg_min >= args.deg_max:
            print("error: --deg-min must be < --deg-max", file=sys.stderr)
            return 2

        run_interactive(
            scenario_name=args.scenario,
            config_path=args.config,
            deg_min=args.deg_min,
            deg_max=args.deg_max,
            initial_deg=args.initial_deg,
            crop_known=not args.no_crop,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
