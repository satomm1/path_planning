"""
Offline precompute of directional wall-distance cache (.npz) for a scenario.

Example:
    python -m social_path_planning.precompute_wall_distances --scenario y2e2

Re-save an existing cache to add/update ``D_right_ros`` (no raycast):

    python -m social_path_planning.precompute_wall_distances --scenario y2e2 --resave-ros
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.wall_distance_cache import (
    NPZ_D_RIGHT,
    precompute_d_right,
    save_wall_distance_cache,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Precompute HxWx8 wall-distance tensor and save as .npz next to the map."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Name in grid_scenarios.json (e.g. y2e2).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path (default: environments/<scenario>_wall_dist.npz).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing cache file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (stderr).",
    )
    parser.add_argument(
        "--resave-ros",
        action="store_true",
        help="Load existing cache .npz and re-save it (adds/updates D_right_ros; no raycast).",
    )
    args = parser.parse_args(argv)

    module_dir = Path(__file__).resolve().parent
    out = (
        Path(args.output).expanduser()
        if args.output
        else module_dir / "environments" / f"{args.scenario}_wall_dist.npz"
    )
    out = out.resolve()

    if args.resave_ros:
        if not out.is_file():
            print(f"--resave-ros: cache file not found: {out}", file=sys.stderr)
            return 1
    elif out.is_file() and not args.force:
        print(f"Refusing to overwrite existing file: {out}\nUse --force to rebuild.", file=sys.stderr)
        return 1

    occ, map_size, map_resolution = load_grid_scenario(args.scenario, plot=False)
    map_dim = [round(map_size[i] / map_resolution) for i in range(2)]
    occ_grid = StochOccupancyGrid2D(
        map_resolution,
        map_dim[0],
        map_dim[1],
        0,
        0,
        10,
        occ.T,
    )

    if args.resave_ros:
        with np.load(out, allow_pickle=False) as data:
            d_right = np.asarray(data[NPZ_D_RIGHT], dtype=np.float64)
        if d_right.shape != (occ_grid.height, occ_grid.width, 8):
            print(
                f"--resave-ros: {NPZ_D_RIGHT} shape {d_right.shape} != "
                f"expected {(occ_grid.height, occ_grid.width, 8)}",
                file=sys.stderr,
            )
            return 1
        save_wall_distance_cache(out, occ_grid, d_right)
        print(f"Re-saved {out} (including ROS layout tensor).")
        return 0

    print(
        f"Precomputing wall distances ({occ_grid.width}x{occ_grid.height}x8)...",
        file=sys.stderr,
    )
    d_right = precompute_d_right(occ_grid, show_progress=not args.no_progress)
    save_wall_distance_cache(out, occ_grid, d_right)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
