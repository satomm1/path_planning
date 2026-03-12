import json
import argparse
from pathlib import Path

import numpy as np

_SUPPORTED_OPS = [
    "outer_boundary",
    "rect",
    "circle",
    "rotated_square_set",
]


def _default_config_path():
    module_dir = Path(__file__).resolve().parent
    candidates = [
        module_dir / "environments" / "grid_scenarios.json",
        module_dir / "grid_scenarios.json",
        Path.cwd() / "environments" / "grid_scenarios.json",
        Path.cwd() / "grid_scenarios.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _point_to_grid(x, map_resolution):
    return round(x / map_resolution)


def _validate_numeric_pair(name, value, scenario_name):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(v, (int, float)) for v in value)
    ):
        raise ValueError(
            f"Scenario '{scenario_name}' {name} must be a list of two numbers."
        )


def _build_from_operations(scenario_name, map_size, map_resolution, operations):
    map_dim = [round(map_size[i] / map_resolution) for i in range(2)]
    occ = np.zeros(map_dim)
    xs = np.arange(map_dim[0]) * map_resolution
    ys = np.arange(map_dim[1]) * map_resolution
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            raise ValueError(
                f"Scenario '{scenario_name}' operation #{i} must be a JSON object."
            )

        op_type = op.get("type")
        if op_type not in _SUPPORTED_OPS:
            available = ", ".join(_SUPPORTED_OPS)
            raise ValueError(
                f"Scenario '{scenario_name}' operation #{i} has unsupported type "
                f"'{op_type}'. Supported types: {available}"
            )

        value = op.get("value")
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Scenario '{scenario_name}' operation #{i} must include numeric 'value'."
            )

        if op_type == "outer_boundary":
            thickness = op.get("thickness_cells", 3)
            if not isinstance(thickness, int) or thickness <= 0:
                raise ValueError(
                    f"Scenario '{scenario_name}' operation #{i} thickness_cells must be a positive integer."
                )
            occ[0:thickness, :] = value
            occ[:, 0:thickness] = value
            occ[-thickness:, :] = value
            occ[:, -thickness:] = value
            continue

        if op_type == "rect":
            if "x" not in op or "y" not in op:
                raise ValueError(
                    f"Scenario '{scenario_name}' operation #{i} rect must include 'x' and 'y' ranges."
                )
            _validate_numeric_pair("rect x range", op["x"], scenario_name)
            _validate_numeric_pair("rect y range", op["y"], scenario_name)

            x0 = _point_to_grid(op["x"][0], map_resolution)
            x1 = _point_to_grid(op["x"][1], map_resolution)
            y0 = _point_to_grid(op["y"][0], map_resolution)
            y1 = _point_to_grid(op["y"][1], map_resolution)
            occ[x0:x1, y0:y1] = value
            continue

        if op_type == "circle":
            center = op.get("center")
            radius = op.get("radius")
            _validate_numeric_pair("circle center", center, scenario_name)
            if not isinstance(radius, (int, float)) or radius < 0:
                raise ValueError(
                    f"Scenario '{scenario_name}' operation #{i} circle radius must be non-negative."
                )
            dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            occ[dist <= radius] = value
            continue

        centers = op.get("centers")
        side = op.get("side")
        theta_deg = op.get("theta_deg")
        if not isinstance(centers, list) or not centers:
            raise ValueError(
                f"Scenario '{scenario_name}' operation #{i} rotated_square_set must include non-empty 'centers'."
            )
        if not isinstance(side, (int, float)) or side <= 0:
            raise ValueError(
                f"Scenario '{scenario_name}' operation #{i} side must be positive."
            )
        if not isinstance(theta_deg, (int, float)):
            raise ValueError(
                f"Scenario '{scenario_name}' operation #{i} theta_deg must be numeric."
            )

        half = side / 2.0
        theta = np.deg2rad(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        for c_idx, center in enumerate(centers):
            _validate_numeric_pair(
                f"rotated_square_set center index {c_idx}", center, scenario_name
            )
            Xc = X - center[0]
            Yc = Y - center[1]
            xr = c * Xc + s * Yc
            yr = -s * Xc + c * Yc
            mask = (np.abs(xr) <= half) & (np.abs(yr) <= half)
            occ[mask] = value

    return occ


def load_grid_config(config_path=None):
    if config_path:
        raw_path = Path(config_path).expanduser()
        if raw_path.is_absolute():
            path = raw_path
        else:
            module_dir = Path(__file__).resolve().parent
            cwd_candidate = Path.cwd() / raw_path
            module_candidate = module_dir / raw_path
            path = cwd_candidate if cwd_candidate.exists() else module_candidate
    else:
        path = _default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Grid scenario config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Grid scenario config must be a JSON object.")

    return config


def load_grid_scenario(
    scenario_name,
    config_path=None,
    plot=False,
    save_path=None,
    show_plot=True,
):
    config = load_grid_config(config_path=config_path)
    if scenario_name not in config:
        available = ", ".join(sorted(config.keys()))
        raise KeyError(
            f"Unknown grid scenario '{scenario_name}'. Available scenarios: {available}"
        )

    scenario = config[scenario_name]
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario '{scenario_name}' must be a JSON object.")

    required_keys = {"map_size", "map_resolution", "operations"}
    missing = required_keys - set(scenario.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Scenario '{scenario_name}' missing keys: {missing_str}")

    map_size = scenario["map_size"]
    map_resolution = scenario["map_resolution"]
    operations = scenario["operations"]

    _validate_numeric_pair("map_size", map_size, scenario_name)
    if not isinstance(map_resolution, (int, float)) or map_resolution <= 0:
        raise ValueError(
            f"Scenario '{scenario_name}' map_resolution must be a positive number."
        )
    if not isinstance(operations, list):
        raise ValueError(
            f"Scenario '{scenario_name}' operations must be a list of operation objects."
        )

    occ = _build_from_operations(
        scenario_name=scenario_name,
        map_size=list(map_size),
        map_resolution=float(map_resolution),
        operations=operations,
    )

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        cmap = ListedColormap(["gray", "#9DC6F2", "black"])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            occ.T,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="lower",
            extent=[0, map_size[0], 0, map_size[1]],
            aspect="equal",
        )
        ax.set_title(f"Scenario Occupancy Grid: {scenario_name}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        if save_path:
            save_target = Path(save_path)
            save_target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_target), dpi=300, bbox_inches="tight")
            print(f"Saved scenario figure to {save_target}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return occ, list(map_size), float(map_resolution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and visualize occupancy grid scenarios."
    )
    parser.add_argument(
        "--scenario",
        default="sample2_default",
        help="Scenario name from environments/grid_scenarios.json",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a scenario config JSON file.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output image path (e.g. scenario.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a plot window (useful with --save).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    args = parser.parse_args()

    config = load_grid_config(args.config)
    if args.list:
        print("Available scenarios:")
        for name in sorted(config.keys()):
            print(f"- {name}")
        raise SystemExit(0)

    should_plot = (not args.no_show) or bool(args.save)
    load_grid_scenario(
        args.scenario,
        config_path=args.config,
        plot=should_plot,
        save_path=args.save,
        show_plot=not args.no_show,
    )