import json
from pathlib import Path

import numpy as np

_SUPPORTED_OPS = [
    "outer_boundary",
    "rect",
    "circle",
    "rotated_square_set",
]


def _default_config_path():
    return Path(__file__).with_name("grid_scenarios.json")


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
    path = Path(config_path) if config_path else _default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Grid scenario config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Grid scenario config must be a JSON object.")

    return config


def load_grid_scenario(scenario_name, config_path=None, plot=False):
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
        plt.imshow(
            occ.T,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="lower",
            extent=[0, map_size[0], 0, map_size[1]],
            aspect="equal",
        )
        plt.title("Scenario Occupancy Grid")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

    return occ, list(map_size), float(map_resolution)
