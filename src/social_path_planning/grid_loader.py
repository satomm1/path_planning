import json
import argparse
from pathlib import Path
import ast

import numpy as np

_SUPPORTED_OPS = [
    "outer_boundary",
    "rect",
    "circle",
    "rotated_square_set",
]
_PLOT_FIGSIZE = (8, 8)
_PLOT_DPI = 300


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


def _resolve_config_path(config_path=None):
    if config_path:
        raw_path = Path(config_path).expanduser()
        if raw_path.is_absolute():
            return raw_path
        module_dir = Path(__file__).resolve().parent
        cwd_candidate = Path.cwd() / raw_path
        module_candidate = module_dir / raw_path
        return cwd_candidate if cwd_candidate.exists() else module_candidate
    return _default_config_path()


def _resolve_relative_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_yaml_scalar(raw_value):
    raw = raw_value.strip()
    if not raw:
        return ""

    if (raw.startswith("[") and raw.endswith("]")) or (
        raw.startswith("{") and raw.endswith("}")
    ):
        return ast.literal_eval(raw)

    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    return raw


def _load_simple_yaml(yaml_path):
    data = {}
    with yaml_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.split("#", 1)[0].strip()
            if not clean or ":" not in clean:
                continue
            key, value = clean.split(":", 1)
            data[key.strip()] = _parse_yaml_scalar(value)
    return data


def _read_pgm_token(file_obj):
    token = bytearray()
    while True:
        ch = file_obj.read(1)
        if not ch:
            break
        if ch.isspace():
            if token:
                break
            continue
        if ch == b"#":
            file_obj.readline()
            if token:
                break
            continue
        token.extend(ch)
    return bytes(token)


def _load_pgm_normalized(pgm_path):
    with pgm_path.open("rb") as f:
        magic = _read_pgm_token(f)
        if magic != b"P5":
            raise ValueError(f"Unsupported PGM format in '{pgm_path}': expected P5.")

        width_token = _read_pgm_token(f)
        height_token = _read_pgm_token(f)
        maxval_token = _read_pgm_token(f)
        if not width_token or not height_token or not maxval_token:
            raise ValueError(f"Invalid PGM header in '{pgm_path}'.")

        width = int(width_token)
        height = int(height_token)
        maxval = int(maxval_token)
        if width <= 0 or height <= 0 or maxval <= 0:
            raise ValueError(f"Invalid PGM dimensions/maxval in '{pgm_path}'.")

        if maxval < 256:
            dtype = np.uint8
        else:
            dtype = ">u2"

        count = width * height
        data = np.fromfile(f, dtype=dtype, count=count)
        if data.size != count:
            raise ValueError(f"Incomplete pixel data in '{pgm_path}'.")

        image = data.reshape((height, width)).astype(np.float32) / float(maxval)
        return image, width, height


def _crop_unknown_only_border(occ_xy):
    known = occ_xy != -1
    if not np.any(known):
        return occ_xy

    known_x = np.any(known, axis=1)
    known_y = np.any(known, axis=0)
    x_idx = np.where(known_x)[0]
    y_idx = np.where(known_y)[0]
    x0, x1 = x_idx[0], x_idx[-1] + 1
    y0, y1 = y_idx[0], y_idx[-1] + 1
    return occ_xy[x0:x1, y0:y1]


def _load_occ_from_map_yaml(map_yaml_path, crop_unknown=False):
    yaml_data = _load_simple_yaml(map_yaml_path)
    required = {"image", "resolution", "negate", "occupied_thresh", "free_thresh"}
    missing = required - set(yaml_data.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Map YAML '{map_yaml_path}' missing keys: {missing_str}")

    image_path = _resolve_relative_path(yaml_data["image"], map_yaml_path.parent)
    if not image_path.exists():
        raise FileNotFoundError(f"Map image not found: {image_path}")

    image_norm, width, height = _load_pgm_normalized(image_path)

    resolution = float(yaml_data["resolution"])
    negate = int(yaml_data["negate"])
    occupied_thresh = float(yaml_data["occupied_thresh"])
    free_thresh = float(yaml_data["free_thresh"])
    if resolution <= 0:
        raise ValueError(f"Map YAML '{map_yaml_path}' resolution must be positive.")
    if negate not in (0, 1):
        raise ValueError(f"Map YAML '{map_yaml_path}' negate must be 0 or 1.")

    occ_prob = image_norm if negate == 1 else (1.0 - image_norm)
    occ_yx = np.full((height, width), -1.0, dtype=np.float32)
    occ_yx[occ_prob > occupied_thresh] = 1.0
    occ_yx[occ_prob < free_thresh] = 0.0

    # ROS image origin is top-left; flip so grid y increases upward.
    occ_yx = np.flipud(occ_yx)
    occ_xy = occ_yx.T
    if crop_unknown:
        occ_xy = _crop_unknown_only_border(occ_xy)

    map_size = [occ_xy.shape[0] * resolution, occ_xy.shape[1] * resolution]
    return occ_xy, map_size, resolution


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
    path = _resolve_config_path(config_path=config_path)
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
    plot_title=None,
):
    config_file = _resolve_config_path(config_path=config_path)
    config = load_grid_config(config_path=config_path)
    if scenario_name not in config:
        available = ", ".join(sorted(config.keys()))
        raise KeyError(
            f"Unknown grid scenario '{scenario_name}'. Available scenarios: {available}"
        )

    scenario = config[scenario_name]
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario '{scenario_name}' must be a JSON object.")

    if "map_yaml" in scenario:
        if not isinstance(scenario["map_yaml"], str):
            raise ValueError(
                f"Scenario '{scenario_name}' map_yaml must be a string path."
            )
        crop_unknown = scenario.get("crop_unknown", True)
        if not isinstance(crop_unknown, bool):
            raise ValueError(
                f"Scenario '{scenario_name}' crop_unknown must be true or false."
            )
        map_yaml_path = _resolve_relative_path(scenario["map_yaml"], config_file.parent)
        occ, map_size, map_resolution = _load_occ_from_map_yaml(
            map_yaml_path, crop_unknown=crop_unknown
        )
    elif "operations" in scenario:
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
    else:
        raise ValueError(
            f"Scenario '{scenario_name}' must define either 'operations' or 'map_yaml'."
        )

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        cmap = ListedColormap(["gray", "#9DC6F2", "black"])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(figsize=_PLOT_FIGSIZE, dpi=_PLOT_DPI)
        ax.imshow(
            occ.T,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="lower",
            extent=[0, map_size[0], 0, map_size[1]],
            aspect="equal",
        )
        title = (
            plot_title
            if isinstance(plot_title, str) and plot_title.strip()
            else f"Scenario Occupancy Grid: {scenario_name}"
        )
        ax.set_title(title, fontsize=24)
        ax.set_xlabel("X (m)", fontsize=20)
        ax.set_ylabel("Y (m)", fontsize=20)

        # Set tick font size
        ax.tick_params(axis="both", which="major", labelsize=20)

        if save_path:
            save_target = Path(save_path)
            save_target.parent.mkdir(parents=True, exist_ok=True)
            # Keep output dimensions consistent across scenarios.
            fig.savefig(str(save_target), dpi=_PLOT_DPI)
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
        "--title",
        default=None,
        help="Optional plot title (used for display/save).",
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
        plot_title=args.title,
    )