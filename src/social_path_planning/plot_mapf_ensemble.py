"""3-panel scaling figure from ensemble benchmark summary + manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from social_path_planning.mapf_comparison.checkpoint import coerce_bool, load_trial_rows
from social_path_planning.mapf_comparison.constants import MAPF_METHODS, METHODS
from social_path_planning.mapf_comparison.motion import cell_size_m
from social_path_planning.mapf_comparison.plot import METHOD_LABELS

ENSEMBLE_METHOD_LABELS = {
    **METHOD_LABELS,
    "pp_path_length": "PP",
    "event_milp_soc": "MICP",
}


def _dt_step(map_resolution: float, mapf_downsample: int, max_velocity_mps: float) -> float:
    return math.sqrt(2) * cell_size_m(map_resolution, mapf_downsample) / max(max_velocity_mps, 1e-9)


def _load_rows(summary_csv: Path) -> list[dict]:
    with summary_csv.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _event_makespan_avg_by_n(trials_csv: Path) -> dict[int, float]:
    by_n: dict[int, list[float]] = {}
    for row in load_trial_rows(trials_csv):
        if row.get("method") != "event_milp_soc" or not coerce_bool(row.get("success")):
            continue
        val = row.get("makespan_seconds")
        if val is None:
            continue
        by_n.setdefault(int(row["num_agents"]), []).append(float(val))
    return {n: float(np.mean(vals)) for n, vals in by_n.items() if vals}


def _pad_axes_for_labels(ax, *, log_y: bool = False) -> None:
    xmin, xmax = ax.get_xlim()
    xspan = xmax - xmin if xmax > xmin else 1.0
    ax.set_xlim(xmin - 0.02 * xspan, xmax + 0.08 * xspan)

    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        return
    if log_y and ymin > 0:
        ax.set_ylim(ymin, ymax * 1.25)
    else:
        pad = (ymax - ymin) * 0.2
        ax.set_ylim(ymin - pad * 0.1, ymax + pad)


def _annotate_line_end(ax, xs: list[int], ys: list[float], label: str, color) -> None:
    y_arr = np.asarray(ys, dtype=float)
    valid = np.isfinite(y_arr)
    if not valid.any():
        return
    idx = int(np.where(valid)[0][-1])
    ax.annotate(
        label,
        xy=(xs[idx], y_arr[idx]),
        xytext=(0, 5),
        textcoords="offset points",
        va="bottom",
        ha="center",
        color=color,
        fontsize=8,
        clip_on=False,
    )


def _series(
    rows: list[dict],
    method: str,
    agents: list[int],
    field: str,
    *,
    dt: float,
    event_makespan_by_n: dict[int, float] | None = None,
) -> list[float]:
    out = []
    for n in agents:
        match = [r for r in rows if r["method"] == method and int(r["num_agents"]) == n]
        if not match:
            out.append(float("nan"))
            continue
        row = match[0]
        if field == "success":
            out.append(100.0 * float(row["success_rate"]))
            continue
        if int(row.get("success_count") or 0) == 0:
            out.append(float("nan"))
            continue
        if field == "runtime":
            out.append(float(row["avg_solver_runtime_s"]))
        elif field == "makespan":
            if method in MAPF_METHODS:
                out.append(float(row["avg_makespan_timesteps"]) * dt)
            elif event_makespan_by_n is not None and int(row["num_agents"]) in event_makespan_by_n:
                out.append(event_makespan_by_n[int(row["num_agents"])])
            else:
                out.append(float(row["avg_makespan_seconds"]))
    return out


def plot_scaling(
    summary_csv: Path,
    manifest_json: Path,
    output: Path,
    dpi: int,
    trials_csv: Path | None = None,
) -> None:
    with manifest_json.open(encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    dt = _dt_step(
        float(cfg["map_resolution"]),
        int(cfg.get("mapf_downsample") or 1),
        float(cfg["motion"]["max_velocity_mps"]),
    )
    rows = _load_rows(summary_csv)
    agents = sorted({int(r["num_agents"]) for r in rows})
    methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
    event_makespan_by_n = _event_makespan_avg_by_n(trials_csv) if trials_csv and trials_csv.is_file() else None

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    panels = [
        ("success", "Success Rate (%)"),
        ("runtime", "Solver Runtime (s)"),
        ("makespan", "Makespan (s)"),
    ]
    panel_titles = {
        "success": "Solver Success",
        "runtime": "Solver Runtime",
        "makespan": "Solution Makespan"
    }
    for ax, (field, ylabel) in zip(axes, panels):
        for method in methods:
            y = _series(rows, method, agents, field, dt=dt, event_makespan_by_n=event_makespan_by_n)
            (line,) = ax.plot(agents, y, marker="o")
            _annotate_line_end(
                ax,
                agents,
                y,
                ENSEMBLE_METHOD_LABELS.get(method, method),
                line.get_color(),
            )
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(panel_titles.get(field, field.capitalize()))
        if field == "runtime":
            ax.set_yscale("log")
        _pad_axes_for_labels(ax, log_y=(field == "runtime"))
    fig.tight_layout(pad=1.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plot ensemble MAPF scaling figure (3 panels).")
    parser.add_argument("--output-prefix", required=True, help="Benchmark output prefix")
    parser.add_argument("--output", default=None, help="PNG path (default: {prefix}_scaling.png)")
    parser.add_argument("--dpi", type=int, default=300)
    cli = parser.parse_args(argv)

    prefix = Path(cli.output_prefix)
    summary_csv = Path(f"{prefix}_summary.csv")
    manifest_json = Path(f"{prefix}_manifest.json")
    if not summary_csv.is_file():
        raise FileNotFoundError(summary_csv)
    if not manifest_json.is_file():
        raise FileNotFoundError(manifest_json)

    output = Path(cli.output) if cli.output else Path(f"{prefix}_scaling.png")
    trials_csv = Path(f"{prefix}_trials.csv")
    plot_scaling(summary_csv, manifest_json, output, cli.dpi, trials_csv=trials_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
