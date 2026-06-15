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
        ("success", "Success rate [%]"),
        ("runtime", "Solver runtime [s]"),
        ("makespan", "Makespan [s]"),
    ]
    for ax, (field, ylabel) in zip(axes, panels):
        for method in methods:
            y = _series(rows, method, agents, field, dt=dt, event_makespan_by_n=event_makespan_by_n)
            ax.plot(agents, y, marker="o", label=METHOD_LABELS.get(method, method))
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        if field == "runtime":
            ax.set_yscale("log")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
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
