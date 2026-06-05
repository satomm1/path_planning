"""Plots for MAPF vs MILP benchmark results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "cbs": "CBS",
    "pp_path_length": "PP (longest path first)",
    "milp_soc": "MILP (sum of times)",
    "milp_makespan": "MILP (makespan obj)",
}

METHOD_ORDER = ["milp_soc", "milp_makespan", "cbs", "pp_path_length"]


def _load_summary_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metric_value(row: dict, metric: str):
    if metric == "soc":
        if row.get("soc_seconds") not in (None, ""):
            return float(row["soc_seconds"])
        return float("nan")
    if metric == "makespan":
        if row.get("makespan_seconds") not in (None, ""):
            return float(row["makespan_seconds"])
        return float("nan")
    if metric == "path_length":
        if row.get("total_path_length_m") not in (None, ""):
            return float(row["total_path_length_m"])
        return float("nan")
    if metric == "runtime":
        val = row.get("solver_runtime_s") or row.get("runtime_s")
        if val not in (None, ""):
            return float(val)
        return float("nan")
    return float("nan")


def plot_metrics_bars(summary_rows: Sequence[dict], output_path: Path, dpi: int = 150):
    """Grouped bar charts: SOC, makespan, path length, solver runtime vs stage size."""
    stages = sorted({int(r["stage_size"]) for r in summary_rows})
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in summary_rows)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    metrics = [
        ("soc", "Sum of completion times [s]"),
        ("makespan", "Makespan [s]"),
        ("path_length", "Total path length [m]"),
        ("runtime", "Solver runtime [s]"),
    ]

    x = np.arange(len(stages))
    width = 0.8 / max(len(methods), 1)

    for ax, (metric_key, ylabel) in zip(axes, metrics):
        for mi, method in enumerate(methods):
            vals = []
            for stage in stages:
                match = [r for r in summary_rows if int(r["stage_size"]) == stage and r["method"] == method]
                if not match or match[0].get("success") in ("False", "false", False, "0", 0):
                    vals.append(np.nan)
                else:
                    vals.append(_metric_value(match[0], metric_key))
            offset = (mi - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, label=METHOD_LABELS.get(method, method))

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in stages])
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Multi-agent coordination (native paths, shared max velocity)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved metrics plot: {output_path}")


def plot_example_map(
    occ_grid,
    fixed_paths: Sequence[Sequence],
    cbs_paths_world: Sequence[Sequence],
    pp_paths_world: Sequence[Sequence],
    output_path: Path,
    dpi: int = 200,
):
    """Overlay fixed A* polylines and MAPF replanned paths (delegates to viz.plot_routes_overlay)."""
    from social_path_planning.mapf_comparison.viz.plots import plot_routes_overlay

    results = {}
    if cbs_paths_world:
        results["cbs"] = {"world_polylines": list(cbs_paths_world), "success": True}
    if pp_paths_world:
        results["pp"] = {"world_polylines": list(pp_paths_world), "success": True}
    plot_routes_overlay(occ_grid, fixed_paths, results, output_path, dpi=dpi)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot MAPF comparison summary CSV.")
    parser.add_argument(
        "--summary-csv",
        default="results/mapf_comparison/sample2_mapf_summary.csv",
    )
    parser.add_argument("--output", default=None, help="PNG path (default: alongside CSV)")
    cli = parser.parse_args(argv)

    summary_path = Path(cli.summary_csv)
    rows = _load_summary_csv(summary_path)
    out = Path(cli.output) if cli.output else summary_path.with_name(summary_path.stem.replace("_summary", "_metrics") + ".png")
    plot_metrics_bars(rows, out)
    return 0
