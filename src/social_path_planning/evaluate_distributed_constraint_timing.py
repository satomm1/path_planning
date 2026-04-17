import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate centralized vs distributed constraint timing from pair timing CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Input CSV produced by distributed constraint timing benchmark (*_pair_timings.csv).",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/distributed_eval/distributed_eval_summary.csv",
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--per-robot-csv",
        type=str,
        default="results/distributed_eval/distributed_eval_per_robot.csv",
        help="Output per-robot workload CSV path.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional output JSON report path.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="results/distributed_eval/distributed_eval_times.png",
        help="Output plot path for centralized vs distributed times.",
    )
    return parser.parse_args()


def _safe_float(value):
    if value is None or value == "":
        return 0.0
    return float(value)


def load_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return rows


def group_rows_by_run_and_stage(rows):
    grouped = defaultdict(list)
    for row in rows:
        run_id = row.get("run_id", "default_run")
        if "stage_size" not in row:
            raise ValueError("Missing required column: stage_size")
        stage_size = int(row["stage_size"])
        grouped[(run_id, stage_size)].append(row)
    return grouped


def expected_pairs(stage_size):
    return stage_size * (stage_size - 1) // 2


def build_pair_times(rows_for_stage, stage_size):
    pair_time = {}
    for row in rows_for_stage:
        i = int(row["robot_i"]) + 1  # convert to 1-based
        j = int(row["robot_j"]) + 1  # convert to 1-based
        if i == j:
            raise ValueError(f"Invalid pair with identical IDs ({i}, {j}) for stage {stage_size}")
        pair_key = tuple(sorted((i, j)))
        if pair_key[0] < 1 or pair_key[1] > stage_size:
            raise ValueError(f"Pair {pair_key} out of bounds for stage {stage_size}")
        if pair_key in pair_time:
            raise ValueError(f"Duplicate pair {pair_key} in stage {stage_size}")

        detect = _safe_float(row.get("pair_detect_time_s"))
        build = _safe_float(row.get("pair_constraint_build_time_s"))
        pair_time[pair_key] = detect + build

    exp = expected_pairs(stage_size)
    if len(pair_time) != exp:
        raise ValueError(
            f"Stage {stage_size} has {len(pair_time)} unique pairs, expected {exp}"
        )
    return pair_time


def assigned_pairs_for_robot(robot_id_1b, stage_size):
    half = stage_size // 2
    hop_count = half if robot_id_1b <= half else half - 1
    pairs = []
    for hop in range(1, hop_count + 1):
        other = ((robot_id_1b - 1 + hop) % stage_size) + 1
        pairs.append(tuple(sorted((robot_id_1b, other))))
    return pairs


def compute_stage_metrics(run_id, stage_size, rows_for_stage):
    if stage_size % 2 != 0:
        raise ValueError(f"Stage {stage_size} is odd; this assignment rule requires even robot counts.")
    if stage_size < 2:
        raise ValueError(f"Stage {stage_size} must be at least 2.")

    pair_time = build_pair_times(rows_for_stage, stage_size)
    centralized_total = sum(pair_time.values())

    per_robot_rows = []
    all_assigned = set()
    all_assigned_with_duplicates = []
    workload_values = []
    for robot_id in range(1, stage_size + 1):
        assigned_pairs = assigned_pairs_for_robot(robot_id, stage_size)
        workload_time = 0.0
        for pair in assigned_pairs:
            if pair not in pair_time:
                raise ValueError(
                    f"Assigned pair {pair} for robot {robot_id} missing from stage {stage_size} data."
                )
            workload_time += pair_time[pair]
            all_assigned.add(pair)
            all_assigned_with_duplicates.append(pair)

        workload_values.append(workload_time)
        per_robot_rows.append(
            {
                "run_id": run_id,
                "stage_size": stage_size,
                "robot_id_1based": robot_id,
                "robot_id_0based": robot_id - 1,
                "assigned_pair_count": len(assigned_pairs),
                "workload_time_s": workload_time,
            }
        )

    if len(set(all_assigned_with_duplicates)) != len(all_assigned_with_duplicates):
        raise ValueError(f"Pair assignment duplicates detected in stage {stage_size}")
    if all_assigned != set(pair_time.keys()):
        missing = sorted(set(pair_time.keys()) - all_assigned)
        extra = sorted(all_assigned - set(pair_time.keys()))
        raise ValueError(
            f"Pair coverage mismatch in stage {stage_size}. Missing={missing[:5]}, Extra={extra[:5]}"
        )

    distributed_makespan = max(workload_values) if workload_values else 0.0
    mean_workload = (sum(workload_values) / len(workload_values)) if workload_values else 0.0
    min_workload = min(workload_values) if workload_values else 0.0
    speedup = centralized_total / distributed_makespan if distributed_makespan > 0 else float("inf")
    imbalance_ratio = (
        (distributed_makespan - min_workload) / distributed_makespan if distributed_makespan > 0 else 0.0
    )
    max_over_mean = distributed_makespan / mean_workload if mean_workload > 0 else float("inf")

    summary_row = {
        "run_id": run_id,
        "stage_size": stage_size,
        "pair_count": expected_pairs(stage_size),
        "centralized_total_constraint_time_s": centralized_total,
        "distributed_makespan_time_s": distributed_makespan,
        "distributed_speedup_vs_centralized": speedup,
        "robot_workload_min_s": min_workload,
        "robot_workload_mean_s": mean_workload,
        "robot_workload_max_s": distributed_makespan,
        "imbalance_ratio": imbalance_ratio,
        "max_over_mean_ratio": max_over_mean,
    }
    return summary_row, per_robot_rows


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_time_comparison_plot(summary_rows, plot_path):
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["run_id"]].append(row)

    fig, ax = plt.subplots(figsize=(8, 5))
    for run_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda r: int(r["stage_size"]))
        stage_sizes = [int(r["stage_size"]) for r in ordered]
        centralized = [float(r["centralized_total_constraint_time_s"]) for r in ordered]
        distributed = [float(r["distributed_makespan_time_s"]) for r in ordered]
        ax.plot(stage_sizes, centralized, marker="o", linewidth=2, label=f"Centralized")
        ax.plot(stage_sizes, distributed, marker="s", linewidth=2, linestyle="--",
                label=f"Distributed")

    ax.set_xlabel("Number of Robots", fontsize=16)
    ax.set_ylabel("Constraint Computation Time (s)", fontsize=16)
    ax.set_title("Centralized vs Distributed Constraint Times", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.input_csv)
    grouped = group_rows_by_run_and_stage(rows)

    summary_rows = []
    per_robot_rows = []
    for (run_id, stage_size) in sorted(grouped.keys(), key=lambda k: (k[0], k[1])):
        stage_summary, stage_robot_rows = compute_stage_metrics(run_id, stage_size, grouped[(run_id, stage_size)])
        summary_rows.append(stage_summary)
        per_robot_rows.extend(stage_robot_rows)

    write_csv(
        args.summary_csv,
        fieldnames=[
            "run_id",
            "stage_size",
            "pair_count",
            "centralized_total_constraint_time_s",
            "distributed_makespan_time_s",
            "distributed_speedup_vs_centralized",
            "robot_workload_min_s",
            "robot_workload_mean_s",
            "robot_workload_max_s",
            "imbalance_ratio",
            "max_over_mean_ratio",
        ],
        rows=summary_rows,
    )
    write_csv(
        args.per_robot_csv,
        fieldnames=[
            "run_id",
            "stage_size",
            "robot_id_1based",
            "robot_id_0based",
            "assigned_pair_count",
            "workload_time_s",
        ],
        rows=per_robot_rows,
    )

    if args.report_json is not None:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_csv": str(Path(args.input_csv).resolve()),
            "summary": summary_rows,
            "per_robot": per_robot_rows,
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    write_time_comparison_plot(summary_rows, args.plot_path)

    print(f"Saved summary CSV: {Path(args.summary_csv)}")
    print(f"Saved per-robot CSV: {Path(args.per_robot_csv)}")
    print(f"Saved time comparison plot: {Path(args.plot_path)}")
    if args.report_json is not None:
        print(f"Saved JSON report: {Path(args.report_json)}")


if __name__ == "__main__":
    main()
