"""Verification report JSON for MAPF visual runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from social_path_planning.mapf_comparison.motion import MotionConfig


def write_verification_report(
    output_path: Path,
    *,
    scenario: str,
    num_agents: int,
    benchmark_seed: int,
    motion: MotionConfig,
    results: Dict[str, dict],
    milp_astar_mode: Optional[str] = None,
) -> dict:
    report: Dict[str, Any] = {
        "scenario": scenario,
        "num_agents": num_agents,
        "benchmark_seed": benchmark_seed,
        "milp_astar_mode": milp_astar_mode,
        "motion": motion.to_manifest_dict(),
        "priority_order": results.get("pp", {}).get("priority_order"),
        "methods": {},
    }
    for method in ("milp", "event_milp", "cbs", "pp"):
        if method not in results:
            continue
        m = results[method]["metrics"]
        report["methods"][method] = m
        rt = m.get("solver_runtime_s", m.get("runtime_s"))
        rt_s = f"{float(rt):.3f}s" if rt is not None else "n/a"
        extra = (
            f" SOC_steps={m.get('soc_timesteps')} conflicts={m.get('conflict_count')} "
            f"static_valid={m.get('static_valid')}"
            if method not in ("milp", "event_milp")
            else ""
        )
        event_extra = ""
        if method == "event_milp":
            total_rt = m.get("runtime_s")
            total_rt_s = f"{float(total_rt):.3f}s" if total_rt is not None else "n/a"
            event_extra = (
                f" encounters={m.get('encounter_count')} "
                f"interest_wps={m.get('interest_waypoint_count')} "
                f"z={m.get('binary_z_count')} "
                f"total_runtime_s={total_rt_s}"
            )
        label = method.upper().replace("_", " ")
        print(
            f"\n{label}: success={m['success']} status={m['status']} "
            f"SOC_s={m.get('soc_seconds')} makespan_s={m.get('makespan_seconds')} "
            f"path_len_m={m.get('total_path_length_m')} solver_runtime_s={rt_s}{extra}{event_extra}"
        )

    def _json_default(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"Wrote {output_path}")
    return report
