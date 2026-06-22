"""Path cache for the temporary-obstacle demo (avoid re-running slow social A*)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 2
DEFAULT_CACHE_FILENAME = "temp_paths_cache.json"
SOLVER_NAME = "AStar_With_Graph"


def _path_to_serializable(path: list[tuple[float, float]]) -> list[list[float]]:
    return [[float(p[0]), float(p[1])] for p in path]


def _path_from_serializable(raw: list) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in raw]


def _file_sha256(path: str | None) -> str | None:
    if path is None or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_fingerprint(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def obstacle_specs_from_overlay(overlay) -> list[dict[str, Any]]:
    specs = []
    for obs in overlay.list_obstacles():
        spec = {"id": obs.id, "shape": obs.shape, "active": obs.active, **obs.params}
        specs.append(spec)
    return sorted(specs, key=lambda s: s.get("id", ""))


def build_cache_fingerprint(
    *,
    scenario_name: str,
    seed: int,
    obstacle_specs: list[dict[str, Any]],
    permanent_heatmap: np.ndarray,
    baseline_dir: str | None = None,
    baseline_graph_file: str | None = None,
    proximity_radius_m: float = 2.5,
    graph_threshold: float,
    min_component_size: int,
) -> str:
    baseline_checkpoint = None
    if baseline_dir is not None:
        baseline_checkpoint = os.path.join(baseline_dir, "heatmap_checkpoint.npy")
    perm_fp = None
    if np.any(permanent_heatmap):
        perm_fp = _array_fingerprint(permanent_heatmap)
    payload = {
        "version": CACHE_VERSION,
        "solver": SOLVER_NAME,
        "scenario_name": scenario_name,
        "seed": int(seed),
        "obstacle_specs": obstacle_specs,
        "permanent_heatmap_sha256": perm_fp,
        "baseline_checkpoint_sha256": _file_sha256(baseline_checkpoint),
        "baseline_graph_sha256": _file_sha256(baseline_graph_file),
        "proximity_radius_m": float(proximity_radius_m),
        "graph_threshold": float(graph_threshold),
        "min_component_size": int(min_component_size),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, DEFAULT_CACHE_FILENAME)


def load_path_cache(cache_path: str) -> dict[str, Any] | None:
    path = Path(cache_path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    return payload


def save_path_cache(cache_path: str, payload: dict[str, Any]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def validate_cache_fingerprint(payload: dict[str, Any], expected_fingerprint: str) -> bool:
    return payload.get("fingerprint") == expected_fingerprint


def records_from_cache(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("paths") or []
    parsed = []
    for rec in records:
        path = _path_from_serializable(rec["path"])
        if len(path) < 2:
            continue
        parsed.append(
            {
                "route_id": int(rec["route_id"]),
                "x_init": (float(rec["x_init"][0]), float(rec["x_init"][1])),
                "x_goal": (float(rec["x_goal"][0]), float(rec["x_goal"][1])),
                "path": path,
                "solve_time_s": rec.get("solve_time_s"),
                "obstacle_affected": rec.get("obstacle_affected"),
                "from_cache": True,
            }
        )
    parsed.sort(key=lambda r: r["route_id"])
    return parsed


def _jsonify_rng_state(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): _jsonify_rng_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_rng_state(v) for v in obj]
    return obj


def _dejsonify_rng_state(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.array(obj["__ndarray__"], dtype=obj["dtype"])
        return {k: _dejsonify_rng_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dejsonify_rng_state(v) for v in obj]
    return obj


def serialize_planning_rng(rng: np.random.Generator) -> dict[str, Any]:
    """Serialize numpy Generator state for JSON path-cache resume."""
    return _jsonify_rng_state(rng.bit_generator.state)


def restore_planning_rng(state_dict: dict[str, Any]) -> np.random.Generator:
    """Restore a numpy Generator from ``serialize_planning_rng`` output."""
    rng = np.random.default_rng()
    rng.bit_generator.state = _dejsonify_rng_state(state_dict)
    return rng


def make_cache_payload(
    *,
    fingerprint: str,
    scenario_name: str,
    seed: int,
    obstacle_specs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    rng_state: dict[str, Any] | None = None,
    planning_attempts: int = 0,
) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "solver": SOLVER_NAME,
        "fingerprint": fingerprint,
        "scenario_name": scenario_name,
        "seed": int(seed),
        "obstacle_specs": obstacle_specs,
        "planning_attempts": int(planning_attempts),
        "rng_state": rng_state,
        "paths": [
            {
                "route_id": int(rec["route_id"]),
                "x_init": [float(rec["x_init"][0]), float(rec["x_init"][1])],
                "x_goal": [float(rec["x_goal"][0]), float(rec["x_goal"][1])],
                "path": _path_to_serializable(rec["path"]),
                "solve_time_s": rec.get("solve_time_s"),
                "obstacle_affected": rec.get("obstacle_affected"),
            }
            for rec in records
        ],
    }
