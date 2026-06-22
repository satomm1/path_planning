"""Incremental CSV checkpointing for ensemble MAPF benchmarks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Set, Tuple

from social_path_planning.mapf_comparison.constants import METHODS

CompletedKey = Tuple[int, int]
TrialRowKey = Tuple[int, int, str]


def coerce_bool(value) -> bool:
    """Parse booleans from CSV strings, JSON, or native bools."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no", ""):
        return False
    return bool(value)


def normalize_trial_row(row: Mapping[str, object]) -> dict:
    """Coerce typed fields after reading from CSV."""
    out = dict(row)
    if "success" in out:
        out["success"] = coerce_bool(out.get("success"))
    for key in ("trial_id", "trial_seed", "num_agents", "conflict_count", "binary_z_count"):
        if key in out and out[key] not in (None, ""):
            out[key] = int(out[key])
    for key in (
        "solver_runtime_s",
        "runtime_s",
        "soc_seconds",
        "makespan_seconds",
        "soc_timesteps",
        "makespan_timesteps",
        "total_path_length_m",
    ):
        if key in out and out[key] not in (None, ""):
            out[key] = float(out[key])
    return out


def trial_row_key(row: Mapping[str, object]) -> TrialRowKey:
    return (int(row["num_agents"]), int(row["trial_id"]), str(row["method"]))


def load_trial_rows(trials_csv: Path, *, normalize: bool = True) -> list[dict]:
    if not trials_csv.is_file():
        return []
    with trials_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if normalize:
        return [normalize_trial_row(row) for row in rows]
    return rows


def dedupe_trial_rows(rows: Sequence[dict]) -> list[dict]:
    """Keep the last row for each (num_agents, trial_id, method)."""
    latest: dict[TrialRowKey, dict] = {}
    for row in rows:
        latest[trial_row_key(row)] = dict(row)
    return list(latest.values())


def load_completed_keys(
    trials_csv: Path,
    *,
    methods: Sequence[str] = METHODS,
    require_all_methods: bool = True,
    require_success: bool = False,
) -> Set[CompletedKey]:
    """
    Return ``(num_agents, trial_id)`` pairs considered complete.

    When ``require_success`` is True, every expected method must have a row with
    ``success=True``. Otherwise only row presence is checked.
    """
    rows = dedupe_trial_rows(load_trial_rows(trials_csv))
    if not rows:
        return set()

    by_key: dict[CompletedKey, dict[str, dict]] = {}
    for row in rows:
        key = (int(row["num_agents"]), int(row["trial_id"]))
        by_key.setdefault(key, {})[str(row.get("method", ""))] = row

    completed: Set[CompletedKey] = set()
    expected = set(methods)
    for key, method_rows in by_key.items():
        seen = set(method_rows)
        if require_all_methods:
            if not expected.issubset(seen):
                continue
            if require_success:
                if all(coerce_bool(method_rows[m].get("success")) for m in expected):
                    completed.add(key)
            else:
                completed.add(key)
        elif seen:
            if require_success:
                if all(coerce_bool(method_rows[m].get("success")) for m in seen):
                    completed.add(key)
            else:
                completed.add(key)
    return completed


def load_methods_to_retry(
    trials_csv: Path,
    *,
    methods: Sequence[str] = METHODS,
) -> dict[CompletedKey, Set[str]]:
    """
    Map each trial key to method names that are missing or not successful.

    Used by ``--retry-failures`` to rerun only CBS/PP/event rows that failed.
    """
    rows = dedupe_trial_rows(load_trial_rows(trials_csv))
    by_key: dict[CompletedKey, dict[str, dict]] = {}
    for row in rows:
        key = (int(row["num_agents"]), int(row["trial_id"]))
        by_key.setdefault(key, {})[str(row["method"])] = row

    retry: dict[CompletedKey, Set[str]] = {}
    expected = set(methods)
    for key, method_rows in by_key.items():
        need = {
            method
            for method in expected
            if method not in method_rows
            or not coerce_bool(method_rows[method].get("success"))
        }
        if need:
            retry[key] = need
    return retry


def append_trial_rows(
    trials_csv: Path,
    rows: Sequence[dict],
    fieldnames: Sequence[str],
) -> None:
    if not rows:
        return
    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not trials_csv.is_file() or trials_csv.stat().st_size == 0
    with trials_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compact_trial_rows(
    trials_csv: Path,
    fieldnames: Sequence[str],
    *,
    extra_rows: Sequence[dict] | None = None,
) -> list[dict]:
    """Merge on-disk rows with optional new rows, dedupe, and rewrite the CSV."""
    merged = dedupe_trial_rows(load_trial_rows(trials_csv) + list(extra_rows or []))
    write_csv(trials_csv, merged, fieldnames)
    return merged
