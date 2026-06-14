"""Incremental CSV checkpointing for ensemble MAPF benchmarks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence, Set, Tuple

from social_path_planning.mapf_comparison.ensemble import METHODS

CompletedKey = Tuple[int, int]


def load_trial_rows(trials_csv: Path) -> list[dict]:
    if not trials_csv.is_file():
        return []
    with trials_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_completed_keys(
    trials_csv: Path,
    *,
    methods: Sequence[str] = METHODS,
    require_all_methods: bool = True,
) -> Set[CompletedKey]:
    """Return ``(num_agents, trial_id)`` pairs considered complete."""
    rows = load_trial_rows(trials_csv)
    if not rows:
        return set()

    by_key: dict[CompletedKey, set[str]] = {}
    for row in rows:
        key = (int(row["num_agents"]), int(row["trial_id"]))
        by_key.setdefault(key, set()).add(str(row.get("method", "")))

    completed: Set[CompletedKey] = set()
    expected = set(methods)
    for key, seen in by_key.items():
        if require_all_methods:
            if expected.issubset(seen):
                completed.add(key)
        else:
            if seen:
                completed.add(key)
    return completed


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
