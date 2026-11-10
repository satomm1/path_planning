"""Compare two captures from ``capture.py`` (or a capture against the committed baseline).

Usage:

    python tests/regression/compare.py OUTDIR                 # vs tests/regression/baseline
    python tests/regression/compare.py BASELINE_DIR OUTDIR    # any two captures
    python tests/regression/compare.py --accept OUTDIR        # make OUTDIR the new baseline

Exit status is 0 when every item matches, 1 otherwise. Differences in the recorded
environment (package versions, solvers, kernel availability, wall-distance cache) are
reported first, because they explain many output differences.
"""

import json
import shutil
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "baseline"
FILES = ("_manifest.json", "_environment.json")


def load(directory, name):
    path = Path(directory) / name
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def main(argv):
    if argv[:1] == ["--accept"] and len(argv) == 2:
        BASELINE.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copyfile(Path(argv[1]) / name, BASELINE / name)
        print(f"baseline updated from {argv[1]}")
        return 0
    if len(argv) == 1:
        base_dir, new_dir = BASELINE, Path(argv[0])
    elif len(argv) == 2:
        base_dir, new_dir = Path(argv[0]), Path(argv[1])
    else:
        print(__doc__)
        return 2

    base, new = load(base_dir, "_manifest.json"), load(new_dir, "_manifest.json")
    if base is None or new is None:
        print("missing _manifest.json in", base_dir if base is None else new_dir)
        return 2

    base_env = load(base_dir, "_environment.json") or {}
    new_env = load(new_dir, "_environment.json") or {}
    env_diff = sorted(k for k in set(base_env) | set(new_env) if base_env.get(k) != new_env.get(k))
    for key in env_diff:
        print(f"ENV {key}: {base_env.get(key)!r} -> {new_env.get(key)!r}")

    keys = sorted(set(base) | set(new))
    bad = [k for k in keys if base.get(k) != new.get(k)]
    for k in bad:
        status = "DIFF" if k in base and k in new else ("MISSING" if k in base else "NEW")
        print(status, k)
    print(f"{len(base)} baseline items, {len(new)} new, {len(bad)} differences"
          + (f" ({len(env_diff)} environment differences)" if env_diff else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
