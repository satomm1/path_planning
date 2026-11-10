# Regression (equivalence) check

Use this check when a change is meant to leave planner outputs unchanged (refactors, speed-ups). It also shows exactly which outputs a deliberate behavior change affects. It hashes about 220 deterministic outputs and compares them with `baseline/_manifest.json`:

- **Event-waypoint MICP**: analyses, problem matrices, printed formulations and solutions. Covers simultaneous, sequential and combined planning on synthetic cases built to hit every encounter branch, plus 2–6 robots taken from the route pool.
- **Dense MILP**: problem data, stopped before solving, and the z-assignment plots.
- **Grids and helpers**: occupancy grids from every `build_occ_grid` variant (`sample2_default`, `y2e2`), random free points, path lengths and route keys.
- **Heatmap and social graph** edges.
- **A\***: paths and telemetry for vanilla, social, and social-with-graph. Each runs through both the numba kernel and the Python loop; they must give identical results.
- **Plots and GIFs**: space-time plots, map context, event-waypoint panels.

## Usage

Run from the repository root. Use the `path_planning` conda env, which the baseline was captured in:

```bash
python tests/regression/capture.py /tmp/capture_out
python tests/regression/compare.py /tmp/capture_out
```

- `compare.py` exits 0 when nothing changed. Otherwise it lists `DIFF`/`MISSING`/`NEW` items. Each item's full text is in the capture directory, `<item>.txt`.
- To see what changed in one item, capture the previous commit too, for example from a `git worktree`. Then diff the two `<item>.txt` files, or run `compare.py OLD_DIR NEW_DIR`.
- `--quick` captures a smaller subset in about 30 s. Compare it only against another `--quick` capture.
- Through pytest it is opt-in, because it takes a few minutes: `SOCIAL_PP_REGRESSION=1 python -m pytest tests/regression -q`.

## Updating the baseline

When a change is meant to alter outputs (like the Big-M sign fix):

1. Confirm the listed differences are exactly the expected ones.
2. Accept the new capture with `python tests/regression/compare.py --accept /tmp/capture_out`.
3. Commit `baseline/` together with the change.

## Environment dependence

`_environment.json` records Python, numpy, scipy, cvxpy (and its solvers), matplotlib, networkx and numba versions. It also records whether the A\* kernel was active, whether the Y2E2 wall-distance cache was present, and a hash of the input data. `compare.py` prints environment differences first, because they explain many output differences:

- cvxpy and the installed solvers change MILP problem data and solutions. MOSEK is chosen first when installed.
- matplotlib and Pillow change PNG and GIF bytes.
- scipy changes `smoothed_path`.
- The Y2E2 items require the gitignored `src/social_path_planning/environments/y2e2_wall_dist.npz`. Build it with `python -m social_path_planning.precompute_wall_distances --scenario y2e2`.

A\* paths and telemetry do not depend on these versions.

## Inputs

`data/sample2_route_pool.json` holds the 71 multi-robot routes used. They are a subset of the `sample2_default` social A\* path pool, stored as lattice indices; world value = 0.2 × index, which is exact. All other inputs are generated deterministically by `capture.py`.
