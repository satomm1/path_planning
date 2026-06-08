# path_planning

ROS 1 Noetic **catkin** package (Python 3) providing occupancy-grid-based path planning (A*, benchmarks, and related tools).

## Layout

- [`package.xml`](package.xml), [`CMakeLists.txt`](CMakeLists.txt), [`setup.py`](setup.py) — catkin + setuptools at the **repository root**.
- Python sources live under [`src/path_planning/path_planning/`](src/path_planning/path_planning/) as the importable package `path_planning`.
- Scenario data: [`src/path_planning/path_planning/environments/`](src/path_planning/path_planning/environments/).
- Generated benchmark artifacts (JSON, CSV, GIF, and related plots) live under [`results/`](results/) so the repository root stays small:
  - `results/distributed_constraint_timing/` — outputs from `benchmark_sparse_snapshots.py` in distributed-constraints mode (pair timings, manifest, path bank).
  - `results/distributed_eval/` — CSV/JSON/plot from `evaluate_distributed_constraint_timing.py`.
  - `results/mapf_comparison/` — MILP vs CBS vs path-length prioritized planning (`benchmark_mapf_comparison.py`).
  - `results/mapf_ensemble/` — multi-trial averaged MAPF metrics (`benchmark_mapf_ensemble.py`).
  - `results/media/` — example animation GIFs.

## Build (catkin)

Use a normal catkin workspace and put this repository under `src` (for example clone or symlink as `~/catkin_ws/src/path_planning`).

```bash
cd ~/catkin_ws
catkin_make   # or: catkin build
source devel/setup.bash
```

`setup.py` uses `catkin_pkg`; that is provided in a ROS Noetic environment. After sourcing `devel/setup.bash`, Python should resolve `import path_planning`.

## Optional Python dependencies

Some scripts use **cvxpy** (for example multi-agent / optimization demos). It is not always available via `rosdep`; install as needed, for example:

```bash
pip3 install cvxpy
```

**MAPF baselines (CBS + prioritized planning)** for comparing grid MAPF against the MILP timing coordinator:

```bash
pip3 install -r requirements-mapf.txt
export PYTHONPATH=/path/to/path_planning/src:$PYTHONPATH
python3 -m social_path_planning.benchmark_mapf_comparison --scenario sample2_default --benchmark-seed 42 --stage-sizes 2,4,8,16 --max-velocity 0.7
```

Comparison code lives in `social_path_planning/mapf_comparison/`:

- **`grid_traversability.TraversabilityModel`** — single source of truth for static passability (`ROBOT_DIAMETER_M = 0.5`, default coarse policy `fine_center` aligned with A* cell centers).
- **`pipeline`** — `prepare_mapf_problem` + `run_mapf_solvers` / `run_milp_solver` (shared by `benchmark_mapf_comparison` and `visualize_mapf`).
- **`metrics`** — unified SOC/makespan/path length plus **static** (obstacle) and **dynamic** (agent–agent) validation on scheduled paths.
- **`viz/`** — route overlays, animations, verification JSON.

CBS/PP and MILP each use **native path geometry**; travel times are comparable because all methods share the same **max velocity** (default **0.7 m/s**, overridable with `--max-velocity`). CBS/PP low-level STA* is **8-connected** with **octile** edge costs (10 per cardinal step, 14 per diagonal) so diagonal shortcuts are costlier than straight corridor travel. Path-bank A* is also 8-connected on the fine grid.

| Metric | Meaning |
|--------|---------|
| `solver_runtime_s` | Wall-clock solve time |
| `soc_seconds` | Sum of per-agent completion times |
| `makespan_seconds` | Latest agent completion time |
| `total_path_length_m` | Sum of executed route lengths (m) |
| `soc_timesteps` | MAPF-only: discrete STA* steps |
| `static_valid` | MAPF: no waypoint in occupied space (via `TraversabilityModel`) |
| `dynamic_valid` | MAPF: no agent–agent conflicts at scheduled timesteps |

**Multi-trial ensemble benchmark** (reportable averages over many random agent subsets):

```bash
export PYTHONPATH=/path/to/path_planning/src:$PYTHONPATH
python3 -m social_path_planning.benchmark_mapf_ensemble \
  --scenario sample2_default \
  --pool-size 32 --num-agents 4 --num-trials 50 \
  --output-prefix results/mapf_ensemble/sample2_n4_t50
```

1. **Pool prep (once, cached):** generates `pool_size` random start/goal routes and caches modified A* polylines in `{prefix}_path_pool.json`. Use `--pregen-only` to build the pool without running trials.
2. **Each trial:** samples `num_agents` routes without replacement; CBS, PP, MILP SOC, and MILP makespan all use the same endpoints. MILP uses cached polylines (no per-trial social A*).
3. **Outputs:** `{prefix}_trials.csv` (per trial × method), `{prefix}_summary.csv` (aggregated), `{prefix}_manifest.json`.

| Summary field | Meaning |
|---------------|---------|
| `success_rate` / `failure_count` | Fraction and count of trials with no valid solution |
| `avg_solver_runtime_s` | Mean planning time over **successful** trials only |
| `avg_makespan_seconds` | Mean task makespan over **successful** trials only |
| `pool_astar_build_s` (manifest) | One-time social/modified A* cost during pool prep; **excluded** from MILP trial averages |

Outputs go under `results/mapf_comparison/` (detailed/summary CSV, manifest JSON, metrics bar chart, example map overlay). Re-plot from a prior run:

```bash
python3 -m social_path_planning.plot_mapf_comparison --summary-csv results/mapf_comparison/sample2_mapf_summary.csv
```

**Visual verification** of CBS / PP (GIF animations, snapshot grid, route overlay):

```bash
python3 -m social_path_planning.visualize_mapf --scenario sample2_default --num-agents 4 --output-dir results/mapf_comparison/verify --max-velocity 0.7
```

Use `--mapf-downsample 2` (or `1`) for finer CBS/PP geometry; keep `--pp-low-level-max-iter 0` and `--cbs-max-iter 0` so limits **auto-scale** with downsample (finer grids need more STA* expansions and wider crop padding). Example: `--mapf-downsample 2 --crop-padding 40`.

Coarse obstacle merge (when `downsample` > 1): default `--mapf-coarse-block-policy fine_center` matches vanilla A* passability at fine cell centers; use `any` for the strictest (corner-based) merge.

Outputs: `milp_animation.gif`, `cbs_animation.gif`, `pp_animation.gif`, `methods_side_by_side.gif`, `routes_overlay.png`, `snapshots_grid.png`, `verification_report.json`. MILP defaults to **modified (social) A\*** geometry at the same starts/goals as the path bank; dashed lines in the overlay are vanilla bank routes. CBS/PP replan on the grid. Use `--milp-astar-mode vanilla` for bank polylines; `--heatmap-prefix` for social A* on a saved graph; `--no-milp` if cvxpy is not installed.

Quick sanity check on an open grid (no occupancy file needed):

```bash
python3 -m social_path_planning.visualize_mapf --demo-open --num-agents 3 --output-dir results/mapf_comparison/verify_demo
```

## Tests

From the repository root, point `PYTHONPATH` at the parent of the `path_planning` package:

```bash
export PYTHONPATH=/path/to/path_planning/src/path_planning:$PYTHONPATH
python3 -m unittest discover -s tests -v
```

## Demos

Example entry point (was previously `path_planning.py` at repo root):

```bash
export PYTHONPATH=/path/to/path_planning/src/path_planning:$PYTHONPATH
python3 -m path_planning.demo_path_planning
```

### Y2E2 social A* (random routes)

The importable package in this repository is `social_path_planning` under `src/`. After building and sourcing your catkin workspace (or by setting `PYTHONPATH` to the `src` directory), run random start/goal tests on the Y2E2 map:

```bash
export PYTHONPATH=/path/to/path_planning/src:$PYTHONPATH
python3 -m social_path_planning.test_y2e2_social_astar -n 5
```

- **`-n` / `--num-paths`**: how many successful **social** (modified) A* paths to collect.
- **Plotting is on by default** (one figure with the map and all paths). Use **`--no-plot`** for headless or CI runs.
- Optional: `--seed`, `--max-attempts`, `--min-separation` (minimum start–goal distance in meters).

For a **vanilla vs modified** comparison on the same map (only pairs where **both** solvers succeed), with optional JSON/CSV output, use:

```bash
python3 -m social_path_planning.compare_astar --scenario y2e2 --num-routes 10
```

To compare vanilla A* against **social A* on a saved heatmap graph** (loads `{prefix}_heatmap.npy`, same as `plan_with_heatmap`):

```bash
python3 -m social_path_planning.compare_astar --scenario y2e2 --num-routes 10 \
    --heatmap-prefix y2e2_routes --output results/y2e2_compare_heatmap.json
```

**Wall-distance cache (faster social A\*):** one-time offline precompute writes `src/social_path_planning/environments/<scenario>_wall_dist.npz`. If that file is present and matches the loaded occupancy grid, planning uses table lookups instead of ray marching. Install **`tqdm`** (`pip install tqdm`) for a row-wise progress bar during precompute; use **`--no-progress`** to disable it.

```bash
python3 -m social_path_planning.precompute_wall_distances --scenario y2e2
```

**Directional heatmap (accumulated paths):** merge many social A* routes into an HSV direction heatmap; for Y2E2, cap start–goal distance so pairs stay local (same idea as `test_y2e2_social_astar`):

```bash
python3 -m social_path_planning.heat_map --scenario y2e2 --num-paths 20 --heatmap-prefix y2e2_routes --max-start-goal-distance 30
```

Use `--no-plot` for headless runs; `--resume` loads `<prefix>_heatmap.npy` before adding paths. Precompute wall distances (command above) first on large maps.
