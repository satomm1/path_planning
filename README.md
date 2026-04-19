# path_planning

ROS 1 Noetic **catkin** package (Python 3) providing occupancy-grid-based path planning (A*, benchmarks, and related tools).

## Layout

- [`package.xml`](package.xml), [`CMakeLists.txt`](CMakeLists.txt), [`setup.py`](setup.py) — catkin + setuptools at the **repository root**.
- Python sources live under [`src/path_planning/path_planning/`](src/path_planning/path_planning/) as the importable package `path_planning`.
- Scenario data: [`src/path_planning/path_planning/environments/`](src/path_planning/path_planning/environments/).
- Generated benchmark artifacts (JSON, CSV, GIF, and related plots) live under [`results/`](results/) so the repository root stays small:
  - `results/distributed_constraint_timing/` — outputs from `benchmark_sparse_snapshots.py` in distributed-constraints mode (pair timings, manifest, path bank).
  - `results/distributed_eval/` — CSV/JSON/plot from `evaluate_distributed_constraint_timing.py`.
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
