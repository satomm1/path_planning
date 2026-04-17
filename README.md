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
