# Plan: homotopy-class alternative paths (Bhattacharya-style class-aware A*)

Branch: `test/homotopy-alternatives` (worktree `.claude/worktrees/homotopy`, based on `noetic` @ 54bdc3f).

Goal: for one start/goal pair, return the K cheapest **topologically distinct** social paths
(one per homotopy class), using the existing social cost (Eq. 2–4) unchanged. Downstream
(later milestone, not in this plan): feed the K candidates to the event MICP for route choice.

Ground rules
- **New files only.** No edits to `a_star.py`, `social_cost.py`, `event_multi_planning.py`,
  defaults, seeds or solvers, so published numbers are untouched.
- Python 3.8 compatible (`from __future__ import annotations`, no `match`, no runtime `X | Y`).
- Imports as `social_path_planning.*`.

---

## 1. `homotopy.py`: obstacle model and signatures

### 1a. Obstacle extraction: `extract_obstacles(occ, *, min_area_m2=1.0, connectivity=4) -> ObstacleSet`
- Blocked mask = `not occ._free_lookup()`: the **configuration-space** mask the planner already uses
  (robot_d inflation, `probs > 0.85` or `< 0`). Gaps the robot can't pass merge two obstacles
  into one automatically, so we never invent infeasible classes.
- `scipy.ndimage.label` with **4-connectivity**: `get_neighbors` only checks the destination
  center, so the 8-connected planner *can* slip diagonally between two diagonally-touching blocked
  cells. 4-connected labelling keeps the topology consistent with the motion model.
- Drop components touching the map border (the "outside"; no representative needed).
- Drop components with area < `min_area_m2`. They become "transparent" and don't appear in
  signatures. This is the main lever against class explosion (chairs and clutter, especially on Y2E2 at 0.05 m).
- Representative point ζᵢ: the component's most interior cell (argmax of
  `distance_transform_edt` inside it).
- `ObstacleSet` dataclass: `labels`, `ids`, `reps` (world), `areas`, `rays`.

### 1b. Rays
- One axis-aligned ray per obstacle, from ζᵢ to the map border. The direction (±x/±y) is the one with
  the **least free-space length** (fewer spurious crossings).
- The ray line is offset to `k·res + res/2` (res = A* lattice resolution), so **no lattice point
  ever lies on a ray**. The crossing test for a lattice edge is then exact and unambiguous. The ray
  origin is inside the obstacle, so the ambiguity at the ray tip never occurs for free-space edges.
- Precompute, per lattice edge direction, a fast `crossings(x1, x2) -> tuple[signed id]`
  (usually empty). For axis rays this is a comparison, with no geometry library.

### 1c. Signatures
- **Homotopy word** (search key): append the signed id on each crossing and cancel adjacent inverses
  (`... a, -a` → `...`). Stored as an immutable `tuple[int, ...]`, which is hashable.
- **Winding vector** (filter + diagnostics): Σ signed `atan2` angle increments around each ζᵢ.
  It's determined by (cell, class), so store it with the state when it's first created.
- `path_word(path, obs)` / `path_winding(path, obs)` for whole polylines (used on smoothed paths
  and in tests).

## 2. Class-aware search: `HomotopyMixin`
`HomotopyAStar(HomotopyMixin, AStar)` and `HomotopyAStar_With_Graph(HomotopyMixin, AStar_With_Graph)`.

- The search state becomes `(xy, word)`. `closed_set`, `came_from`, `cost_to_arrive` and `dist_to_right` are
  keyed on that pair. `get_neighbors(xy)` and `cost(x1, x2, d2r_prev)` are **reused as-is**, so
  the social cost, ε=0.01 and graph weights are identical.
- **Mirror `AStar.modified_solve` exactly**, including its current semantics: the priority accumulates
  social edge cost, relaxation compares geometric `cost_to_arrive`, `dist2right_prev` is taken from
  the parent's parent, and h is Manhattan. This lets the degenerate case reproduce today's planner
  bit for bit (test 4). Ordering by social cost while relaxing on distance is a quirk of the existing
  code; we copy it rather than fix it here.
- `_compiled_inputs` already returns `None` for subclasses, so this runs the Python loop
  (about 100× slower than the numba kernel; see §5).
- **K-best in one search** (the Bhattacharya approach): don't stop at the first goal pop. Each time a
  goal state with a *new* word is popped, record it as a candidate. Stop when:
  - K classes have been found, or
  - the popped priority exceeds `(1 + alpha) × best`, or
  - the time limit is hit.
- Pruning (all configurable):
  - `max_abs_winding` (default 2π): reject a state that has looped fully around any obstacle.
  - `max_word_len` (default `2 × n_obstacles`).
  - `max_letter_repeats` (default 1): no repeated obstacle letter in the reduced word.
  - Once `best` is known, drop pushes with priority > `(1 + alpha) × best`.
- Optional targeting: `target_word=` (find the best path in one given class) and `blocked_words=`.
- Each result is a `HomotopyAlternative` dataclass holding:
  - `word`, `winding`, `path`, `smoothed_path`, `cost` (search priority)
  - `length_m`, `social_penalty`, `fraction_on_graph` (same definitions as the telemetry)

  Results are sorted by recomputed cost.
- Post-filter: `merge_overlap_threshold` uses `compare_path_overlap.spatial_overlap_metrics`.
  It drops a class that is topologically distinct but practically the same route (e.g. either side of a lone
  column in an open hall).

## 3. Scripts
- `demo_homotopy_alternatives.py` (argparse): `--scenario`, `--start/--goal` or `--seed`, `--k`,
  `--alpha`, `--min-area`, optional heatmap/graph. It plots the obstacles, ζᵢ, rays and the K paths,
  each colour labelled with its word, length and social penalty.
- `survey_homotopy_classes.py`: for each scenario (`sample1-5_default`, `circular_default`,
  `hall_with_room_default`, `y2e2_aligned`) it reports the obstacle count after filtering and, over N random
  pairs, the distribution of the number of classes within α. This answers "where does this matter?" and
  feeds the paper. It writes to `results/` (gitignored), supports `--resume`, and runs in the background.

## 4. Tests: `tests/test_homotopy.py`
1. **One square obstacle**, start left / goal right: K=2 gives two paths with different words
   (`(+a)` vs `()` or similar), both collision-free. A looping third class is excluded by the default filters.
2. **Two obstacles stacked vertically**: exactly 3 non-looping classes (above both, between, below both).
3. **Signature invariants**: two different paths on the same side have the same word; the smoothed path
   has the same word as the raw path; no lattice point lies on any ray.
4. **Equivalence regression**: with no obstacles retained (or K=1), the `HomotopyAStar` /
   `HomotopyAStar_With_Graph` first path equals `AStar` / `AStar_With_Graph` with `use_compiled=False`
   on `sample2_default`. This guarantees we really reuse the paper's cost.
5. **Obstacle extraction**: border components and small components are dropped; a narrow (< robot_d) gap
   merges two blocks into one component.

The full suite stays at the known state (71 pass, 2 skip, 1 known wall-cache failure).

## 5. Performance (measure before optimizing)
- State count ≈ cells × reachable words. The area filter, word limits and α-cutoff bound it.
- Measure on `sample2_default` and `circular_default` first.
- If the Python loop is too slow on Y2E2, there are two options, in order:
  - (a) Run the class-aware search on **H only**, plus short off-graph connectors (approach "c"), which shrinks the
    state space dramatically.
  - (b) A numba variant of `astar_kernel` with word ids interned in a Python-side table.

  Decide after the survey numbers.

## Status and findings (2026-11-10)
- **M1 done**: `homotopy.py`, `tests/test_homotopy.py` (16 tests).
  - Rays tie-break through a symbolic perturbation, so diagonal steps through ray
    intersections have a well-defined crossing order.
  - Obstacle counts (min_area 1 m²): sample1 2, sample2/3 4, sample4/5 5, circular 5,
    hall_with_room 8, y2e2_aligned 10. The counts are the same with no area filter.
- **M2 done**: `homotopy_astar.py` (`HomotopyAStar`, `HomotopyAStar_With_Graph`),
  `tests/test_homotopy_astar.py` (11 tests). It's identical to `AStar` / `AStar_With_Graph`
  (Python loop) when no obstacle is retained.
  Changes from §2:
  - **`max_letter_repeats` dropped.** Rays pass through other obstacles, so a simple path
    that weaves around B can cross A's ray twice (`a b -a`). The filter would prune real
    classes. Looping is filtered by winding instead; `max_word_len` (default 2n+1) bounds
    commutator-type words.
  - **Winding in closed form.** It's computed from (cell, word) as
    `phi(end) - phi(start) + 2*pi*net_crossings`, with no per-edge trig.
  - The cut-off is `|winding_i| < 2*pi`. That keeps exactly the two direct ways around
    each obstacle.
- **Finding: the base planner's quirk shows up per class.** The heap orders on social
  cost but relaxes on geometric distance, so a class's returned path isn't its social
  optimum. It can change with `alpha`, because pruning changes which relaxation wins. The
  paper's planner has the same property for its single path. Worth deciding before M5
  whether alternatives should use a "clean" variant (relax on the social g) that's
  clearly marked as separate from the published planner.
- **Timing (Python loop, no wall cache, sample2 500×500).** A corner-to-corner pair took
  about 32 s and 100k states for 4 classes. Short pairs take under 1 s.
- **Compiled search done (§5b)**: `homotopy_kernel.py` (numba). It's used automatically when
  `astar_kernel` is usable, under the same eligibility rules. Otherwise it falls back to the
  Python loop.
  - Results equal the Python loop exactly: classes, costs, paths, explored-state counts and
    telemetry. 13 equivalence tests cover this.
  - Example pairs are about 40× faster: circular 1.0–1.3 s, sample2 corner pair 1.4 s,
    hall_with_room 4.4 s (was 191 s, 489k states).
  - Needs the `path_planning` conda env. `C:\Python312` has no numba, so there both kernels
    silently use Python.
  - The only possible deviation is the winding filter at exactly ±2π, because `atan2` runs
    in compiled code.
- **M3 done**: `demo_homotopy_alternatives.py` (CLI) and `tests/test_demo_homotopy_alternatives.py`
  (4 smoke tests). The demo writes a PNG and a JSON (classes, winding, cost, length, social
  penalty, fraction on H, paths) to `results/homotopy_alternatives/`.
  - `--heatmap` switches to the graph-biased cost (Eq. 4). `grid2_heatmap.npy` is too sparse
    to be useful (H has 125 nodes). `outputs_temp_obstacles/baseline_permanent_heatmap.npy`
    gives 4518 nodes, puts 37–67% of each alternative on H, and changes the class ranking.
  - Y2E2 now runs: 10 obstacles, 4 classes in about 10 s.
- **Merge filter done** (§2 post-filter). It's opt-in through `merge_overlap` (off by
  default) and `merge_distance_m` (default 0.5 m = robot diameter).
  - Overlap = the length fraction of a candidate's segments with an endpoint within
    `merge_distance_m` of a kept path's vertices. That's the proximity rule of the Fig. 11 metric
    (`compare_path_overlap`), expressed as a fraction.
  - It runs **inside the search**. A class whose overlap with a cheaper kept class is at least
    the threshold goes to `merged_alternatives` (with `merged_into`) and doesn't count toward K,
    so the search continues to K distinct routes. Kept classes report `max_overlap`.
  - The compiled kernel implements the same decision (same sums, same order). There are 6 new
    compiled-equivalence cases and 4 unit tests.
  - Calibration: distinct classes overlap their cheaper neighbours by about 0.3–0.6, and near-duplicates by
    0.68–0.74, so 0.65 is a reasonable demo value (`--merge-overlap 0.65`).
    - hall_with_room: merges #2 (`8-`, 0.68). Only 3 distinct classes fit within α=0.5.
    - Y2E2: merges 3 near-duplicates and finds 2 farther distinct routes instead (17 s).
- **M4 done**: `survey_homotopy_classes.py` and `tests/test_survey_homotopy_classes.py` (4 tests).
  - First run: `results/homotopy_survey/modified_k8_a0.5_seed42/`. Settings: 30 random pairs per
    map (seed 42, ≥ 30 m apart), K=8, social cost, no merge, compiled engine. 240 pairs took about 10 min.
  - Mean classes within (1+α) × best:

    | map | obstacles | α=0.1 | α=0.3 | α=0.5 | ≥2 classes at 0.3 | search median / baseline median |
    |---|---|---|---|---|---|---|
    | sample1 | 2 | 1.2 | 1.4 | 1.5 | 30% | 0.29 s / 0.10 s |
    | sample2 | 4 | 1.2 | 1.5 | 2.1 | 30% | 0.61 / 0.27 |
    | sample3 | 4 | 1.2 | 1.3 | 1.8 | 31% | 0.59 / 0.21 |
    | sample4 | 5 | 1.3 | 1.7 | 2.2 | 47% | 0.94 / 0.31 |
    | sample5 | 5 | 1.3 | 1.7 | 2.2 | 43% | 0.79 / 0.27 |
    | circular | 5 | 1.3 | 1.6 | 1.9 | 39% | 0.26 / 0.12 |
    | hall_with_room | 8 | 1.2 | 2.1 | 3.7 | 47% | 2.0 / 0.38 |
    | y2e2_aligned | 10 | 2.2 | 3.7 | 4.4 | 73% | 4.5 / 1.06 |

  - Findings:
    - Alternatives within 30% of the best cost exist for 30–47% of pairs on the synthetic
      maps and 73% on the real building (Y2E2).
    - The median 2nd-best class costs 1.08–1.23 × the best.
    - The best class equals the baseline social A* path's class for 97–100% of pairs. So
      class 1 is effectively the paper's path, and alternatives are purely additive.
    - Overhead vs the baseline (both compiled) is a median 2.7–6.4× (p90 up to 10×). The
      worst case was Y2E2 at 27 s, with K=8 and α=0.5. Smaller K/α costs less.
    - Caveat: this run has no merge filter, so hall_with_room and Y2E2 counts include
      near-duplicates. Re-run with `--merge-overlap 0.65` to count distinct routes.
  - §5 decision: the H-only search isn't needed for now. The compiled full-grid search is
    fast enough on every map.
- **Graph-biased survey** (`--heatmap SCENARIO=PATH`). Output in
  `results/homotopy_survey/modified_k8_a0.5_seed42_graph/`: the same 30 pairs per map, with both
  the baseline (`AStar_With_Graph`) and the classes using the Eq. 4 cost.
  - Heatmaps that match their grid: sample2 uses `outputs_temp_obstacles/baseline_permanent_heatmap.npy`
    (H = 4518 nodes); `y2e2_aligned` uses `y2e2_routes_heatmap.npy` (H = 73325 nodes, no mass on
    walls). Not usable:
    - `y2e2_routes_heatmap_ros.npy` (other frame: 95% of its mass on walls);
    - `grid2_heatmap.npy` and `vector_heatmap.npy` (too sparse: 125 and 490 nodes);
    - the other maps (no heatmaps).
  - Aggregate class counts barely move (sample2 α=0.3: 1.47 → 1.57; Y2E2: 3.67 → 3.33). Pairs
    with ≥2 classes: 30% → 33% and 73% → 77%. The 2nd/best cost ratio is unchanged (1.16 / 1.08).
  - **But H changes *which* routes are preferred.** On Y2E2 the best class changed in 11 of 30
    pairs, and the set of classes within α=0.3 matched in only 9 of 30. On sample2: 5 and 8 of 30
    changed.
  - **Alternatives follow conventions about as well as the best path.** On Y2E2 a median 93% of
    each alternative lies on H, the same as the best class and the baseline. sample2's sparser H
    gives about 55% for all three.
  - The best class still equals the baseline's class for 97–100% of pairs.
  - **Timing — corrected.** That first graph run seemed to show graph mode slower (e.g. the
    Y2E2 baseline 1.06 → 1.42 s). The cause was comparing two separate runs while machine load
    changed: even the plain baseline ran 1.5× slower then than later.
    - The survey now times plain social A* on each pair in the same run when a heatmap is
      given (`baseline_no_graph`, `graph_speedup`). Output in `..._graph_paired/`, with
      `--wall-cache none` as before.
    - Paired results:
      - **Y2E2** (H = 73k nodes): graph-biased social A* is **1.43× faster** (median; p90
        1.96×), faster on 87% of pairs. It explores 0.53× the nodes.
      - **sample2** (sparse H, 4.5k nodes): 1.00×, explores 0.91× the nodes.
      - The homotopy search costs 5.4× / 2.9× the graph-biased baseline.
      - Class counts match the first graph run exactly, as they should: the search is
        deterministic.
    - Standalone probes agree. On Y2E2, with H social-cost evaluations halve; it's 0.74× the
      time compiled and 0.64× in Python, and with the wall cache 0.76× / 0.70×. On sample2
      it's 0.92× in Python but 1.09× compiled: there are few evaluations to save, so the
      fixed overhead dominates.
  - **Possible issue in the Fig. 8 benchmark (existing code, not changed here).**
    `benchmark_sparse_snapshots.py` times both the A* baseline and the graph snapshots with
    `solve(mode="vanilla")`. Vanilla mode uses plain distance and never calls `cost()`, and
    `AStar_With_Graph` doesn't override it. So the snapshot runs ignore the graph: the probe
    gave identical paths and explored-node counts on every pair for both maps. The plotted
    "Social Graph Speedup vs A*" is therefore vanilla A* against itself (timing noise). This
    has been the case since the script was added (1799169, 2026-03-16). If Fig. 8 came from
    this script, its speedup needs re-measuring with `mode="modified"`. The paired numbers
    above suggest a real speedup does exist for social A* on a rich H.

## Milestones
1. `homotopy.py` §1, plus tests 3 and 5.
2. `HomotopyMixin` §2, plus tests 1, 2 and 4.
3. Demo script and figures on `circular_default` / `hall_with_room_default`.
4. Survey script and timing numbers; decide on §5.
5. (Next plan) Level-1 route choice: per-candidate `EventMultiAgentSequentialPlanner`, then the joint
   route-choice MICP.

## Open choices (defaults picked; easy to change)
- `min_area_m2=1.0`, `alpha=0.3`, `K=4`, `max_abs_winding=2π`.
- Word-keyed search (exact homotopy) rather than winding-keyed (homology). Winding is kept as a filter.
