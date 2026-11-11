"""
Class-aware social A*: the K cheapest paths in distinct homotopy classes.

The approach follows Bhattacharya, Likhachev & Kumar (2012): the search state is
``(lattice state, reduced word)`` (see ``homotopy.py``). One search returns several
classes: it does not stop at the first goal, and every goal state popped with a new word
becomes a candidate.

The search loop mirrors ``AStar.modified_solve`` / ``vanilla_solve`` line for line, so the
social cost of Eq. 2-4 (``cost``), the neighbour model (``get_neighbors``) and the heap
ordering are exactly the paper's. This includes two quirks of the existing planner, kept on
purpose:

* the heap priority accumulates the social edge cost, while relaxation compares the
  geometric ``cost_to_arrive``;
* ``dist2right_prev`` is the stored value of the parent's parent, and ``h`` is the
  Manhattan distance (not admissible on the 8-connected lattice).

With no obstacles retained every word is empty, and the first path found equals today's
planner (see ``tests/test_homotopy_astar.py``).

Because of the inadmissible heuristic, the ``alpha`` cut-off ("classes up to (1+alpha) x
the best cost") is a search bound rather than a guarantee. It's the same kind of guarantee the base
planner gives for its single path.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable, List, Optional

import numpy as np

from social_path_planning import astar_kernel
from social_path_planning.a_star import AStar, AStar_With_Graph
from social_path_planning.homotopy import ObstacleSet, Word, extend_word, extract_obstacles


@dataclass
class HomotopyAlternative:
    word: Word
    winding: np.ndarray  # rad around each obstacle, ordered like ObstacleSet.ids
    path: list  # A* lattice states (world coords)
    smoothed_path: list
    cost: float  # search cost at the goal pop: sum of edge costs (social, or distance in vanilla)
    length_m: float
    social_penalty: float  # sum of rightness_penalty along the path (telemetry definition)
    fraction_on_graph: Optional[float]  # None without a social graph
    # Largest overlap with a cheaper kept alternative (0 for the first), see OverlapMerger.
    max_overlap: float = 0.0
    # For a merged class: index in ``alternatives`` of the kept class it duplicates.
    merged_into: Optional[int] = None


class OverlapMerger:
    """Practical-duplicate test between alternatives (opt-in ``merge_overlap``).

    Same proximity rule as the paper's overlap metric (``compare_path_overlap``): a
    segment of the candidate overlaps a kept path when one of its endpoints is within
    ``distance_m`` of one of the kept path's vertices. The overlap is the length fraction of
    such segments:

        overlap(A, P) = sum(len(a_i a_i+1) for overlapping segments) / length(A)

    A candidate whose overlap with some kept alternative is ``>= threshold`` is a
    duplicate of the one it overlaps most. The test runs in cell-index space (integer
    offsets within ``distance_m / res`` cells), and the compiled kernel evaluates the same
    sums in the same order, so both engines make identical decisions.
    """

    def __init__(self, width, height, threshold, distance_m, resolution):
        self.width = int(width)
        self.height = int(height)
        self.threshold = float(threshold)
        self.r2 = (float(distance_m) / float(resolution)) ** 2
        reach = int(math.floor(math.sqrt(self.r2)))
        self.offsets = np.array(
            [(dc, dr) for dc in range(-reach, reach + 1) for dr in range(-reach, reach + 1)
             if dc * dc + dr * dr <= self.r2],
            dtype=np.int64,
        ).reshape(-1, 2)
        self.masks = []  # one [row, col] bool mask per kept alternative

    def overlaps(self, cells):
        return [overlap_fraction(cells, mask) for mask in self.masks]

    def keep(self, cells):
        mask = np.zeros((self.height, self.width), dtype=bool)
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        for dc, dr in self.offsets:
            c = cells[:, 0] + dc
            r = cells[:, 1] + dr
            ok = (c >= 0) & (c < self.width) & (r >= 0) & (r < self.height)
            mask[r[ok], c[ok]] = True
        self.masks.append(mask)

    def decide(self, cells):
        """(merged_into or None, max overlap) for a candidate, keeping it if not merged."""
        values = self.overlaps(cells)
        best = max(values) if values else 0.0
        if values and best >= self.threshold:
            return values.index(best), best
        self.keep(cells)
        return None, best


def overlap_fraction(cells, mask):
    """Length fraction of the polyline ``cells`` whose segments touch ``mask`` (see OverlapMerger)."""
    height, width = mask.shape
    total = 0.0
    covered = 0.0

    def inside(c, r):
        return 0 <= c < width and 0 <= r < height and bool(mask[r, c])

    prev_in = inside(*cells[0]) if len(cells) else False
    for (c1, r1), (c2, r2) in zip(cells[:-1], cells[1:]):
        dc = c2 - c1
        dr = r2 - r1
        seg = math.sqrt(dc * dc + dr * dr)
        cur_in = inside(c2, r2)
        total += seg
        if prev_in or cur_in:
            covered += seg
        prev_in = cur_in
    if total == 0.0:
        return 1.0
    return covered / total


class HomotopyMixin:
    """Mix in before ``AStar`` / ``AStar_With_Graph``; see module docstring."""

    def __init__(
        self,
        *args,
        obstacles: Optional[ObstacleSet] = None,
        min_area_m2: float = 1.0,
        k: int = 4,
        alpha: float = 0.3,
        max_abs_winding: Optional[float] = 2 * math.pi,
        max_word_len: Optional[int] = None,
        target_word: Optional[Word] = None,
        blocked_words: Iterable[Word] = (),
        time_limit_s: Optional[float] = None,
        merge_overlap: Optional[float] = None,
        merge_distance_m: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if obstacles is None:
            obstacles = extract_obstacles(
                self.occupancy,
                min_area_m2=min_area_m2,
                statespace_lo=self.statespace_lo,
                statespace_hi=self.statespace_hi,
                astar_resolution=self.resolution,
            )
        self.obstacles = obstacles
        self.k = int(k)
        self.alpha = float(alpha)
        self.max_abs_winding = max_abs_winding
        self.max_word_len = (2 * obstacles.num_obstacles + 1) if max_word_len is None else int(max_word_len)
        self.target_word = None if target_word is None else tuple(target_word)
        self.blocked_words = {tuple(w) for w in blocked_words}
        self.time_limit_s = time_limit_s
        # merge_overlap: None = keep every class (default); else merge practical duplicates
        # (OverlapMerger). merge_distance_m defaults to the robot diameter, as in Fig. 11.
        self.merge_overlap = None if merge_overlap is None else float(merge_overlap)
        self.merge_distance_m = float(merge_distance_m)
        self.alternatives: List[HomotopyAlternative] = []
        self.merged_alternatives: List[HomotopyAlternative] = []
        self.start_state = (self.x_init, ())
        self._start_cell = self.obstacles.state_index(self.x_init)
        # Search containers keyed on (xy, word), replacing the base class's xy-keyed ones.
        self.closed_set = set()
        self.came_from = {}
        self.cost_to_arrive = {self.start_state: 0}
        self.dist_to_right = {self.start_state: 0}
        self.priority_queue = [(self.manhattan_distance(self.x_init, self.x_goal), self.start_state)]
        self._angle_cache = {}
        self.explored_states = 0
        self.used_compiled = False

    # ------------------------------------------------------------------
    # Solve entry points (same return conventions as AStar.solve)
    # ------------------------------------------------------------------

    def modified_solve(self, return_timing=False, return_telemetry=False, log_telemetry=False):
        return self._homotopy_solve("modified", return_timing, return_telemetry, log_telemetry)

    def vanilla_solve(self, return_timing=False, return_telemetry=False, log_telemetry=False):
        return self._homotopy_solve("vanilla", return_timing, return_telemetry, log_telemetry)

    def solve_alternatives(self, mode="modified") -> List[HomotopyAlternative]:
        self.solve(mode=mode)
        return self.alternatives

    # ------------------------------------------------------------------
    # Class bookkeeping
    # ------------------------------------------------------------------

    def _step_word(self, x1, x2, word):
        c1, r1 = self.get_index(x1)
        c2, r2 = self.get_index(x2)
        for letter in self.obstacles.edge_crossings(c1, r1, c2, r2):
            word = extend_word(word, letter)
        return word

    def _winding(self, xy, word):
        cell = self.get_index(xy)
        angles = self._angle_cache.get(cell)
        if angles is None:
            angles = self.obstacles.ray_angles(*cell)
            self._angle_cache[cell] = angles
        start = self._angle_cache.get(self._start_cell)
        if start is None:
            start = self.obstacles.ray_angles(*self._start_cell)
            self._angle_cache[self._start_cell] = start
        return angles - start + 2 * math.pi * self.obstacles.abelianize(word)

    def _word_allowed(self, xy, word):
        if len(word) > self.max_word_len:
            return False
        if self.max_abs_winding is not None and self.obstacles.num_obstacles:
            if np.max(np.abs(self._winding(xy, word))) >= self.max_abs_winding:
                return False
        return True

    def _goal_word_accepted(self, word, found):
        if word in found or word in self.blocked_words:
            return False
        return self.target_word is None or word == self.target_word

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _homotopy_solve(self, mode, return_timing, return_telemetry, log_telemetry):
        social = mode == "modified"
        label = "Homotopy social A*" if social else "Homotopy vanilla A*"
        time_limit = self.time_limit_s
        if time_limit is None:
            time_limit = 750 if social else 150  # base-class limits
        self._begin_solve_telemetry(mode)
        t_start = time.time()
        result = None
        compiled = self._homotopy_compiled_inputs(mode)
        if compiled is not None:
            result = self._compiled_search(compiled, t_start, time_limit)
        self.used_compiled = result is not None
        if result is None:
            result = self._python_search(social, t_start, time_limit)
        # [(word, cost, path, merged_into, max_overlap)] in goal-pop order. Goal states have
        # h = 0, so pops come in non-decreasing cost order and kept indices are cost ranks.
        found, timed_out = result

        elapsed = time.time() - t_start
        self.last_solve_time = elapsed
        self.alternatives = []
        self.merged_alternatives = []
        for word, cost, path, merged_into, overlap in found:
            alt = self._make_alternative(word, cost, path)
            alt.max_overlap = float(overlap)
            alt.merged_into = merged_into
            (self.alternatives if merged_into is None else self.merged_alternatives).append(alt)
        success = bool(self.alternatives)
        if success:
            self.path = self.alternatives[0].path
            self.smoothed_path = self.alternatives[0].smoothed_path
            merged = f" ({len(self.merged_alternatives)} merged)" if self.merged_alternatives else ""
            print(f"{label} found {len(self.alternatives)} class(es){merged} in {elapsed:.2f} seconds.")
        elif timed_out:
            print("A* took too long.")
        return self._emit_solve_return(success, elapsed, mode, return_timing, return_telemetry, log_telemetry)

    def _finalize_solve_telemetry(self, success, mode, elapsed):
        super()._finalize_solve_telemetry(success, mode, elapsed)
        self.last_solve_telemetry["explored_nodes"] = int(self.explored_states)

    def _python_search(self, social, t_start, time_limit):
        """The reference loop (mirrors AStar.modified_solve / vanilla_solve)."""
        found = {}  # word -> (cost, goal state, merged_into, max_overlap), insertion = pop order
        n_kept = 0
        merger = self._make_merger()
        bound = math.inf
        timed_out = False

        while self.priority_queue:
            current_cost, s_current = heappop(self.priority_queue)
            if current_cost > bound:
                break
            if time.time() - t_start > time_limit:
                timed_out = True
                break
            if s_current in self.closed_set:
                continue
            self.closed_set.add(s_current)
            x_current, word = s_current

            if x_current == self.x_goal:
                # Goal states are never expanded: a path through the goal and back is not
                # a useful alternative.
                if self._goal_word_accepted(word, found):
                    merged_into, overlap = None, 0.0
                    if merger is not None:
                        cells = [self.get_index(x) for x in self._reconstruct_from(s_current)]
                        merged_into, overlap = merger.decide(cells)
                    found[word] = (current_cost, s_current, merged_into, overlap)
                    if merged_into is not None:
                        continue  # a practical duplicate: doesn't count toward k
                    n_kept += 1
                    if n_kept == 1:
                        bound = (1.0 + self.alpha) * current_cost
                    if (
                        n_kept >= self.k
                        or self.target_word is not None
                        or self.obstacles.num_obstacles == 0  # only one class exists
                    ):
                        break
                continue

            for x_neigh in self.get_neighbors(x_current):
                w_neigh = self._step_word(x_current, x_neigh, word)
                s_neigh = (x_neigh, w_neigh)
                if s_neigh in self.closed_set:
                    continue

                tentative_cost_to_arrive = self.cost_to_arrive[s_current] + self.distance(x_current, x_neigh)
                if social:
                    if s_current == self.start_state:
                        dist2right_prev = 0
                    else:
                        dist2right_prev = self.dist_to_right[self.came_from[s_current]]
                if s_neigh not in self.cost_to_arrive or tentative_cost_to_arrive < self.cost_to_arrive[s_neigh]:
                    if s_neigh not in self.cost_to_arrive and not self._word_allowed(x_neigh, w_neigh):
                        continue
                    if social:
                        cost_x_x_neigh, dist2right = self.cost(x_current, x_neigh, dist2right_prev)
                    else:
                        cost_x_x_neigh = self.distance(x_current, x_neigh)
                    priority = current_cost + cost_x_x_neigh + self.h(x_neigh) - self.h(x_current)
                    if priority > bound:
                        continue
                    if social:
                        self.dist_to_right[s_neigh] = dist2right
                    self.came_from[s_neigh] = s_current
                    self.cost_to_arrive[s_neigh] = tentative_cost_to_arrive
                    heappush(self.priority_queue, (priority, s_neigh))

        self.explored_states = len(self.closed_set)
        return [
            (word, cost, self._reconstruct_from(state), merged_into, overlap)
            for word, (cost, state, merged_into, overlap) in found.items()
        ], timed_out

    def _make_merger(self):
        if self.merge_overlap is None or self.target_word is not None:
            return None
        occ = self.occupancy
        return OverlapMerger(occ.width, occ.height, self.merge_overlap, self.merge_distance_m, occ.resolution)

    # ------------------------------------------------------------------
    # Compiled search (homotopy_kernel)
    # ------------------------------------------------------------------

    def _homotopy_compiled_inputs(self, mode):
        """Arguments for ``homotopy_kernel._solve``, or None to use the Python loop.

        Same eligibility rules as ``AStar._compiled_inputs``.
        """
        from social_path_planning import homotopy_kernel
        from social_path_planning.occupancy_grid import StochOccupancyGrid2D

        if not (self.use_compiled and homotopy_kernel.AVAILABLE):
            return None
        if type(self) not in (HomotopyAStar, HomotopyAStar_With_Graph):
            return None  # subclasses may override cost or neighbour methods
        occ = self.occupancy
        if type(occ) is not StochOccupancyGrid2D:
            return None
        if self.closed_set or self.came_from or len(self.priority_queue) != 1:
            return None  # only a fresh problem
        if not homotopy_kernel.kernel_ready():
            return None
        res = self.resolution
        idx = [astar_kernel.lattice_index(v, res) for v in (*self.x_init, *self.x_goal)]
        if None in idx:
            return None
        start_i, start_j, goal_i, goal_j = idx
        width, height = int(occ.width), int(occ.height)
        if np.shape(occ.probs) != (height, width):
            return None
        i0, j0, ni_size, nj_size = astar_kernel.lattice_bounds(occ, res)
        if not (i0 <= start_i < i0 + ni_size and j0 <= start_j < j0 + nj_size):
            return None

        graph_mode = mode == "modified" and isinstance(self, AStar_With_Graph)
        if mode == "vanilla":
            kmode = astar_kernel.MODE_VANILLA
        elif graph_mode:
            kmode = astar_kernel.MODE_GRAPH
        else:
            kmode = astar_kernel.MODE_MODIFIED
        if graph_mode:
            tables = astar_kernel.graph_tables(self.graph, width, height)
            if tables is None:
                return None
        else:
            tables = np.zeros((1, 1, 9), dtype=np.bool_), np.zeros((1, 1, 9))
        edge_has, edge_w = tables

        lo = tuple(float(v) for v in self.statespace_lo)
        hi = tuple(float(v) for v in self.statespace_hi)
        cache = occ.__dict__.setdefault("_astar_kernel_cache", {})
        key = (lo, hi, occ.robot_d, occ.resolution)
        entry = cache.get(key)
        if entry is None or entry[0] is not occ.probs:
            entry = (
                occ.probs,
                np.ascontiguousarray(occ.probs, dtype=np.float64),
                astar_kernel.center_ok_table(occ, self.statespace_lo, self.statespace_hi),
            )
            cache[key] = entry
        _, probs, center_ok = entry

        d_right = getattr(occ, "_d_right", None)
        use_cache = d_right is not None
        if use_cache:
            d_right = np.ascontiguousarray(d_right, dtype=np.float64)
        else:
            d_right = np.zeros((1, 1, 8))
        from social_path_planning.wall_distance_cache import _UNIT_TRAVEL_DIRS

        obs = self.obstacles
        obs_pos = np.zeros(max(obs.ids, default=0) + 1, dtype=np.int64)
        for pos, obstacle_id in enumerate(obs.ids):
            obs_pos[obstacle_id] = pos
        h_ptr, h_origin2, h_sign, h_id, v_ptr, v_origin2, v_sign, v_id = homotopy_kernel.ray_tables(
            obs, width, height
        )
        rays = [obs.rays[i] for i in obs.ids]
        ray_ox = np.array([ray.origin[0] for ray in rays], dtype=float)
        ray_oy = np.array([ray.origin[1] for ray in rays], dtype=float)
        ray_dx = np.array([float(ray.sign) if ray.axis == 0 else 0.0 for ray in rays], dtype=float)
        ray_dy = np.array([0.0 if ray.axis == 0 else float(ray.sign) for ray in rays], dtype=float)
        start_angles = np.asarray(obs.ray_angles(*self._start_cell), dtype=float)

        specials = [tuple(w) for w in sorted(self.blocked_words)]
        has_target = self.target_word is not None
        if has_target:
            specials.append(self.target_word)
        special_ptr = np.zeros(len(specials) + 1, dtype=np.int64)
        flat = []
        for q, w in enumerate(specials):
            flat.extend(int(v) for v in w)
            special_ptr[q + 1] = len(flat)
        if any(abs(v) not in obs.rays for v in flat):
            return None  # a word names an obstacle that isn't retained: leave it to Python

        use_winding = self.max_abs_winding is not None
        merger = self._make_merger()
        merge_offsets = merger.offsets if merger is not None else np.zeros((1, 2), dtype=np.int64)
        return (
            kmode, start_i, start_j, goal_i, goal_j, float(res), i0, j0, ni_size, nj_size,
            float(occ.origin_x), float(occ.origin_y), width, height, center_ok,
            float(occ.resolution), probs, float(occ.thresh), use_cache, d_right,
            np.ascontiguousarray(_UNIT_TRAVEL_DIRS, dtype=np.float64),
            float(self.robot_d), float(self.desired_dist_right_extra),
            edge_has, edge_w,
            obs.num_obstacles, obs_pos, h_ptr, h_origin2, h_sign, h_id, v_ptr, v_origin2, v_sign, v_id,
            ray_ox, ray_oy, ray_dx, ray_dy, start_angles,
            use_winding, float(self.max_abs_winding) if use_winding else 0.0,
            int(self.max_word_len), int(self.k), float(self.alpha),
            np.array(flat, dtype=np.int64), special_ptr, len(self.blocked_words), has_target,
            merger is not None, float(merger.threshold) if merger is not None else 0.0,
            np.ascontiguousarray(merge_offsets, dtype=np.int64),
        )

    def _compiled_search(self, inputs, t_start, time_limit):
        from social_path_planning import homotopy_kernel

        (status, found_word, found_cost, found_state, found_merged, found_overlap, n_found,
         s_a, s_b, s_parent, n_states, n_closed, w_parent, w_last, n_words, tel) = homotopy_kernel._solve(
            *inputs, float(t_start), float(time_limit)
        )
        if status == homotopy_kernel.STATUS_ABORT:
            return None
        res = self.resolution
        i0, j0 = inputs[6], inputs[7]

        def word_of(w):
            letters = []
            while w != 0:
                letters.append(int(w_last[w]))
                w = int(w_parent[w])
            return tuple(reversed(letters))

        def path_of(state):
            cells = []
            while state >= 0:
                cells.append((res * (int(s_a[state]) + i0), res * (int(s_b[state]) + j0)))
                state = int(s_parent[state])
            return list(reversed(cells))

        found = [
            (
                word_of(int(found_word[q])),
                float(found_cost[q]),
                path_of(int(found_state[q])),
                None if found_merged[q] < 0 else int(found_merged[q]),
                float(found_overlap[q]),
            )
            for q in range(n_found)
        ]
        self.explored_states = int(n_closed)
        self.priority_queue = []
        t = self.last_solve_telemetry
        t["cost_eval_count"] = int(tel[0])
        t["social_cost_sum"] = float(tel[1])
        t["social_cost_max"] = float(tel[2])
        t["social_nonzero_edges"] = int(tel[3])
        t["graph_edge_cost_evals"] = int(tel[4])
        t["off_graph_social_evals"] = int(tel[5])
        t["graph_weight_sum"] = float(tel[6])
        t["off_graph_edge_cost_sum"] = float(tel[7])
        return found, status == homotopy_kernel.STATUS_TIMED_OUT

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _reconstruct_from(self, goal_state):
        states = [goal_state]
        current = goal_state
        while current != self.start_state:
            current = self.came_from[current]
            states.append(current)
        return [xy for xy, _ in reversed(states)]

    def _make_alternative(self, word, cost, path):
        self.path = path
        self.smooth_path()
        length = 0.0
        penalty = 0.0
        prev_d2r = 0.0
        for x1, x2 in zip(path[:-1], path[1:]):
            length += float(self.distance(x1, x2))
            pen, d2r = self.rightness_penalty(x1, x2, prev_d2r)
            penalty += float(pen)
            prev_d2r = float(d2r)
        fraction = None
        if isinstance(self, AStar_With_Graph):
            metrics = {}
            AStar_With_Graph._append_graph_path_metrics(self, metrics)
            fraction = metrics["path_fraction_on_graph"]
        return HomotopyAlternative(
            word=word,
            winding=self._winding(self.x_goal, word),
            path=path,
            smoothed_path=self.smoothed_path,
            cost=float(cost),
            length_m=length,
            social_penalty=penalty,
            fraction_on_graph=fraction,
        )


class HomotopyAStar(HomotopyMixin, AStar):
    """Class-aware social A* (Eq. 2-3 cost)."""


class HomotopyAStar_With_Graph(HomotopyMixin, AStar_With_Graph):
    """Class-aware graph-biased social A* (Eq. 4 cost)."""
