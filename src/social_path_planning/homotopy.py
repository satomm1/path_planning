"""
Homotopy-class bookkeeping for alternative social paths.

Obstacle model, rays, and path signatures used by the class-aware A* (Bhattacharya,
Likhachev & Kumar, "Topological constraints in search-based robot path planning", 2012).

Everything is done in **cell-index space** of the occupancy grid: an A* lattice state ``x``
is checked for passability at cell ``(col, row) = round((x - origin) / res)``
(``AStar.get_neighbors``). That is the same table ``astar_kernel.center_ok_table`` builds,
so the obstacles here are exactly what the planner treats as blocked, including the
robot-diameter inflation and the state-space bounds.

Topology on the 8-connected lattice
-----------------------------------
A lattice edge moves by at most one cell per axis and only its destination is checked.
Two blocked cells that touch only diagonally can therefore be squeezed between, while
4-adjacent blocked cells cannot. Obstacles are therefore **4-connected** components of the
blocked mask. Components touching the grid border are part of the "outside" and get no
signature letter.

Rays
----
Each kept obstacle gets one axis-aligned ray from a representative point inside it to the
map border. Horizontal rays lie on a half-integer row line ``y = r* + 1/2`` starting at
``x = c*``; vertical rays lie on ``x = c* + 1/2`` starting at ``y = r*``. Lattice points
are integer, so no lattice point lies on a ray, and a lattice edge can't cross a ray's
line exactly at its origin (that would need the blocked cell ``(c*, r*)`` as an
endpoint). Crossings of lattice edges are therefore exact.

Signatures
----------
* ``word``: the signed obstacle ids of ray crossings in order, with adjacent inverse pairs
  cancelled. This is a reduced word in the free group on the obstacles, so two paths with
  the same endpoints are homotopic iff their words are equal. Sign ``+i`` means the path
  crossed ray ``i`` counter-clockwise around obstacle ``i``.
* ``winding``: the total signed angle swept around each representative point. This is the
  homology (H-signature) view, used as a looping filter and as a consistency check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from social_path_planning import astar_kernel

Word = Tuple[int, ...]

# Ray directions: (axis, sign). axis 0 = ray runs along +/-x (horizontal line in y),
# axis 1 = ray runs along +/-y (vertical line in x).
RAY_DIRECTIONS = ((0, 1), (0, -1), (1, 1), (1, -1))


@dataclass(frozen=True)
class Ray:
    """Axis-aligned ray in cell-index space.

    ``line2`` is twice the coordinate of the ray's line (always odd, i.e. a half-integer
    line) and ``origin2`` is twice the origin coordinate along the ray's axis (always even).
    """

    obstacle_id: int
    axis: int  # 0: horizontal ray (line y = line2/2), 1: vertical ray (line x = line2/2)
    sign: int  # +1: ray runs toward increasing coordinate, -1: decreasing
    line2: int
    origin2: int

    @property
    def origin(self) -> Tuple[float, float]:
        """Ray origin (col, row) in cell-index space; also the winding reference point."""
        if self.axis == 0:
            return (self.origin2 / 2.0, self.line2 / 2.0)
        return (self.line2 / 2.0, self.origin2 / 2.0)


@dataclass
class ObstacleSet:
    """Obstacles retained for signatures on one occupancy grid."""

    labels: np.ndarray  # [row, col] component label per blocked cell (0 = free)
    ids: List[int]  # retained obstacle ids (1..n, contiguous)
    component_of_id: Dict[int, int]  # obstacle id -> component label in ``labels``
    rep_cells: Dict[int, Tuple[int, int]]  # obstacle id -> representative (col, row)
    areas_m2: Dict[int, float]
    rays: Dict[int, Ray]
    resolution: float
    origin_x: float
    origin_y: float
    # Lookup of rays by the doubled line coordinate, for O(1) lattice-edge checks.
    _horizontal_by_line: Dict[int, List[Ray]] = field(default_factory=dict, repr=False)
    _vertical_by_line: Dict[int, List[Ray]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._horizontal_by_line = {}
        self._vertical_by_line = {}
        for ray in self.rays.values():
            table = self._horizontal_by_line if ray.axis == 0 else self._vertical_by_line
            table.setdefault(ray.line2, []).append(ray)

    @property
    def num_obstacles(self) -> int:
        return len(self.ids)

    def ray_origins(self) -> np.ndarray:
        """(n, 2) array of ray origins (col, row), ordered like ``ids``."""
        return np.array([self.rays[i].origin for i in self.ids], dtype=float).reshape(-1, 2)

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------

    def to_index(self, xy) -> Tuple[float, float]:
        """World point -> continuous cell-index coordinates.

        A* lattice states ``k * res`` map to exactly the integer cell index that
        ``AStar.get_index`` returns, so polylines through lattice states (raw or smoothed)
        are measured in the same frame as the obstacles.
        """
        res = self.resolution
        return (
            float(xy[0]) / res - round(self.origin_x / res),
            float(xy[1]) / res - round(self.origin_y / res),
        )

    def to_world(self, cr) -> Tuple[float, float]:
        res = self.resolution
        return (
            (float(cr[0]) + round(self.origin_x / res)) * res,
            (float(cr[1]) + round(self.origin_y / res)) * res,
        )

    # ------------------------------------------------------------------
    # Crossings
    # ------------------------------------------------------------------

    # Tie-breaking. Lattice diagonals pass exactly through points where ray lines meet: a
    # horizontal and a vertical line at a cell corner, or several rays sharing one line.
    # To make the crossing order well defined we use symbolically perturbed rays (still
    # valid rays: origins stay inside their obstacles, no lattice point lies on them):
    #   horizontal ray i: y = line + i*eta
    #   vertical ray i:   x = line + xi + i*eta        with 0 < eta << xi infinitesimal.
    # Consequences, used by both crossing routines:
    #   * a point exactly on an unperturbed line is below / left of it;
    #   * at equal t, moving +x crosses horizontal lines before vertical ones, moving -x
    #     (or dx == 0) the reverse;
    #   * among rays on the same line, lower ids come first when moving in the increasing
    #     direction across the line.

    def edge_crossings(self, c1: int, r1: int, c2: int, r2: int) -> Tuple[int, ...]:
        """Signed obstacle ids crossed by the lattice edge (c1, r1) -> (c2, r2), in order.

        For the class-aware search: integer cell indices, a step of at most one cell per
        axis. Exact integer arithmetic; agrees with ``segment_crossings`` on the same edge.
        """
        horiz = []
        if r1 != r2:
            rays = self._horizontal_by_line.get(r1 + r2)  # odd: half-integer row line
            if rays:
                x2 = c1 + c2  # doubled crossing x (|dc| <= 1): never equals an even origin2
                up = r2 > r1
                horiz = sorted(
                    (ray for ray in rays if (x2 - ray.origin2) * ray.sign > 0),
                    key=lambda ray: ray.obstacle_id, reverse=not up,
                )
                horiz = [_horizontal_letter(ray, up) for ray in horiz]
        vert = []
        if c1 != c2:
            rays = self._vertical_by_line.get(c1 + c2)
            if rays:
                y2 = r1 + r2
                right = c2 > c1
                vert = sorted(
                    (ray for ray in rays if (y2 - ray.origin2) * ray.sign > 0),
                    key=lambda ray: ray.obstacle_id, reverse=not right,
                )
                vert = [_vertical_letter(ray, right) for ray in vert]
        if c2 > c1:
            return tuple(horiz + vert)
        return tuple(vert + horiz)

    def segment_crossings(self, p, q) -> List[Tuple[float, int]]:
        """(t, signed id) for every ray crossed by the segment p -> q (cell-index coords).

        For general segments (e.g. smoothed paths), using the perturbed-ray convention above.
        """
        px, py = float(p[0]), float(p[1])
        qx, qy = float(q[0]), float(q[1])
        moving_plus_x = qx > px
        hits = []
        for ray in self.rays.values():
            line = ray.line2 / 2.0
            origin = ray.origin2 / 2.0
            if ray.axis == 0:
                a, b, u0, u1 = py, qy, px, qx
            else:
                a, b, u0, u1 = px, qx, py, qy
            if (a > line) == (b > line):
                continue
            t = (line - a) / (b - a)
            u = u0 + t * (u1 - u0)
            if (u - origin) * ray.sign <= 0:
                continue
            increasing = b > a
            if ray.axis == 0:
                group = 0 if moving_plus_x else 1
                letter = _horizontal_letter(ray, increasing)
            else:
                group = 1 if moving_plus_x else 0
                letter = _vertical_letter(ray, increasing)
            id_rank = ray.obstacle_id if increasing else -ray.obstacle_id
            hits.append((t, group, id_rank, letter))
        hits.sort()
        return [(t, letter) for t, _, _, letter in hits]

    # ------------------------------------------------------------------
    # Signatures of whole polylines (world coordinates)
    # ------------------------------------------------------------------

    def path_word(self, path_world: Sequence[Sequence[float]], word: Word = ()) -> Word:
        pts = [self.to_index(p) for p in path_world]
        for p, q in zip(pts[:-1], pts[1:]):
            for _, letter in self.segment_crossings(p, q):
                word = extend_word(word, letter)
        return word

    def state_index(self, x) -> Tuple[int, int]:
        """Cell index of an A* lattice state; same as ``AStar.get_index``."""
        res = self.resolution
        return (
            int(round((x[0] - self.origin_x) / res)),
            int(round((x[1] - self.origin_y) / res)),
        )

    def lattice_path_word(self, states: Sequence[Sequence[float]], word: Word = ()) -> Word:
        """Word of an A* state sequence, using the exact per-edge test (what the search uses)."""
        idx = [self.state_index(x) for x in states]
        for (c1, r1), (c2, r2) in zip(idx[:-1], idx[1:]):
            for letter in self.edge_crossings(c1, r1, c2, r2):
                word = extend_word(word, letter)
        return word

    def abelianize(self, word: Word) -> np.ndarray:
        """Net signed crossing count per obstacle (ordered like ``ids``): the homology class."""
        pos = {obstacle_id: k for k, obstacle_id in enumerate(self.ids)}
        counts = np.zeros(len(self.ids), dtype=int)
        for letter in word:
            counts[pos[abs(letter)]] += 1 if letter > 0 else -1
        return counts

    def ray_angles(self, c: float, r: float) -> np.ndarray:
        """Angle in [0, 2*pi) of (c, r) around each ray origin, measured CCW from the ray.

        Ordered like ``ids``. Lattice points never lie on a ray, so the value is never 0
        for them, and the jump from 2*pi to 0 happens exactly at a ray crossing.
        """
        out = np.empty(len(self.ids))
        for k, obstacle_id in enumerate(self.ids):
            ray = self.rays[obstacle_id]
            ox, oy = ray.origin
            vx, vy = c - ox, r - oy
            dx, dy = (ray.sign, 0) if ray.axis == 0 else (0, ray.sign)
            out[k] = math.atan2(dx * vy - dy * vx, dx * vx + dy * vy) % (2 * math.pi)
        return out

    def winding_from_word(self, start_cell, cell, word: Word) -> np.ndarray:
        """Winding of any path from ``start_cell`` to ``cell`` with this word, in closed form.

        winding_i = phi_i(end) - phi_i(start) + 2*pi * (net CCW crossings of ray i).
        """
        return (
            self.ray_angles(*cell) - self.ray_angles(*start_cell)
            + 2 * math.pi * self.abelianize(word)
        )

    def path_winding(self, path_world: Sequence[Sequence[float]]) -> np.ndarray:
        """Signed angle (rad) swept around each obstacle's ray origin, ordered like ``ids``."""
        pts = np.array([self.to_index(p) for p in path_world], dtype=float).reshape(-1, 2)
        refs = self.ray_origins()
        if len(pts) < 2 or len(refs) == 0:
            return np.zeros(len(refs))
        return winding_increments(pts[:-1], pts[1:], refs).sum(axis=0)


def _horizontal_letter(ray: Ray, increasing_y: bool) -> int:
    # Ray along +x: crossing upward goes counter-clockwise around the origin.
    # Ray along -x: crossing downward does.
    ccw = increasing_y if ray.sign > 0 else not increasing_y
    return ray.obstacle_id if ccw else -ray.obstacle_id


def _vertical_letter(ray: Ray, increasing_x: bool) -> int:
    # Ray along +y: crossing leftward (decreasing x) is counter-clockwise.
    # Ray along -y: crossing rightward is.
    ccw = (not increasing_x) if ray.sign > 0 else increasing_x
    return ray.obstacle_id if ccw else -ray.obstacle_id


def extend_word(word: Word, letter: int) -> Word:
    """Append one signed letter to a reduced word, cancelling an adjacent inverse."""
    if word and word[-1] == -letter:
        return word[:-1]
    return word + (letter,)


def reduce_word(letters: Sequence[int]) -> Word:
    word: Word = ()
    for letter in letters:
        word = extend_word(word, int(letter))
    return word


def winding_increments(p: np.ndarray, q: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """(m, n) signed angle each of m segments p->q sweeps around each of n reference points."""
    a = p[:, None, :] - refs[None, :, :]
    b = q[:, None, :] - refs[None, :, :]
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    return np.arctan2(cross, dot)


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def blocked_mask(occupancy, statespace_lo=None, statespace_hi=None) -> np.ndarray:
    """[row, col] True where the A* planner treats the cell as impassable."""
    if statespace_lo is None:
        statespace_lo = (-math.inf, -math.inf)
    if statespace_hi is None:
        statespace_hi = (math.inf, math.inf)
    ok = astar_kernel.center_ok_table(occupancy, statespace_lo, statespace_hi)
    return ~ok


def extract_obstacles(
    occupancy,
    *,
    min_area_m2: float = 1.0,
    statespace_lo=None,
    statespace_hi=None,
    astar_resolution: Optional[float] = None,
) -> ObstacleSet:
    """Label the planner's blocked cells into obstacles and give each kept one a ray.

    ``min_area_m2``: components smaller than this get no letter (clutter). They are still
    impassable to the planner; they just don't split homotopy classes.
    """
    res = float(occupancy.resolution)
    if astar_resolution is not None and not math.isclose(float(astar_resolution), res):
        raise ValueError(
            "homotopy signatures assume the A* lattice resolution equals the grid "
            f"resolution (got {astar_resolution} vs {res})"
        )
    blocked = blocked_mask(occupancy, statespace_lo, statespace_hi)
    height, width = blocked.shape
    four = ndimage.generate_binary_structure(2, 1)
    labels, num = ndimage.label(blocked, structure=four)

    border = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ])).tolist())
    border.discard(0)

    cell_area = res * res
    counts = np.bincount(labels.ravel(), minlength=num + 1)
    interior = ndimage.distance_transform_edt(blocked)

    kept = [
        lab for lab in range(1, num + 1)
        if lab not in border and counts[lab] * cell_area >= min_area_m2
    ]
    # Stable ids: order components by their first cell in row-major order.
    if kept:
        first = ndimage.minimum(np.arange(labels.size).reshape(labels.shape), labels, kept)
        kept = [lab for _, lab in sorted(zip(np.atleast_1d(first).tolist(), kept))]

    ids, component_of_id, rep_cells, areas, rays = [], {}, {}, {}, {}
    if kept:
        reps = ndimage.maximum_position(interior, labels, kept)
    else:
        reps = []
    for obstacle_id, (lab, rep_rc) in enumerate(zip(kept, reps), start=1):
        row, col = int(rep_rc[0]), int(rep_rc[1])
        ids.append(obstacle_id)
        component_of_id[obstacle_id] = lab
        rep_cells[obstacle_id] = (col, row)
        areas[obstacle_id] = float(counts[lab] * cell_area)
        rays[obstacle_id] = _choose_ray(obstacle_id, col, row, blocked)

    return ObstacleSet(
        labels=labels,
        ids=ids,
        component_of_id=component_of_id,
        rep_cells=rep_cells,
        areas_m2=areas,
        rays=rays,
        resolution=res,
        origin_x=float(occupancy.origin_x),
        origin_y=float(occupancy.origin_y),
    )


def _choose_ray(obstacle_id: int, col: int, row: int, blocked: np.ndarray) -> Ray:
    """Pick the axis direction whose ray passes through the fewest free cells.

    Fewer free cells under the ray means fewer crossings in ordinary paths, so words stay
    short. Ties keep the order of ``RAY_DIRECTIONS``.
    """
    height, width = blocked.shape
    best = None
    for axis, sign in RAY_DIRECTIONS:
        if axis == 0:
            # Line y = row + 1/2 between rows `row` and `row + 1` (clipped at the top edge).
            rows = [row] + ([row + 1] if row + 1 < height else [])
            cells = blocked[rows, col + 1:] if sign > 0 else blocked[rows, :col]
        else:
            cols = [col] + ([col + 1] if col + 1 < width else [])
            cells = blocked[row + 1:, cols] if sign > 0 else blocked[:row, cols]
        free_count = int(np.count_nonzero(~cells))
        if best is None or free_count < best[0]:
            best = (free_count, axis, sign)
    _, axis, sign = best
    if axis == 0:
        return Ray(obstacle_id, 0, sign, line2=2 * row + 1, origin2=2 * col)
    return Ray(obstacle_id, 1, sign, line2=2 * col + 1, origin2=2 * row)
