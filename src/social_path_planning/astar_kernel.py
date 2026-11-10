"""Optional numba-compiled A* search loop for ``a_star.AStar`` / ``AStar_With_Graph``.

The kernel reproduces the pure-Python search in ``a_star.py`` bit for bit: the same
lattice, neighbor order, heap ordering, cost-update rule, social cost
(``social_cost.rightness_penalty``), wall-distance ray-cast / cache lookup, graph
biasing and telemetry sums. ``AStar`` falls back to the Python loop whenever the
kernel is unavailable or a problem is outside what it handles.

Disable with the environment variable ``SOCIAL_ASTAR_NUMBA=0``. Compiled code is
cached on disk (``cache=True``) next to this module.

Exactness relies on three platform behaviors that are checked once before first use
(:func:`kernel_ready`); if any check fails the kernel is disabled:

- ``np.linalg.norm`` of a 2-vector equals ``sqrt(x*x + y*y)`` (true for the numpy
  builds tested; a BLAS that fuses multiply-adds would differ),
- :func:`py_hypot` (a port of CPython's ``math.hypot``) matches ``math.hypot``,
- ``np.rint`` matches Python ``round`` (both round half to even).
"""

from __future__ import annotations

import math
import os
import time
import warnings
import weakref

import numpy as np

MODE_VANILLA = 0
MODE_MODIFIED = 1
MODE_GRAPH = 2

STATUS_EXHAUSTED = 0
STATUS_FOUND = 1
STATUS_TIMED_OUT = -1
STATUS_ABORT = -2  # problem outside the kernel's lattice bounds: caller uses the Python loop

TIME_CHECK_EVERY = 1024
NO_PARENT = -(2 ** 62)  # parent-index sentinel (lattice indices can be negative)
_LATTICE_MARGIN = 3

_DISABLED_VALUES = ("0", "false", "no", "off")

AVAILABLE = False
UNAVAILABLE_REASON = None
if os.environ.get("SOCIAL_ASTAR_NUMBA", "1").strip().lower() in _DISABLED_VALUES:
    UNAVAILABLE_REASON = "disabled by SOCIAL_ASTAR_NUMBA"
else:
    try:
        from numba import njit, objmode
    except ImportError:
        UNAVAILABLE_REASON = "numba is not installed"
    else:
        AVAILABLE = True


if AVAILABLE:
    import heapq

    @njit(cache=True)
    def _wall_time():
        with objmode(t="float64"):
            t = time.time()
        return t

    @njit(cache=True)
    def _two_prod(a, b):
        """Exact product ``a*b = hi + lo`` (Dekker/Veltkamp; same as ``fma(a, b, -hi)``)."""
        hi = a * b
        c = 134217729.0 * a
        a_hi = c - (c - a)
        a_lo = a - a_hi
        c = 134217729.0 * b
        b_hi = c - (c - b)
        b_lo = b - b_hi
        lo = ((a_hi * b_hi - hi) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
        return hi, lo

    @njit(cache=True)
    def _hypot_scaled(x, y, mx):
        # CPython Modules/mathmodule.c vector_norm() for n == 2.
        m, max_e = math.frexp(mx)
        scale = math.ldexp(1.0, -max_e)
        csum = 1.0
        frac1 = 0.0
        frac2 = 0.0
        for v in (x, y):
            v = v * scale
            p_hi, p_lo = _two_prod(v, v)
            s = csum + p_hi
            z = s - csum
            s_lo = p_hi - z
            csum = s
            frac1 += p_lo
            frac2 += s_lo
        h = math.sqrt(csum - 1.0 + (frac1 + frac2))
        p_hi, p_lo = _two_prod(-h, h)
        s = csum + p_hi
        z = s - csum
        s_lo = p_hi - z
        csum = s
        frac1 += p_lo
        frac2 += s_lo
        v = csum - 1.0 + (frac1 + frac2)
        h += v / (2.0 * h)
        return h / scale

    @njit(cache=True)
    def py_hypot(x, y):
        """Bit-exact port of CPython's two-argument ``math.hypot``."""
        x = abs(x)
        y = abs(y)
        mx = 0.0
        if x > mx:
            mx = x
        if y > mx:
            mx = y
        if math.isinf(mx):
            return mx
        if math.isnan(x) or math.isnan(y):
            return math.nan
        if mx == 0.0:
            return mx
        m, max_e = math.frexp(mx)
        if max_e < -1023:
            dbl_min = 2.2250738585072014e-308
            return dbl_min * _hypot_scaled(x / dbl_min, y / dbl_min, mx / dbl_min)
        return _hypot_scaled(x, y, mx)

    @njit(cache=True)
    def norm2(x, y):
        """``np.linalg.norm([x, y])`` (verified at first use by :func:`kernel_ready`)."""
        return math.sqrt(x * x + y * y)

    @njit(cache=True)
    def rint_int(v):
        return int(np.rint(v))

    @njit(cache=True)
    def _raycast_right(px, py, td0, td1, dist_thresh, ox, oy, grid_res, width, height, probs, thresh):
        # StochOccupancyGrid2D._dist_to_wall_right_raycast(x=(px, py), travel_dir=(td0, td1)).
        r0 = td1
        r1 = -td0
        rn = norm2(r0, r1)
        if rn < 1e-12:
            return 100.0
        r0 = r0 / rn
        r1 = r1 / rn
        x_min = ox
        x_max = ox + width * grid_res
        y_min = oy
        y_max = oy + height * grid_res
        step = grid_res * 0.25
        if step < 1e-4:
            step = 1e-4
        t_max = 0.0
        if r0 > 0:
            t_max = max(t_max, (x_max - px) / r0)
        elif r0 < 0:
            t_max = max(t_max, (x_min - px) / r0)
        if r1 > 0:
            t_max = max(t_max, (y_max - py) / r1)
        elif r1 < 0:
            t_max = max(t_max, (y_min - py) / r1)
        if t_max <= 0:
            return 100.0
        if t_max > dist_thresh:
            t_max = dist_thresh
        n_steps = int(math.ceil(t_max / step))
        for i in range(1, n_steps + 1):
            t = i * step
            sx = px + r0 * t
            sy = py + r1 * t
            if sx < x_min or sx >= x_max or sy < y_min or sy >= y_max:
                return 100.0
            col = math.floor((sx - ox) / grid_res)
            row = math.floor((sy - oy) / grid_res)
            if row < 0 or row >= height or col < 0 or col >= width:
                return 100.0
            p = probs[int(row), int(col)]
            if p >= thresh or p < 0:
                return t
        return 100.0

    @njit(cache=True)
    def _travel_dir_index(td0, td1, unit_dirs):
        # wall_distance_cache.travel_dir_to_dir_idx
        n = norm2(td0, td1)
        if n < 1e-9:
            return 0
        t0 = td0 / n
        t1 = td1 / n
        best = 0
        best_dot = unit_dirs[0, 0] * t0 + unit_dirs[0, 1] * t1
        for k in range(1, unit_dirs.shape[0]):
            d = unit_dirs[k, 0] * t0 + unit_dirs[k, 1] * t1
            if d > best_dot:
                best_dot = d
                best = k
        return best

    @njit(cache=True)
    def _dist_to_wall_right(px, py, td0, td1, ox, oy, grid_res, width, height, probs, thresh,
                            use_cache, d_right, unit_dirs):
        # StochOccupancyGrid2D.dist_to_wall_right with the default dist_thresh (15.0).
        if use_cache:
            col = rint_int((px - ox) / grid_res)
            row = rint_int((py - oy) / grid_res)
            if col < 0:
                col = 0
            elif col > width - 1:
                col = width - 1
            if row < 0:
                row = 0
            elif row > height - 1:
                row = height - 1
            k = _travel_dir_index(td0, td1, unit_dirs)
            return d_right[row, col, k]
        return _raycast_right(px, py, td0, td1, 15.0, ox, oy, grid_res, width, height, probs, thresh)

    @njit(cache=True)
    def _rightness_penalty(x1x, x1y, x2x, x2y, gx, gy, sx0, sy0, d2r_prev,
                           robot_d, extra, astar_res,
                           ox, oy, grid_res, width, height, probs, thresh,
                           use_cache, d_right, unit_dirs):
        # social_cost.rightness_penalty(occupancy, x_init, x_goal, x1, x2, dist2right_prev, ...)
        if norm2(x2x - gx, x2y - gy) < 2:
            return 0.0, 0.0
        if norm2(x2x - sx0, x2y - sy0) < 2:
            return 0.0, 0.0
        penalty = 0.0
        d0 = x2x - x1x
        d1 = x2y - x1y
        n = norm2(d0, d1)
        td0 = d0 / n
        td1 = d1 / n
        dist_to_right = _dist_to_wall_right(x2x, x2y, td0, td1, ox, oy, grid_res, width, height,
                                            probs, thresh, use_cache, d_right, unit_dirs)
        if dist_to_right > 10:
            penalty += astar_res
            dist_to_left = _dist_to_wall_right(x2x, x2y, -td0, -td1, ox, oy, grid_res, width, height,
                                               probs, thresh, use_cache, d_right, unit_dirs)
            if dist_to_left > 3:
                dist_to_left_prev = _dist_to_wall_right(x1x, x1y, -td0, -td1, ox, oy, grid_res, width,
                                                        height, probs, thresh, use_cache, d_right, unit_dirs)
                delta = dist_to_left - dist_to_left_prev
                if delta < 0 or delta > 10:
                    delta = 0.0
                penalty += 5 * delta
            else:
                v = 4 - dist_to_left
                penalty = v if v > 0 else 0.0
        else:
            delta_raw = dist_to_right - d2r_prev
            desired = robot_d / 2 + extra
            penalty = abs(dist_to_right - desired)
            delta_far = delta_raw
            if delta_far > 15 or delta_far < 0:
                delta_far = 0.0
            if dist_to_right > desired:
                penalty += 2 * delta_far
            elif dist_to_right < desired and -15 < delta_raw < 0:
                penalty += 2 * (-delta_raw)
        return penalty, dist_to_right

    @njit(cache=True)
    def _solve(mode, start_i, start_j, goal_i, goal_j, astar_res, i0, j0, ni_size, nj_size,
               ox, oy, width, height, center_ok,
               grid_res, probs, thresh, use_cache, d_right, unit_dirs, robot_d, extra,
               edge_has, edge_w, t_start, time_limit):
        """
        Run A*. Returns ``(status, closed, cost, parent_i, parent_j, d2r, telemetry)``.

        Lattice state ``(i, j)`` is the point ``(astar_res * i, astar_res * j)`` and is
        stored at array index ``(i - i0, j - j0)``. ``telemetry`` holds, in order:
        cost_eval_count, social_cost_sum, social_cost_max, social_nonzero_edges,
        graph_edge_cost_evals, off_graph_social_evals, graph_weight_sum,
        off_graph_edge_cost_sum.
        """
        closed = np.zeros((ni_size, nj_size), dtype=np.bool_)
        cost = np.full((ni_size, nj_size), np.inf)
        parent_i = np.full((ni_size, nj_size), NO_PARENT, dtype=np.int64)
        parent_j = np.full((ni_size, nj_size), NO_PARENT, dtype=np.int64)
        d2r = np.zeros((ni_size, nj_size))
        tel = np.zeros(8)

        gx = astar_res * goal_i
        gy = astar_res * goal_j
        sx0 = astar_res * start_i
        sy0 = astar_res * start_j
        cost[start_i - i0, start_j - j0] = 0.0
        h0 = abs(sx0 - gx) + abs(sy0 - gy)
        # Heap entries (priority, x, y, i, j) all float64: same ordering as Python's
        # (priority, (x, y)); i, j are exact small integers.
        heap = [(h0, sx0, sy0, float(start_i), float(start_j))]
        pops = 0
        status = STATUS_EXHAUSTED
        offsets = (1, 0, -1)

        while len(heap) > 0:
            prio, cx, cy, fi, fj = heapq.heappop(heap)
            ci = int(fi)
            cj = int(fj)
            if ci == goal_i and cj == goal_j:
                status = STATUS_FOUND
                break
            pops += 1
            if pops % TIME_CHECK_EVERY == 0:
                if _wall_time() - t_start > time_limit:
                    status = STATUS_TIMED_OUT
                    break
            a = ci - i0
            b = cj - j0
            if closed[a, b]:
                continue
            closed[a, b] = True
            cost_c = cost[a, b]
            h_c = abs(cx - gx) + abs(cy - gy)
            if ci == start_i and cj == start_j:
                d2r_prev = 0.0
            else:
                d2r_prev = d2r[parent_i[a, b] - i0, parent_j[a, b] - j0]

            for ii in offsets:
                for jj in offsets:
                    if ii == 0 and jj == 0:
                        continue
                    x0 = cx + ii * astar_res
                    x1 = cy + jj * astar_res
                    col = rint_int((x0 - ox) / astar_res)
                    row = rint_int((x1 - oy) / astar_res)
                    if col < 0 or row < 0 or col >= width or row >= height:
                        continue
                    if not center_ok[row, col]:
                        continue
                    ni = rint_int(x0 / astar_res)
                    nj = rint_int(x1 / astar_res)
                    na = ni - i0
                    nb = nj - j0
                    if na < 0 or nb < 0 or na >= ni_size or nb >= nj_size:
                        return STATUS_ABORT, closed, cost, parent_i, parent_j, d2r, tel
                    if closed[na, nb]:
                        continue
                    nx = astar_res * ni
                    ny = astar_res * nj
                    dist = py_hypot(cx - nx, cy - ny)
                    tentative = cost_c + dist
                    if not (tentative < cost[na, nb]):
                        continue

                    if mode == MODE_VANILLA:
                        edge = dist
                    elif mode == MODE_MODIFIED:
                        social, nd2r = _rightness_penalty(
                            cx, cy, nx, ny, gx, gy, sx0, sy0, d2r_prev, robot_d, extra, astar_res,
                            ox, oy, grid_res, width, height, probs, thresh, use_cache, d_right, unit_dirs)
                        edge = dist + social
                        d2r[na, nb] = nd2r
                        tel[0] += 1
                        s = edge - dist
                        tel[1] += s
                        if s > tel[2]:
                            tel[2] = s
                        if s > 1e-9:
                            tel[3] += 1
                    else:
                        c1x = rint_int((cx - ox) / astar_res)
                        c1y = rint_int((cy - oy) / astar_res)
                        c2x = rint_int((nx - ox) / astar_res)
                        c2y = rint_int((ny - oy) / astar_res)
                        ddx = c2x - c1x
                        ddy = c2y - c1y
                        on_graph = False
                        w = 0.0
                        if (0 <= c1x < width and 0 <= c1y < height
                                and -1 <= ddx <= 1 and -1 <= ddy <= 1):
                            k = (ddx + 1) * 3 + (ddy + 1)
                            if edge_has[c1x, c1y, k]:
                                on_graph = True
                                w = edge_w[c1x, c1y, k]
                        if on_graph:
                            edge = w
                            nd2r = 0.0
                            tel[0] += 1
                            tel[4] += 1
                            tel[6] += w
                            s = w - dist
                        else:
                            social, nd2r = _rightness_penalty(
                                cx, cy, nx, ny, gx, gy, sx0, sy0, d2r_prev, robot_d, extra, astar_res,
                                ox, oy, grid_res, width, height, probs, thresh, use_cache, d_right,
                                unit_dirs)
                            edge = dist + social + 0.01
                            tel[0] += 1
                            tel[5] += 1
                            tel[7] += edge
                            s = edge - dist
                        d2r[na, nb] = nd2r
                        tel[1] += s
                        if s > tel[2]:
                            tel[2] = s
                        if s > 1e-9:
                            tel[3] += 1

                    parent_i[na, nb] = ci
                    parent_j[na, nb] = cj
                    cost[na, nb] = tentative
                    h_n = abs(nx - gx) + abs(ny - gy)
                    heapq.heappush(heap, (prio + edge + h_n - h_c, nx, ny, float(ni), float(nj)))

        return status, closed, cost, parent_i, parent_j, d2r, tel


# ---------------------------------------------------------------------------
# Python-side preparation and checks
# ---------------------------------------------------------------------------

_READY = None  # None = not checked yet, True/False after kernel_ready()


def _self_check():
    """Return None if kernel float behavior matches this platform, else a reason string."""
    rng = np.random.default_rng(12345)
    res = 0.2
    k1 = rng.integers(-3000, 3000, 4000)
    k2 = k1 + rng.integers(-1, 2, 4000)
    lattice = (res * k2 - res * k1).tolist()
    random = rng.uniform(-60, 60, 4000).tolist()
    values = lattice + random + [0.0, -0.0, 1e-300, 5e-324, 1.5, 2.5, -2.5, 1e300]
    pairs = list(zip(values, values[len(values) // 2:] + values[: len(values) // 2]))
    with np.errstate(over="ignore"):
        for x, y in pairs:
            if float(np.linalg.norm(np.array([x, y], dtype=float))) != norm2(x, y):
                return "np.linalg.norm differs from sqrt(x*x + y*y) on this platform"
            if math.hypot(x, y) != py_hypot(x, y):
                return "math.hypot differs from the kernel port on this platform"
    for v in [0.5, 1.5, 2.5, -0.5, -1.5, 3.4999999999999996, 1e15 + 0.5] + random[:200]:
        if round(v) != rint_int(v):
            return "np.rint differs from Python round"
    return None


def kernel_ready():
    """Compile/load the kernel and run the platform checks once; True if usable."""
    global _READY, UNAVAILABLE_REASON
    if _READY is not None:
        return _READY
    if not AVAILABLE:
        _READY = False
        return False
    try:
        reason = _self_check()
        if reason is None:
            _warm_up()
    except Exception as exc:  # pragma: no cover - defensive
        reason = f"kernel failed to compile or run: {type(exc).__name__}: {exc}"
    if reason is not None:
        UNAVAILABLE_REASON = reason
        warnings.warn(f"social_path_planning.astar_kernel disabled: {reason}", RuntimeWarning)
        _READY = False
    else:
        _READY = True
    return _READY


def _warm_up():
    """Run each mode once on a tiny grid so compilation/cache loading happens outside timing."""
    probs = np.zeros((6, 6))
    center_ok = np.ones((6, 6), dtype=np.bool_)
    unit_dirs = np.zeros((8, 2))
    d_right = np.zeros((1, 1, 8))
    edge_has = np.zeros((6, 6, 9), dtype=np.bool_)
    edge_w = np.zeros((6, 6, 9))
    for mode in (MODE_VANILLA, MODE_MODIFIED, MODE_GRAPH):
        _solve(mode, 0, 0, 5, 5, 1.0, -3, -3, 12, 12, 0.0, 0.0, 6, 6, center_ok,
               1.0, probs, 0.5, False, d_right, unit_dirs, 0.4, 0.25,
               edge_has, edge_w, time.time(), 1e9)


def lattice_index(value, res):
    """Integer ``k`` with ``res * k == value`` exactly, or None if ``value`` is off-lattice."""
    k = round(value / res)
    if res * k != value:
        return None
    return int(k)


def center_ok_table(occupancy, statespace_lo, statespace_hi):
    """
    ``ok[row, col]`` = ``AStar.is_free(fine_cell_center_world(occupancy, col, row))``.

    Vectorized with the same floating-point operations as the scalar calls; cells
    whose snapped index falls off the free lookup use the scalar window check.
    """
    width, height = occupancy.width, occupancy.height
    res = float(occupancy.resolution)
    cols = np.arange(width)
    rows = np.arange(height)
    cx = occupancy.origin_x + (cols + 0.5) * res
    cy = occupancy.origin_y + (rows + 0.5) * res
    # StochOccupancyGrid2D.is_free(center): snap with the grid resolution, truncate to an index.
    gres = occupancy.resolution
    sx = gres * np.round(cx / gres)
    sy = gres * np.round(cy / gres)
    gx = np.trunc((sx - occupancy.origin_x) / gres).astype(np.int64)
    gy = np.trunc((sy - occupancy.origin_y) / gres).astype(np.int64)
    free_rows = occupancy._free_lookup()
    ok = np.zeros((height, width), dtype=np.bool_)
    col_in = (gx >= 0) & (gx < width)
    row_in = (gy >= 0) & (gy < height)
    if free_rows is not None and col_in.all() and row_in.all():
        free = np.asarray(free_rows, dtype=np.bool_)
        ok[:, :] = free[np.ix_(gy, gx)]
    else:
        for r in range(height):
            for c in range(width):
                ok[r, c] = occupancy.is_free((float(cx[c]), float(cy[r])))
    lo = statespace_lo
    hi = statespace_hi
    ok &= ((lo[0] <= cx) & (cx < hi[0]))[None, :]
    ok &= ((lo[1] <= cy) & (cy < hi[1]))[:, None]
    return ok


_GRAPH_TABLES = weakref.WeakKeyDictionary()


def graph_tables(graph, width, height):
    """
    ``(edge_has, edge_w)`` arrays of shape ``(width, height, 9)`` for a social graph whose
    nodes are ``(col, row)`` cells and edges join 8-neighbors; None if the graph has any
    edge the arrays cannot represent. Cached per graph object (rebuilt if its size changes).

    Only plain ``networkx.DiGraph`` graphs are converted: undirected graphs answer
    ``has_edge`` in both directions and other graph-like objects may behave differently,
    so those return None (the caller then uses the Python loop).
    """
    import networkx as nx

    if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
        return None
    key = (graph.number_of_nodes(), graph.number_of_edges(), width, height)
    try:
        cached = _GRAPH_TABLES.get(graph)
    except TypeError:
        cached = None
    if cached is not None and cached[0] == key:
        return cached[1]
    edge_has = np.zeros((width, height, 9), dtype=np.bool_)
    edge_w = np.zeros((width, height, 9))
    result = (edge_has, edge_w)
    for u, v, data in graph.edges(data=True):
        try:
            ux, uy = u
            vx, vy = v
            if int(ux) != ux or int(uy) != uy or int(vx) != vx or int(vy) != vy:
                result = None
                break
            ux, uy, vx, vy = int(ux), int(uy), int(vx), int(vy)
            weight = float(data["weight"])
        except (TypeError, ValueError, KeyError):
            result = None
            break
        dx, dy = vx - ux, vy - uy
        if not (0 <= ux < width and 0 <= uy < height) or not (-1 <= dx <= 1 and -1 <= dy <= 1) or dx == dy == 0:
            result = None
            break
        k = (dx + 1) * 3 + (dy + 1)
        edge_has[ux, uy, k] = True
        edge_w[ux, uy, k] = weight
    try:
        _GRAPH_TABLES[graph] = (key, result)
    except TypeError:
        pass
    return result


def lattice_bounds(occupancy, astar_res):
    """Lattice index offset and size covering every state whose grid index is in range."""
    lo_i = math.floor(occupancy.origin_x / astar_res) - _LATTICE_MARGIN
    lo_j = math.floor(occupancy.origin_y / astar_res) - _LATTICE_MARGIN
    size_i = int(occupancy.width + 2 * _LATTICE_MARGIN + 1)
    size_j = int(occupancy.height + 2 * _LATTICE_MARGIN + 1)
    return int(lo_i), int(lo_j), size_i, size_j
