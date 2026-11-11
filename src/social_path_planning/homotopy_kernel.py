"""Optional numba-compiled search loop for ``homotopy_astar.HomotopyMixin``.

Reproduces ``HomotopyMixin._homotopy_solve`` (the Python loop) result for result: the same
lattice, neighbour order, social cost, graph biasing, word bookkeeping, filters, alpha
bound and heap ordering. It reuses the bit-exact building blocks of ``astar_kernel``
(``py_hypot``, ``rint_int``, ``_rightness_penalty``, the wall-distance cache/ray-cast) and
falls back to the Python loop whenever ``astar_kernel`` would.

Data structures
---------------
* Words are interned in a trie: ``word id -> (parent id, last letter, length, net
  crossing counts)``. Extending a word by a letter either cancels its last letter (go to
  the parent) or follows/creates a child. Id 0 is the empty word.
* Search states ``(cell, word id)`` get dense ids through a typed dict. Per-state arrays
  hold the cell, word, parent state, geometric cost-to-arrive, stored right-wall distance
  and the closed flag.
* The heap is a binary heap over ``(priority, x, y, state)``. Ties are broken exactly as
  Python compares ``(priority, ((x, y), word))``: x, then y, then the word's letters
  lexicographically (a proper prefix sorts first). Distinct states never compare equal,
  so any correct heap pops the same sequence as ``heapq``.

One documented difference: the winding filter evaluates ``atan2`` in compiled code, so a
state whose winding lands within one ulp of ``max_abs_winding`` could be filtered
differently from the Python loop. Time limits are checked every 1024 pops, as in
``astar_kernel``.
"""
from __future__ import annotations

import math

import numpy as np

from social_path_planning import astar_kernel

AVAILABLE = astar_kernel.AVAILABLE

STATUS_DONE = 0  # queue exhausted, alpha bound reached or k classes found
STATUS_TIMED_OUT = -1
STATUS_ABORT = -2  # a state fell outside the lattice arrays: use the Python loop

if AVAILABLE:
    from numba import njit, types
    from numba.typed import Dict

    from social_path_planning.astar_kernel import (
        MODE_GRAPH,
        MODE_MODIFIED,
        MODE_VANILLA,
        TIME_CHECK_EVERY,
        _rightness_penalty,
        _wall_time,
        py_hypot,
        rint_int,
    )

    TWO_PI = 2 * math.pi

    # ------------------------------------------------------------------
    # Word trie
    # ------------------------------------------------------------------

    @njit(cache=True)
    def _word_letters(w, w_parent, w_last, w_len):
        n = w_len[w]
        out = np.empty(n, dtype=np.int64)
        k = n - 1
        while w != 0:
            out[k] = w_last[w]
            w = w_parent[w]
            k -= 1
        return out

    @njit(cache=True)
    def _word_less(wa, wb, w_parent, w_last, w_len):
        """Python tuple ``<`` on the two words' letter sequences."""
        if wa == wb:
            return False
        la = _word_letters(wa, w_parent, w_last, w_len)
        lb = _word_letters(wb, w_parent, w_last, w_len)
        n = min(la.shape[0], lb.shape[0])
        for k in range(n):
            if la[k] != lb[k]:
                return la[k] < lb[k]
        return la.shape[0] < lb.shape[0]

    # ------------------------------------------------------------------
    # Heap (entries in parallel arrays; caller owns growth)
    # ------------------------------------------------------------------

    @njit(cache=True)
    def _entry_less(i, j, hp, hx, hy, hs, s_word, w_parent, w_last, w_len):
        if hp[i] != hp[j]:
            return hp[i] < hp[j]
        if hx[i] != hx[j]:
            return hx[i] < hx[j]
        if hy[i] != hy[j]:
            return hy[i] < hy[j]
        return _word_less(s_word[hs[i]], s_word[hs[j]], w_parent, w_last, w_len)

    @njit(cache=True)
    def _swap(i, j, hp, hx, hy, hs):
        hp[i], hp[j] = hp[j], hp[i]
        hx[i], hx[j] = hx[j], hx[i]
        hy[i], hy[j] = hy[j], hy[i]
        hs[i], hs[j] = hs[j], hs[i]

    @njit(cache=True)
    def _sift_up(pos, hp, hx, hy, hs, s_word, w_parent, w_last, w_len):
        while pos > 0:
            parent = (pos - 1) >> 1
            if _entry_less(pos, parent, hp, hx, hy, hs, s_word, w_parent, w_last, w_len):
                _swap(pos, parent, hp, hx, hy, hs)
                pos = parent
            else:
                break

    @njit(cache=True)
    def _sift_down(n, hp, hx, hy, hs, s_word, w_parent, w_last, w_len):
        pos = 0
        while True:
            left = 2 * pos + 1
            if left >= n:
                break
            best = left
            right = left + 1
            if right < n and _entry_less(right, left, hp, hx, hy, hs, s_word, w_parent, w_last, w_len):
                best = right
            if _entry_less(best, pos, hp, hx, hy, hs, s_word, w_parent, w_last, w_len):
                _swap(best, pos, hp, hx, hy, hs)
                pos = best
            else:
                break

    # ------------------------------------------------------------------
    # Geometry helpers (mirror homotopy.ObstacleSet)
    # ------------------------------------------------------------------

    @njit(cache=True)
    def _crossings(c1, r1, c2, r2, height, width,
                   h_ptr, h_origin2, h_sign, h_id, v_ptr, v_origin2, v_sign, v_id, out):
        """ObstacleSet.edge_crossings into ``out``; returns the number of letters."""
        n = 0
        nh = 0
        # Horizontal rays on the line between rows r1 and r2, ids ascending in the table.
        if r1 != r2:
            row = min(r1, r2)
            if 0 <= row < height:
                x2 = c1 + c2
                up = r2 > r1
                lo = h_ptr[row]
                hi = h_ptr[row + 1]
                if up:
                    rng_start, rng_stop, rng_step = lo, hi, 1
                else:
                    rng_start, rng_stop, rng_step = hi - 1, lo - 1, -1
                for q in range(rng_start, rng_stop, rng_step):
                    if (x2 - h_origin2[q]) * h_sign[q] > 0:
                        ccw = up if h_sign[q] > 0 else (not up)
                        out[n] = h_id[q] if ccw else -h_id[q]
                        n += 1
            nh = n
        if c1 != c2:
            col = min(c1, c2)
            if 0 <= col < width:
                y2 = r1 + r2
                right = c2 > c1
                lo = v_ptr[col]
                hi = v_ptr[col + 1]
                if right:
                    rng_start, rng_stop, rng_step = lo, hi, 1
                else:
                    rng_start, rng_stop, rng_step = hi - 1, lo - 1, -1
                for q in range(rng_start, rng_stop, rng_step):
                    if (y2 - v_origin2[q]) * v_sign[q] > 0:
                        ccw = (not right) if v_sign[q] > 0 else right
                        out[n] = v_id[q] if ccw else -v_id[q]
                        n += 1
        if c2 <= c1 and nh > 0 and n > nh:
            # Vertical crossings come first when not moving +x: rotate.
            tmp = out[:n].copy()
            nv = n - nh
            for q in range(nv):
                out[q] = tmp[nh + q]
            for q in range(nh):
                out[nv + q] = tmp[q]
        return n

    @njit(cache=True)
    def _overlap_fraction(cells_c, cells_r, mask):
        """homotopy_astar.overlap_fraction (same sums in the same order)."""
        height = mask.shape[0]
        width = mask.shape[1]
        m = cells_c.shape[0]
        total = 0.0
        covered = 0.0
        if m == 0:
            return 1.0
        c = cells_c[0]
        r = cells_r[0]
        prev_in = 0 <= c < width and 0 <= r < height and mask[r, c]
        for q in range(1, m):
            c1 = cells_c[q - 1]
            r1 = cells_r[q - 1]
            c2 = cells_c[q]
            r2 = cells_r[q]
            dc = c2 - c1
            dr = r2 - r1
            seg = math.sqrt(dc * dc + dr * dr)
            cur_in = 0 <= c2 < width and 0 <= r2 < height and mask[r2, c2]
            total += seg
            if prev_in or cur_in:
                covered += seg
            prev_in = cur_in
        if total == 0.0:
            return 1.0
        return covered / total

    @njit(cache=True)
    def _stamp(cells_c, cells_r, offsets, mask):
        height = mask.shape[0]
        width = mask.shape[1]
        for q in range(offsets.shape[0]):
            dc = offsets[q, 0]
            dr = offsets[q, 1]
            for p in range(cells_c.shape[0]):
                c = cells_c[p] + dc
                r = cells_r[p] + dr
                if 0 <= c < width and 0 <= r < height:
                    mask[r, c] = True

    @njit(cache=True)
    def _ray_angle(c, r, ox, oy, dx, dy):
        vx = c - ox
        vy = r - oy
        return math.atan2(dx * vy - dy * vx, dx * vx + dy * vy) % TWO_PI

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @njit(cache=True)
    def _solve(mode, start_i, start_j, goal_i, goal_j, astar_res, i0, j0, ni_size, nj_size,
               ox, oy, width, height, center_ok,
               grid_res, probs, thresh, use_cache, d_right, unit_dirs, robot_d, extra,
               edge_has, edge_w,
               n_obs, obs_pos, h_ptr, h_origin2, h_sign, h_id, v_ptr, v_origin2, v_sign, v_id,
               ray_ox, ray_oy, ray_dx, ray_dy, start_angles,
               use_winding, max_abs_winding, max_word_len, k_max, alpha,
               special_letters, special_ptr, n_blocked, has_target,
               use_merge, merge_threshold, merge_offsets,
               t_start, time_limit):
        """Returns (status, found_word, found_cost, found_state, found_merged, found_overlap,
        n_found, s_a, s_b, s_parent, n_states, n_closed, w_parent, w_last, n_words, tel).

        ``found_*`` lists every accepted goal class in pop order; ``found_merged`` is -1 for
        a kept class, else the kept index it duplicates (OverlapMerger)."""
        # --- word trie -------------------------------------------------
        w_cap = 1024
        w_parent = np.zeros(w_cap, dtype=np.int64)
        w_last = np.zeros(w_cap, dtype=np.int64)
        w_len = np.zeros(w_cap, dtype=np.int64)
        w_counts = np.zeros((w_cap, max(n_obs, 1)), dtype=np.int64)
        n_words = 1
        n_letters = 2 * n_obs + 1
        child = Dict.empty(key_type=types.int64, value_type=types.int64)

        # --- states ----------------------------------------------------
        s_cap = 1 << 16
        s_a = np.zeros(s_cap, dtype=np.int64)
        s_b = np.zeros(s_cap, dtype=np.int64)
        s_word = np.zeros(s_cap, dtype=np.int64)
        s_parent = np.full(s_cap, -1, dtype=np.int64)
        s_cost = np.zeros(s_cap)
        s_d2r = np.zeros(s_cap)
        s_closed = np.zeros(s_cap, dtype=np.bool_)
        n_states = 1
        n_cells = ni_size * nj_size
        state_of = Dict.empty(key_type=types.int64, value_type=types.int64)

        # --- heap ------------------------------------------------------
        h_cap = 1 << 16
        hp = np.zeros(h_cap)
        hx = np.zeros(h_cap)
        hy = np.zeros(h_cap)
        hs = np.zeros(h_cap, dtype=np.int64)
        h_n = 0

        tel = np.zeros(8)
        buf = np.zeros(max(64, 2 * n_obs + 2), dtype=np.int64)
        f_cap = max(k_max, 1) * 4
        found_word = np.full(f_cap, -1, dtype=np.int64)
        found_cost = np.zeros(f_cap)
        found_state = np.full(f_cap, -1, dtype=np.int64)
        found_merged = np.full(f_cap, -1, dtype=np.int64)
        found_overlap = np.zeros(f_cap)
        n_found = 0
        n_kept = 0
        if use_merge:
            masks = np.zeros((max(k_max, 1), height, width), dtype=np.bool_)
        else:
            masks = np.zeros((1, 1, 1), dtype=np.bool_)

        # Intern the blocked / target words first (so goal checks are id comparisons).
        n_special = special_ptr.shape[0] - 1
        special_ids = np.zeros(max(n_special, 1), dtype=np.int64)
        for q in range(n_special):
            w = 0
            for p in range(special_ptr[q], special_ptr[q + 1]):
                letter = special_letters[p]
                if w != 0 and w_last[w] == -letter:
                    w = w_parent[w]
                    continue
                key = w * n_letters + (letter + n_obs)
                if key in child:
                    w = child[key]
                    continue
                if n_words == w_cap:
                    w_cap *= 2
                    w_parent = np.concatenate((w_parent, np.zeros(w_cap - n_words, dtype=np.int64)))
                    w_last = np.concatenate((w_last, np.zeros(w_cap - n_words, dtype=np.int64)))
                    w_len = np.concatenate((w_len, np.zeros(w_cap - n_words, dtype=np.int64)))
                    w_counts = np.concatenate((w_counts, np.zeros((w_cap - n_words, w_counts.shape[1]), dtype=np.int64)))
                nw = n_words
                n_words += 1
                w_parent[nw] = w
                w_last[nw] = letter
                w_len[nw] = w_len[w] + 1
                for o in range(n_obs):
                    w_counts[nw, o] = w_counts[w, o]
                pos = obs_pos[abs(letter)]
                w_counts[nw, pos] += 1 if letter > 0 else -1
                child[key] = nw
                w = nw
            special_ids[q] = w
        target_id = special_ids[n_blocked] if has_target else -1

        gx = astar_res * goal_i
        gy = astar_res * goal_j
        sx0 = astar_res * start_i
        sy0 = astar_res * start_j
        s_a[0] = start_i - i0
        s_b[0] = start_j - j0
        state_of[s_b[0] + nj_size * s_a[0]] = 0  # word 0 -> key = cell key
        hp[0] = abs(sx0 - gx) + abs(sy0 - gy)
        hx[0] = sx0
        hy[0] = sy0
        hs[0] = 0
        h_n = 1

        bound = np.inf
        status = STATUS_DONE
        pops = 0
        n_closed = 0
        offsets = (1, 0, -1)

        while h_n > 0:
            prio = hp[0]
            cx = hx[0]
            cy = hy[0]
            sc = hs[0]
            h_n -= 1
            if h_n > 0:
                hp[0] = hp[h_n]
                hx[0] = hx[h_n]
                hy[0] = hy[h_n]
                hs[0] = hs[h_n]
                _sift_down(h_n, hp, hx, hy, hs, s_word, w_parent, w_last, w_len)

            if prio > bound:
                break
            pops += 1
            if pops % TIME_CHECK_EVERY == 0:
                if _wall_time() - t_start > time_limit:
                    status = STATUS_TIMED_OUT
                    break
            if s_closed[sc]:
                continue
            s_closed[sc] = True
            n_closed += 1
            ci = s_a[sc] + i0
            cj = s_b[sc] + j0
            wc = s_word[sc]

            if ci == goal_i and cj == goal_j:
                accepted = True
                for q in range(n_found):
                    if found_word[q] == wc:
                        accepted = False
                for q in range(n_blocked):
                    if special_ids[q] == wc:
                        accepted = False
                if has_target and wc != target_id:
                    accepted = False
                if accepted:
                    merged = -1
                    overlap = 0.0
                    if use_merge:
                        m = 0
                        s = sc
                        while s >= 0:
                            m += 1
                            s = s_parent[s]
                        cells_c = np.empty(m, dtype=np.int64)
                        cells_r = np.empty(m, dtype=np.int64)
                        s = sc
                        q = m - 1
                        while s >= 0:
                            cells_c[q] = rint_int((astar_res * (s_a[s] + i0) - ox) / astar_res)
                            cells_r[q] = rint_int((astar_res * (s_b[s] + j0) - oy) / astar_res)
                            s = s_parent[s]
                            q -= 1
                        best = -1
                        for q in range(n_kept):
                            v = _overlap_fraction(cells_c, cells_r, masks[q])
                            if best < 0 or v > overlap:
                                overlap = v
                                best = q
                        if best >= 0 and overlap >= merge_threshold:
                            merged = best
                        else:
                            _stamp(cells_c, cells_r, merge_offsets, masks[n_kept])
                    if n_found == f_cap:
                        grow = f_cap
                        f_cap *= 2
                        found_word = np.concatenate((found_word, np.full(grow, -1, dtype=np.int64)))
                        found_cost = np.concatenate((found_cost, np.zeros(grow)))
                        found_state = np.concatenate((found_state, np.full(grow, -1, dtype=np.int64)))
                        found_merged = np.concatenate((found_merged, np.full(grow, -1, dtype=np.int64)))
                        found_overlap = np.concatenate((found_overlap, np.zeros(grow)))
                    found_word[n_found] = wc
                    found_cost[n_found] = prio
                    found_state[n_found] = sc
                    found_merged[n_found] = merged
                    found_overlap[n_found] = overlap
                    n_found += 1
                    if merged >= 0:
                        continue  # practical duplicate: doesn't count toward k
                    n_kept += 1
                    if n_kept == 1:
                        bound = (1.0 + alpha) * prio
                    if n_kept >= k_max or has_target or n_obs == 0:
                        break
                continue

            cost_c = s_cost[sc]
            h_c = abs(cx - gx) + abs(cy - gy)
            if sc == 0:
                d2r_prev = 0.0
            else:
                d2r_prev = s_d2r[s_parent[sc]]
            c1x = rint_int((cx - ox) / astar_res)
            c1y = rint_int((cy - oy) / astar_res)

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
                        return (STATUS_ABORT, found_word, found_cost, found_state, found_merged,
                                found_overlap, n_found, s_a, s_b, s_parent, n_states, n_closed,
                                w_parent, w_last, n_words, tel)
                    nx = astar_res * ni
                    ny = astar_res * nj

                    # Word of the neighbour.
                    c2x = rint_int((nx - ox) / astar_res)
                    c2y = rint_int((ny - oy) / astar_res)
                    n_cross = _crossings(c1x, c1y, c2x, c2y, height, width, h_ptr, h_origin2, h_sign,
                                         h_id, v_ptr, v_origin2, v_sign, v_id, buf)
                    wn = wc
                    new_word_pending = False
                    pend_parent = 0
                    pend_letter = 0
                    for q in range(n_cross):
                        letter = buf[q]
                        if wn != 0 and w_last[wn] == -letter:
                            wn = w_parent[wn]
                            continue
                        key = wn * n_letters + (letter + n_obs)
                        if key in child:
                            wn = child[key]
                            continue
                        # Create the child word (words are cheap; unused ones are harmless).
                        if n_words == w_cap:
                            w_cap *= 2
                            w_parent = np.concatenate((w_parent, np.zeros(w_cap - n_words, dtype=np.int64)))
                            w_last = np.concatenate((w_last, np.zeros(w_cap - n_words, dtype=np.int64)))
                            w_len = np.concatenate((w_len, np.zeros(w_cap - n_words, dtype=np.int64)))
                            w_counts = np.concatenate((w_counts, np.zeros((w_cap - n_words, w_counts.shape[1]), dtype=np.int64)))
                        nw = n_words
                        n_words += 1
                        w_parent[nw] = wn
                        w_last[nw] = letter
                        w_len[nw] = w_len[wn] + 1
                        for o in range(n_obs):
                            w_counts[nw, o] = w_counts[wn, o]
                        pos = obs_pos[abs(letter)]
                        w_counts[nw, pos] += 1 if letter > 0 else -1
                        child[key] = nw
                        wn = nw

                    skey = wn * n_cells + na * nj_size + nb
                    sn = -1
                    if skey in state_of:
                        sn = state_of[skey]
                    if sn >= 0 and s_closed[sn]:
                        continue
                    dist = py_hypot(cx - nx, cy - ny)
                    tentative = cost_c + dist
                    if sn >= 0 and not (tentative < s_cost[sn]):
                        continue
                    if sn < 0:
                        # New state: word filters (HomotopyMixin._word_allowed).
                        if w_len[wn] > max_word_len:
                            continue
                        if use_winding and n_obs > 0:
                            worst = 0.0
                            for o in range(n_obs):
                                ang = _ray_angle(c2x, c2y, ray_ox[o], ray_oy[o], ray_dx[o], ray_dy[o])
                                wv = abs((ang - start_angles[o]) + TWO_PI * w_counts[wn, o])
                                if wv > worst:
                                    worst = wv
                            if worst >= max_abs_winding:
                                continue

                    nd2r = 0.0
                    if mode == MODE_VANILLA:
                        edge = dist
                    elif mode == MODE_MODIFIED:
                        social, nd2r = _rightness_penalty(
                            cx, cy, nx, ny, gx, gy, sx0, sy0, d2r_prev, robot_d, extra, astar_res,
                            ox, oy, grid_res, width, height, probs, thresh, use_cache, d_right, unit_dirs)
                        edge = dist + social
                        tel[0] += 1
                        s = edge - dist
                        tel[1] += s
                        if s > tel[2]:
                            tel[2] = s
                        if s > 1e-9:
                            tel[3] += 1
                    else:
                        ddx = c2x - c1x
                        ddy = c2y - c1y
                        on_graph = False
                        w = 0.0
                        if (0 <= c1x < width and 0 <= c1y < height
                                and -1 <= ddx <= 1 and -1 <= ddy <= 1):
                            kk = (ddx + 1) * 3 + (ddy + 1)
                            if edge_has[c1x, c1y, kk]:
                                on_graph = True
                                w = edge_w[c1x, c1y, kk]
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
                        tel[1] += s
                        if s > tel[2]:
                            tel[2] = s
                        if s > 1e-9:
                            tel[3] += 1

                    h_nb = abs(nx - gx) + abs(ny - gy)
                    priority = prio + edge + h_nb - h_c
                    if priority > bound:
                        continue

                    if sn < 0:
                        if n_states == s_cap:
                            grow = s_cap
                            s_cap *= 2
                            s_a = np.concatenate((s_a, np.zeros(grow, dtype=np.int64)))
                            s_b = np.concatenate((s_b, np.zeros(grow, dtype=np.int64)))
                            s_word = np.concatenate((s_word, np.zeros(grow, dtype=np.int64)))
                            s_parent = np.concatenate((s_parent, np.full(grow, -1, dtype=np.int64)))
                            s_cost = np.concatenate((s_cost, np.zeros(grow)))
                            s_d2r = np.concatenate((s_d2r, np.zeros(grow)))
                            s_closed = np.concatenate((s_closed, np.zeros(grow, dtype=np.bool_)))
                        sn = n_states
                        n_states += 1
                        s_a[sn] = na
                        s_b[sn] = nb
                        s_word[sn] = wn
                        state_of[skey] = sn
                    if mode != MODE_VANILLA:
                        s_d2r[sn] = nd2r
                    s_parent[sn] = sc
                    s_cost[sn] = tentative

                    if h_n == h_cap:
                        grow = h_cap
                        h_cap *= 2
                        hp = np.concatenate((hp, np.zeros(grow)))
                        hx = np.concatenate((hx, np.zeros(grow)))
                        hy = np.concatenate((hy, np.zeros(grow)))
                        hs = np.concatenate((hs, np.zeros(grow, dtype=np.int64)))
                    hp[h_n] = priority
                    hx[h_n] = nx
                    hy[h_n] = ny
                    hs[h_n] = sn
                    h_n += 1
                    _sift_up(h_n - 1, hp, hx, hy, hs, s_word, w_parent, w_last, w_len)

        return (status, found_word, found_cost, found_state, found_merged, found_overlap,
                n_found, s_a, s_b, s_parent,
                n_states, n_closed, w_parent, w_last, n_words, tel)


def ray_tables(obstacles, width, height):
    """CSR tables of rays by row (horizontal) / column (vertical), ids ascending."""
    horiz = [[] for _ in range(height)]
    vert = [[] for _ in range(width)]
    for obstacle_id in sorted(obstacles.ids):
        ray = obstacles.rays[obstacle_id]
        idx = (ray.line2 - 1) // 2
        (horiz if ray.axis == 0 else vert)[idx].append(ray)

    def csr(rows):
        ptr = np.zeros(len(rows) + 1, dtype=np.int64)
        origin2, sign, ids = [], [], []
        for k, rays in enumerate(rows):
            for ray in rays:
                origin2.append(ray.origin2)
                sign.append(ray.sign)
                ids.append(ray.obstacle_id)
            ptr[k + 1] = len(ids)
        return (ptr, np.array(origin2, dtype=np.int64), np.array(sign, dtype=np.int64),
                np.array(ids, dtype=np.int64))

    return csr(horiz) + csr(vert)


_READY = None  # None = not checked yet


def kernel_ready():
    """astar_kernel's platform checks plus one warm-up compile of this kernel.

    Any compile or run failure disables the kernel (with a warning) so callers fall
    back to the Python loop.
    """
    global _READY
    if _READY is not None:
        return _READY
    if not (AVAILABLE and astar_kernel.kernel_ready()):
        _READY = False
        return False
    try:
        _warm_up()
        _READY = True
    except Exception as exc:  # pragma: no cover - defensive
        import warnings

        warnings.warn(
            f"social_path_planning.homotopy_kernel disabled: {type(exc).__name__}: {exc}", RuntimeWarning
        )
        _READY = False
    return _READY


def _warm_up():
    import time

    probs = np.zeros((6, 6))
    center_ok = np.ones((6, 6), dtype=np.bool_)
    unit_dirs = np.zeros((8, 2))
    d_right = np.zeros((1, 1, 8))
    edge_has = np.zeros((6, 6, 9), dtype=np.bool_)
    edge_w = np.zeros((6, 6, 9))
    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0)
    ptr6 = np.zeros(7, dtype=np.int64)
    for mode in (MODE_VANILLA, MODE_MODIFIED, MODE_GRAPH):
        _solve(mode, 0, 0, 5, 5, 1.0, -3, -3, 12, 12, 0.0, 0.0, 6, 6, center_ok,
               1.0, probs, 0.5, False, d_right, unit_dirs, 0.4, 0.25,
               edge_has, edge_w,
               0, np.zeros(1, dtype=np.int64), ptr6, empty_i, empty_i, empty_i, ptr6, empty_i, empty_i, empty_i,
               empty_f, empty_f, empty_f, empty_f, empty_f,
               True, 2 * math.pi, 1, 2, 0.3,
               empty_i, np.zeros(1, dtype=np.int64), 0, False,
               mode == MODE_MODIFIED, 0.8, np.zeros((1, 2), dtype=np.int64),
               time.time(), 1e9)
