"""Milestone 1 of HOMOTOPY_PLAN.md: obstacle extraction, rays and path signatures."""
import itertools
import math

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from social_path_planning.homotopy import (
    ObstacleSet,
    Ray,
    blocked_mask,
    extend_word,
    extract_obstacles,
    reduce_word,
)
from social_path_planning.occupancy_grid import StochOccupancyGrid2D

RES = 0.2
ROBOT_D = 0.4  # half-window of 1 cell: an obstacle cell (c, r) blocks (c-1..c, r-1..r)
STEPS = [(dc, dr) for dc in (-1, 0, 1) for dr in (-1, 0, 1) if dc or dr]


def make_grid(rects, width=40, height=30, origin=(0.0, 0.0)):
    """Grid with obstacle rectangles given as (col0, col1, row0, row1), half-open."""
    probs = np.zeros((height, width))
    for c0, c1, r0, r1 in rects:
        probs[r0:r1, c0:c1] = 1.0
    return StochOccupancyGrid2D(RES, width, height, origin[0], origin[1], 10, probs, robot_d=ROBOT_D)


def lattice_walk(obs, waypoints):
    """8-connected A* states (world coords) through the given cell-index waypoints."""
    waypoints = [(int(c), int(r)) for c, r in waypoints]
    c, r = waypoints[0]
    cells = [(c, r)]
    for wc, wr in waypoints[1:]:
        while (c, r) != (wc, wr):
            c += (wc > c) - (wc < c)
            r += (wr > r) - (wr < r)
            cells.append((c, r))
    return [obs.to_world(cell) for cell in cells]


def smooth(states):
    """Same spline as AStar.smooth_path."""
    path = np.array(states)
    t = np.linspace(0, 1, len(path))
    cs_x, cs_y = CubicSpline(t, path[:, 0]), CubicSpline(t, path[:, 1])
    t_new = np.linspace(0, 1, num=5 * len(path))
    return [(cs_x(ti), cs_y(ti)) for ti in t_new]


def assert_free(occ, obs, states):
    blocked = blocked_mask(occ)
    for x in states:
        c, r = obs.state_index(x)
        assert not blocked[r, c], f"state {x} (cell {(c, r)}) is blocked"


@pytest.fixture
def one_block():
    occ = make_grid([(15, 25, 12, 18)])  # blocked after inflation: cols 14..24, rows 11..18
    return occ, extract_obstacles(occ, min_area_m2=1.0)


# ----------------------------------------------------------------------
# Test 5: obstacle extraction
# ----------------------------------------------------------------------

def test_single_obstacle_extracted(one_block):
    occ, obs = one_block
    assert obs.ids == [1]
    col, row = obs.rep_cells[1]
    assert obs.labels[row, col] == obs.component_of_id[1]
    assert obs.areas_m2[1] == pytest.approx(np.count_nonzero(blocked_mask(occ)) * RES * RES)


def test_blocked_mask_matches_planner_passability(one_block):
    from social_path_planning.a_star import AStar
    from social_path_planning.mapf_comparison.grid_traversability import fine_cell_center_world

    occ, _ = one_block
    planner = AStar([0, 0], [occ.width * RES, occ.height * RES], (1, 1), (2, 2), occ, resolution=RES)
    blocked = blocked_mask(occ, planner.statespace_lo, planner.statespace_hi)
    expected = np.array([
        [not planner.is_free(fine_cell_center_world(occ, c, r)) for c in range(occ.width)]
        for r in range(occ.height)
    ])
    np.testing.assert_array_equal(blocked, expected)
    rows, cols = np.nonzero(blocked)
    assert (cols.min(), cols.max(), rows.min(), rows.max()) == (14, 24, 11, 18)


def test_border_component_dropped():
    occ = make_grid([(0, 10, 10, 20), (20, 30, 10, 18)])  # first block touches the left edge
    obs = extract_obstacles(occ, min_area_m2=0.0)
    assert obs.num_obstacles == 1
    col, row = obs.rep_cells[1]
    assert col >= 20


def test_small_component_dropped_unless_threshold_lowered():
    occ = make_grid([(15, 25, 12, 18), (5, 6, 5, 6)])  # second: one cell -> 2x2 blocked = 0.16 m^2
    assert extract_obstacles(occ, min_area_m2=1.0).num_obstacles == 1
    assert extract_obstacles(occ, min_area_m2=0.0).num_obstacles == 2


def test_gap_narrower_than_robot_merges_obstacles():
    # One free column between the raw blocks; inflation closes it.
    narrow = make_grid([(10, 15, 10, 18), (16, 21, 10, 18)])
    assert extract_obstacles(narrow, min_area_m2=0.0).num_obstacles == 1
    wide = make_grid([(10, 15, 10, 18), (18, 23, 10, 18)])
    assert extract_obstacles(wide, min_area_m2=0.0).num_obstacles == 2


def test_diagonally_touching_cells_are_separate_obstacles():
    # Raw cells (10,10) and (12,12) inflate to 2x2 blocks (9..10 and 11..12) touching only
    # at a corner; the 8-connected planner can slip between them, so they stay separate.
    occ = make_grid([(10, 11, 10, 11), (12, 13, 12, 13)])
    blocked = blocked_mask(occ)
    assert blocked[10, 10] and blocked[11, 11] and not blocked[10, 11] and not blocked[11, 10]
    assert extract_obstacles(occ, min_area_m2=0.0).num_obstacles == 2


def test_rays_lie_between_lattice_points_and_start_inside_obstacle():
    occ = make_grid([(15, 25, 12, 18), (3, 8, 3, 8), (30, 36, 20, 26)])
    obs = extract_obstacles(occ, min_area_m2=0.0)
    blocked = blocked_mask(occ)
    assert obs.num_obstacles == 3
    for obstacle_id, ray in obs.rays.items():
        assert ray.line2 % 2 == 1 and ray.origin2 % 2 == 0
        col, row = obs.rep_cells[obstacle_id]
        assert blocked[row, col]
        # The cell on the other side of the ray's line at its origin is blocked too, so the
        # origin sits inside the obstacle, not on its boundary.
        if ray.axis == 0:
            assert blocked[row + 1, col]
        else:
            assert blocked[row, col + 1]


def test_mismatched_astar_resolution_rejected(one_block):
    occ, _ = one_block
    with pytest.raises(ValueError):
        extract_obstacles(occ, astar_resolution=2 * RES)


def test_extract_on_paper_scenarios():
    from social_path_planning.common import build_occ_grid

    for scenario in ("sample2_default", "circular_default"):
        occ, _, res, statespace_hi = build_occ_grid(scenario, wall_cache="none")
        lo = [0, 0]
        obs = extract_obstacles(
            occ, min_area_m2=1.0, statespace_lo=lo, statespace_hi=statespace_hi, astar_resolution=res,
        )
        blocked = blocked_mask(occ, lo, statespace_hi)
        assert obs.num_obstacles >= 1, scenario
        for obstacle_id in obs.ids:
            col, row = obs.rep_cells[obstacle_id]
            assert blocked[row, col]
            assert obs.labels[row, col] == obs.component_of_id[obstacle_id]


# ----------------------------------------------------------------------
# Test 3: signature invariants
# ----------------------------------------------------------------------

def test_word_reduction():
    assert extend_word((1, 2), -2) == (1,)
    assert reduce_word([1, 2, -2, -1, 3]) == (3,)
    assert reduce_word([1, -2]) == (1, -2)


def test_two_sides_of_obstacle_have_different_words(one_block):
    occ, obs = one_block
    start, goal = (5, 15), (35, 15)
    above = lattice_walk(obs, [start, (10, 24), (30, 24), goal])
    above_alt = lattice_walk(obs, [start, (12, 27), (28, 22), goal])
    below = lattice_walk(obs, [start, (10, 5), (30, 5), goal])
    for path in (above, above_alt, below):
        assert_free(occ, obs, path)

    w_above = obs.lattice_path_word(above)
    assert w_above == obs.lattice_path_word(above_alt)  # homotopic -> same word
    assert w_above != obs.lattice_path_word(below)

    # above followed by below reversed is a loop around the obstacle: winding +-2*pi,
    # and its word is a single letter.
    loop = above + below[::-1][1:]
    loop_word = obs.lattice_path_word(loop)
    assert len(loop_word) == 1 and abs(loop_word[0]) == 1
    assert obs.path_winding(loop)[0] == pytest.approx(2 * math.pi * np.sign(loop_word[0]))


def test_smoothed_path_keeps_its_word(one_block):
    occ, obs = one_block
    for waypoints in (
        [(5, 15), (10, 24), (30, 24), (35, 15)],
        [(5, 15), (10, 5), (30, 5), (35, 15)],
    ):
        states = lattice_walk(obs, waypoints)
        assert obs.path_word(smooth(states)) == obs.lattice_path_word(states)


def test_ccw_loop_is_positive_and_path_times_reverse_is_trivial(one_block):
    _, obs = one_block
    ccw = lattice_walk(obs, [(30, 8), (30, 24), (10, 24), (10, 8), (30, 8)])
    assert obs.lattice_path_word(ccw) == (1,)
    assert obs.lattice_path_word(ccw[::-1]) == (-1,)
    assert obs.lattice_path_word(ccw + ccw[::-1][1:]) == ()
    outside = lattice_walk(obs, [(30, 2), (38, 2), (38, 28), (30, 2)])
    assert obs.lattice_path_word(outside) == ()


def test_closed_loops_winding_matches_word():
    occ = make_grid([(8, 14, 8, 14), (24, 30, 8, 14), (16, 22, 18, 24)])
    obs = extract_obstacles(occ, min_area_m2=0.0)
    assert obs.num_obstacles == 3
    blocked = blocked_mask(occ)
    rng = np.random.default_rng(0)
    checked = nontrivial = 0
    for _ in range(3000):
        waypoints = [tuple(rng.integers(0, [40, 30])) for _ in range(5)]
        loop = lattice_walk(obs, waypoints + [waypoints[0]])
        # Winding is only defined for loops in free space (a walk through an obstacle can
        # pass through the reference point itself).
        if any(blocked[r, c] for c, r in (obs.state_index(x) for x in loop)):
            continue
        winding = obs.path_winding(loop) / (2 * math.pi)
        counts = obs.abelianize(obs.lattice_path_word(loop))
        np.testing.assert_allclose(winding, counts, atol=1e-9)
        checked += 1
        nontrivial += bool(np.any(counts))
    assert checked >= 20 and nontrivial >= 5


def test_lattice_word_matches_polyline_word_on_random_walks():
    occ = make_grid([(8, 14, 8, 14), (24, 30, 8, 14), (16, 22, 18, 24)])
    obs = extract_obstacles(occ, min_area_m2=0.0)
    rng = np.random.default_rng(1)
    for _ in range(25):
        waypoints = [tuple(rng.integers(0, [40, 30])) for _ in range(6)]
        states = lattice_walk(obs, waypoints)
        assert obs.lattice_path_word(states) == obs.path_word(states)


def test_edge_and_segment_crossings_agree_at_ray_intersections():
    # Hand-placed rays that share lines and meet at lattice-cell corners: the cases where
    # the tie-breaking convention decides the order.
    rays = {
        1: Ray(1, 0, +1, line2=21, origin2=10),   # y = 10.5, x > 5
        2: Ray(2, 0, -1, line2=21, origin2=40),   # y = 10.5, x < 20 (overlaps ray 1)
        3: Ray(3, 1, +1, line2=25, origin2=4),    # x = 12.5, y > 2 (meets y = 10.5)
        4: Ray(4, 1, -1, line2=25, origin2=30),   # x = 12.5, y < 15 (overlaps ray 3)
    }
    obs = ObstacleSet(
        labels=np.zeros((30, 30), dtype=int), ids=[1, 2, 3, 4], component_of_id={},
        rep_cells={}, areas_m2={}, rays=rays, resolution=1.0, origin_x=0.0, origin_y=0.0,
    )
    checked_ties = 0
    for c, r in itertools.product(range(0, 26), range(0, 21)):
        for dc, dr in STEPS:
            exact = obs.edge_crossings(c, r, c + dc, r + dr)
            general = tuple(letter for _, letter in obs.segment_crossings((c, r), (c + dc, r + dr)))
            assert exact == general, ((c, r), (dc, dr))
            checked_ties += len(exact) > 1
    assert checked_ties > 0
