"""Milestone 2 of HOMOTOPY_PLAN.md: class-aware social A*."""
import math

import networkx as nx
import numpy as np
import pytest

from social_path_planning.a_star import AStar, AStar_With_Graph
from social_path_planning.homotopy import blocked_mask
from social_path_planning.homotopy_astar import (
    HomotopyAStar,
    HomotopyAStar_With_Graph,
    OverlapMerger,
    overlap_fraction,
)

from test_homotopy import RES, make_grid


def cell(c, r):
    return (c * RES, r * RES)


def planner(occ, start, goal, cls=HomotopyAStar, **kwargs):
    kwargs.setdefault("min_area_m2", 0.0)
    hi = [occ.width * RES, occ.height * RES]
    return cls([0, 0], hi, cell(*start), cell(*goal), occ, resolution=RES, **kwargs)


def assert_valid_alternative(p, alt):
    obs = p.obstacles
    blocked = blocked_mask(p.occupancy, p.statespace_lo, p.statespace_hi)
    assert alt.path[0] == p.x_init and alt.path[-1] == p.x_goal
    for x1, x2 in zip(alt.path[:-1], alt.path[1:]):
        (c1, r1), (c2, r2) = obs.state_index(x1), obs.state_index(x2)
        assert max(abs(c2 - c1), abs(r2 - r1)) == 1
        assert not blocked[r2, c2]
    # The word the search carried equals the word recomputed from the path, and the
    # closed-form winding equals the winding measured along it.
    assert obs.lattice_path_word(alt.path) == alt.word
    np.testing.assert_allclose(alt.winding, obs.path_winding(alt.path), atol=1e-9)
    assert np.all(np.abs(alt.winding) < 2 * math.pi)


@pytest.fixture
def one_block():
    return make_grid([(15, 25, 12, 18)])


@pytest.fixture
def two_stacked():
    return make_grid([(15, 25, 8, 14), (15, 25, 24, 30)], width=40, height=40)


# ----------------------------------------------------------------------
# Test 1: one obstacle
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["vanilla", "modified"])
def test_one_obstacle_gives_two_classes(one_block, mode):
    p = planner(one_block, (5, 15), (35, 15), k=5, alpha=2.0)
    alts = p.solve_alternatives(mode)
    assert len(alts) == 2  # above and below; looping classes are filtered by winding
    assert len({a.word for a in alts}) == 2
    assert {len(a.word) for a in alts} == {0, 1}
    assert [a.cost for a in alts] == sorted(a.cost for a in alts)
    for alt in alts:
        assert_valid_alternative(p, alt)
    assert p.path == alts[0].path  # solve() leaves the best class as .path


def test_loops_appear_only_without_winding_filter(one_block):
    p = planner(one_block, (5, 15), (35, 15), k=3, alpha=3.0, max_abs_winding=None)
    alts = p.solve_alternatives("vanilla")
    assert len(alts) == 3
    # The third class loops once around the obstacle.
    assert np.max(np.abs(alts[2].winding)) >= 2 * math.pi
    assert all(np.max(np.abs(a.winding)) < 2 * math.pi for a in alts[:2])


def test_k_caps_the_number_of_classes(one_block):
    p = planner(one_block, (5, 15), (35, 15), k=1, alpha=2.0)
    assert len(p.solve_alternatives("vanilla")) == 1


def test_target_and_blocked_words(one_block):
    both = planner(one_block, (5, 15), (35, 15), k=5, alpha=2.0).solve_alternatives("vanilla")
    second = both[1].word

    targeted = planner(one_block, (5, 15), (35, 15), target_word=second, alpha=100.0)
    alts = targeted.solve_alternatives("vanilla")
    assert [a.word for a in alts] == [second]
    assert alts[0].length_m == pytest.approx(both[1].length_m)

    blocked = planner(one_block, (5, 15), (35, 15), k=5, alpha=2.0, blocked_words=[both[0].word])
    assert [a.word for a in blocked.solve_alternatives("vanilla")] == [second]


# ----------------------------------------------------------------------
# Test 2: two stacked obstacles
# ----------------------------------------------------------------------

def test_two_stacked_obstacles_give_three_simple_classes(two_stacked):
    p = planner(two_stacked, (3, 19), (36, 19), k=10, alpha=0.6)
    alts = p.solve_alternatives("vanilla")
    assert len(alts) == 3  # between, below both, above both
    assert alts[0].word == ()  # straight through the gap is shortest
    assert sorted(len(a.word) for a in alts) == [0, 1, 1]
    assert len({abs(a.word[0]) for a in alts if a.word}) == 2  # one per obstacle
    for alt in alts:
        assert_valid_alternative(p, alt)


def test_two_stacked_obstacles_social_mode(two_stacked):
    p = planner(two_stacked, (3, 19), (36, 19), k=10, alpha=100.0)
    alts = p.solve_alternatives("modified")
    assert len(alts) >= 3
    assert {(), } <= {a.word for a in alts}
    for alt in alts:
        assert_valid_alternative(p, alt)
        assert alt.cost >= alt.length_m - 1e-9  # social cost only adds to distance


# ----------------------------------------------------------------------
# Test 4: equivalence with the paper's planner when no obstacle is retained
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample2():
    from social_path_planning.common import build_occ_grid

    occ, _, res, hi = build_occ_grid("sample2_default", wall_cache="none")
    return occ, res, hi


@pytest.mark.parametrize("mode", ["modified", "vanilla"])
def test_matches_astar_without_obstacles(sample2, mode):
    occ, res, hi = sample2
    start, goal = (2.0, 50.0), (30.0, 50.0)
    base = AStar([0, 0], hi, start, goal, occ, resolution=res)
    base.use_compiled = False
    assert base.solve(mode=mode)
    p = HomotopyAStar([0, 0], hi, start, goal, occ, resolution=res, min_area_m2=math.inf)
    assert p.obstacles.num_obstacles == 0
    assert p.solve(mode=mode)
    assert len(p.alternatives) == 1 and p.alternatives[0].word == ()
    assert p.path == base.path


def test_matches_astar_with_graph_without_obstacles(sample2):
    occ, res, hi = sample2
    start, goal = (2.0, 50.0), (30.0, 50.0)
    # A small social graph: the edges of a plain social A* path, plus a detour-free
    # copy shifted one row, so the graph cost (Eq. 4) actually changes the search.
    seed = AStar([0, 0], hi, start, goal, occ, resolution=res)
    assert seed.solve()
    graph = nx.DiGraph()
    for x1, x2 in zip(seed.path[:-1], seed.path[1:]):
        i1, i2 = seed.get_index(x1), seed.get_index(x2)
        graph.add_edge(i1, i2, weight=seed.distance(x1, x2))
        graph.add_edge((i1[0], i1[1] - 1), (i2[0], i2[1] - 1), weight=seed.distance(x1, x2))

    base = AStar_With_Graph([0, 0], hi, start, goal, occ, graph, resolution=res)
    base.use_compiled = False
    assert base.solve()
    p = HomotopyAStar_With_Graph([0, 0], hi, start, goal, occ, graph, resolution=res, min_area_m2=math.inf)
    assert p.solve()
    assert p.path == base.path
    assert p.alternatives[0].fraction_on_graph == pytest.approx(
        base.last_solve_telemetry["path_fraction_on_graph"]
    )


def test_sample2_obstacles_retained_and_paths_valid(sample2):
    occ, res, hi = sample2
    # Across the 2x2 block layout: several staircase routes within alpha.
    p = HomotopyAStar([0, 0], hi, (2.0, 2.0), (40.0, 52.0), occ, resolution=res, k=3, alpha=0.3)
    assert p.obstacles.num_obstacles == 4
    alts = p.solve_alternatives("vanilla")
    assert len(alts) >= 2
    for alt in alts:
        assert_valid_alternative(p, alt)


# ----------------------------------------------------------------------
# Overlap merge filter
# ----------------------------------------------------------------------

def test_overlap_fraction_basics():
    merger = OverlapMerger(20, 10, threshold=0.5, distance_m=0.0, resolution=RES)
    row = [(c, 5) for c in range(10)]
    merger.keep(row)
    mask = merger.masks[0]
    assert overlap_fraction(row, mask) == 1.0
    assert overlap_fraction([(c, 8) for c in range(10)], mask) == 0.0
    # Two of four unit segments touch the kept path (the first via its start vertex).
    half = [(9, 5), (10, 5), (11, 5), (12, 5), (13, 5)]
    assert overlap_fraction(half, mask) == pytest.approx(0.25)
    # The proximity radius widens the mask: within 0.5 m = 2.5 cells of the row.
    wide = OverlapMerger(20, 10, threshold=0.5, distance_m=0.5, resolution=RES)
    wide.keep(row)
    assert overlap_fraction([(c, 7) for c in range(10)], wide.masks[0]) == 1.0
    assert overlap_fraction([(c, 8) for c in range(10)], wide.masks[0]) == 0.0


def test_merge_threshold_above_one_changes_nothing(two_stacked):
    plain = planner(two_stacked, (3, 19), (36, 19), k=10, alpha=2.0).solve_alternatives("vanilla")
    p = planner(two_stacked, (3, 19), (36, 19), k=10, alpha=2.0, merge_overlap=1.01)
    alts = p.solve_alternatives("vanilla")
    assert [(a.word, a.cost, a.path) for a in alts] == [(a.word, a.cost, a.path) for a in plain]
    assert p.merged_alternatives == []
    assert alts[0].max_overlap == 0.0
    assert all(0.0 < a.max_overlap < 1.0 for a in alts[1:])


def test_merged_classes_do_not_count_toward_k(two_stacked):
    # Threshold 0 merges everything after the first class into it.
    p = planner(two_stacked, (3, 19), (36, 19), k=2, alpha=2.0, merge_overlap=0.0)
    alts = p.solve_alternatives("vanilla")
    assert len(alts) == 1
    assert len(p.merged_alternatives) >= 2  # the search kept going past k
    assert all(m.merged_into == 0 for m in p.merged_alternatives)
    for alt in alts + p.merged_alternatives:
        assert_valid_alternative(p, alt)


def test_merge_keeps_the_cheaper_of_two_near_duplicates(two_stacked):
    plain = planner(two_stacked, (3, 19), (36, 19), k=10, alpha=2.0, merge_overlap=1.01)
    plain_alts = plain.solve_alternatives("vanilla")
    overlaps = [a.max_overlap for a in plain_alts]
    threshold = max(overlaps[1:])  # merge exactly the most redundant class
    p = planner(two_stacked, (3, 19), (36, 19), k=10, alpha=2.0, merge_overlap=threshold)
    alts = p.solve_alternatives("vanilla")
    redundant = plain_alts[overlaps.index(threshold)].word
    assert redundant not in [a.word for a in alts]
    assert redundant in [a.word for a in p.merged_alternatives]


# ----------------------------------------------------------------------
# Compiled search (homotopy_kernel) == Python loop
# ----------------------------------------------------------------------

TEL_KEYS = ["cost_eval_count", "social_cost_sum", "social_cost_max", "social_nonzero_edges",
            "graph_edge_cost_evals", "off_graph_social_evals", "graph_weight_sum", "off_graph_edge_cost_sum"]

KERNEL_CASES = [
    ("one", (5, 15), (35, 15), dict(k=5, alpha=2.0)),
    ("one", (5, 15), (35, 15), dict(k=3, alpha=3.0, max_abs_winding=None)),
    ("two", (3, 19), (36, 19), dict(k=10, alpha=2.0)),
    ("two", (3, 19), (36, 19), dict(target_word=(1,), alpha=100.0)),
    ("two", (3, 19), (36, 19), dict(k=10, alpha=2.0, blocked_words=[()])),
    ("two", (3, 19), (36, 19), dict(k=10, alpha=2.0, max_word_len=0)),
    ("two", (3, 19), (36, 19), dict(k=10, alpha=3.0, merge_overlap=0.5)),
    ("two", (3, 19), (36, 19), dict(k=10, alpha=3.0, merge_overlap=0.0, merge_distance_m=1.0)),
    ("one", (5, 15), (35, 15), dict(k=3, alpha=3.0, max_abs_winding=None, merge_overlap=0.7)),
]


def _kernel_available():
    from social_path_planning import homotopy_kernel

    return homotopy_kernel.kernel_ready()


@pytest.mark.parametrize("mode", ["vanilla", "modified"])
@pytest.mark.parametrize("grid_name,start,goal,kwargs", KERNEL_CASES)
def test_compiled_search_matches_python(one_block, two_stacked, grid_name, start, goal, kwargs, mode):
    if not _kernel_available():
        pytest.skip("numba kernel unavailable")
    grid = one_block if grid_name == "one" else two_stacked
    runs = []
    for compiled in (False, True):
        p = planner(grid, start, goal, **kwargs)
        p.use_compiled = compiled
        if compiled:
            assert p._homotopy_compiled_inputs(mode) is not None
        _, tel = p.solve(mode=mode, return_telemetry=True)
        runs.append((p, tel))
    (pp, telp), (pc, telc) = runs
    assert pc.used_compiled and not pp.used_compiled

    def summary(alts):
        return [(a.word, a.cost, a.path, a.merged_into, a.max_overlap) for a in alts]

    assert summary(pc.alternatives) == summary(pp.alternatives)
    assert summary(pc.merged_alternatives) == summary(pp.merged_alternatives)
    assert pc.explored_states == pp.explored_states
    assert telc["explored_nodes"] == telp["explored_nodes"]
    for key in TEL_KEYS:
        assert telc[key] == telp[key], key


def test_compiled_graph_search_matches_python(sample2):
    if not _kernel_available():
        pytest.skip("numba kernel unavailable")
    occ, res, hi = sample2
    start, goal = (2.0, 2.0), (40.0, 52.0)
    seed = AStar([0, 0], hi, start, goal, occ, resolution=res)
    assert seed.solve()
    graph = nx.DiGraph()
    for x1, x2 in zip(seed.path[:-1], seed.path[1:]):
        i1, i2 = seed.get_index(x1), seed.get_index(x2)
        graph.add_edge(i1, i2, weight=seed.distance(x1, x2))
    runs = []
    for compiled in (False, True):
        p = HomotopyAStar_With_Graph([0, 0], hi, start, goal, occ, graph, resolution=res, k=3, alpha=1.0)
        p.use_compiled = compiled
        _, tel = p.solve(return_telemetry=True)
        runs.append((p, tel))
    (pp, telp), (pc, telc) = runs
    assert len(pc.alternatives) >= 1
    assert [(a.word, a.cost, a.path) for a in pc.alternatives] == [
        (a.word, a.cost, a.path) for a in pp.alternatives
    ]
    for key in TEL_KEYS:
        assert telc[key] == telp[key], key
