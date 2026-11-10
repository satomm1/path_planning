"""Capture deterministic outputs of the planners for refactor equivalence checks.

Usage (from anywhere; uses the repository this file lives in):

    python tests/regression/capture.py OUTDIR [--quick]

Writes one text file per captured item to OUTDIR plus ``_manifest.json`` (SHA-256 of
each item) and ``_environment.json``. Compare two captures, or a capture against the
committed baseline, with ``compare.py``. See ``tests/regression/README.md``.

Captured items: event-waypoint MICP analyses, problem matrices, printed formulations
and solutions (simultaneous / sequential / combined, synthetic and multi-robot pool
cases); dense-MILP problem data and z-assignment plots; occupancy grids from every
``build_occ_grid`` variant; random free points, path lengths, route keys; heatmap and
social-graph edges; A* paths and telemetry (compiled kernel and Python loop); schedule
figures and GIFs.
"""

import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse as sp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
DATA = Path(__file__).resolve().parent / "data" / "sample2_route_pool.json"

if len(sys.argv) < 2:
    sys.exit(__doc__)
OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
QUICK = "--quick" in sys.argv[2:]
os.chdir(REPO)
REC = {}


def h(b):
    return hashlib.sha256(b).hexdigest()


def put(name, text):
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    REC[name] = h(text.encode())
    with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)


def put_file(name, path):
    with open(path, "rb") as f:
        REC[name] = h(f.read())


@contextlib.contextmanager
def capture():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def problem_data_text(prob):
    """cvxpy problem matrices (as hashes), sizes and normalized constraint text."""
    data, _chain, _inv = prob.get_problem_data(cp.SCIPY)
    lines = []
    for k in sorted(data):
        if k == "param_prob":
            continue
        v = data[k]
        if sp.issparse(v):
            v = sp.csc_matrix(v)
            v.sort_indices()
            lines.append(f"{k}: sparse{v.shape} nnz={v.nnz} " + h(v.indptr.tobytes() + v.indices.tobytes() + v.data.tobytes()))
        elif isinstance(v, np.ndarray):
            lines.append(f"{k}: arr{v.shape} {v.dtype} " + h(np.ascontiguousarray(v).tobytes()))
        else:
            lines.append(f"{k}: {v!r}"[:2000])
    lines.append(f"n_constraints={len(prob.constraints)} n_vars={sum(v.size for v in prob.variables())}")
    ids = {}

    def norm(s):
        # cvxpy variable names carry a process-global counter; renumber by first use.
        return re.sub(r"var(\d+)", lambda m: "v%d" % ids.setdefault(m.group(1), len(ids)), s)

    lines.append("objective=" + norm(str(prob.objective)))
    lines.append("constraint_reprs_hash=" + h(norm("\n".join(str(c) for c in prob.constraints)).encode()))
    lines.append("var_shapes=" + repr([(v.shape, bool(v.attributes.get("boolean"))) for v in prob.variables()]))
    return "\n".join(lines)


def times_text(ts):
    return repr([[float(x) for x in t] for t in ts])


def attempt(fn):
    with capture() as buf:
        try:
            res = fn()
        except Exception as e:
            res = f"ERR {type(e).__name__}: {e}"
    return res, buf.getvalue()


# ---------------- inputs ----------------
with open(DATA, encoding="utf-8") as f:
    _fixture = json.load(f)
_res = _fixture["resolution"]
POOL = [None] * _fixture["pool_size"]
pool = [None] * _fixture["pool_size"]
for _idx, _route in _fixture["routes"].items():
    POOL[int(_idx)] = [(_res * a, _res * b) for a, b in _route["path"]]
    pool[int(_idx)] = {
        "x_init": [_res * v for v in _route["x_init"]],
        "x_goal": [_res * v for v in _route["x_goal"]],
    }


def line(p0, p1, step=0.2):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    n = int(round(np.linalg.norm(p1 - p0) / step))
    return [tuple(map(float, np.round(p0 + (p1 - p0) * k / n, 6))) for k in range(n + 1)]


def bend(*corners):
    pts = line(corners[0], corners[1])
    for a, b in zip(corners[1:-1], corners[2:]):
        pts += line(a, b)[1:]
    return pts


# Synthetic cases chosen to exercise every encounter branch (intersection / opposite /
# same direction, with a binary or with fixed start/goal ordering, leader a or b).
SYN = {
    "cross_a_early": [line((0, 5), (10, 5)), line((1, -4), (1, 6))],
    "headon": [line((0, 0), (10, 0)), line((10, 0.3), (0, 0.3))],
    "follow": [line((0, 0), (10, 0)), line((-3, 0.2), (8, 0.2))],
    "start_in_conflict": [line((0, 0), (10, 0)), line((5, 0.2), (5, 8))],
    "goal_in_conflict": [line((0, 0), (10, 0)), line((5, 8), (5, 0.2))],
    "three_way": [line((0, 5), (10, 5)), line((5, 0), (5, 10)), line((0, 0), (10, 10))],
    "merge_same": [bend((0, -5), (0, 0), (10, 0), (10, 5)), line((-5, 0.2), (14, 0.2))],
    "opp_goal_b": [bend((0, -5), (0, 0), (10, 0), (10, -5)), line((14, 0.3), (5, 0.3))],
    "opp_goal_a": [line((14, 0.3), (5, 0.3)), bend((0, -5), (0, 0), (10, 0), (10, -5))],
    "same_start_b": [bend((0, -5), (0, 0), (10, 0), (10, 5)), line((3, 0.2), (14, 0.2))],
}
rng = np.random.default_rng(0)
POOL_SETS = {}
for n in ([2, 3, 4, 6] if not QUICK else [3]):
    idx = rng.choice(len(POOL), size=n, replace=False)
    POOL_SETS[f"pool_n{n}"] = [POOL[i][::3] for i in idx]

import social_path_planning.event_multi_planning as em  # noqa: E402
import social_path_planning.multi_planning as mp  # noqa: E402
from social_path_planning import trajectory_viz as tv  # noqa: E402


def min_time(agent_path, v):
    t = [0.0]
    wps = agent_path.interest_waypoints
    for k in range(len(wps) - 1):
        t.append(t[-1] + max(0.0, wps[k + 1].s - wps[k].s) / v)
    return t


# ---------------- event MILP ----------------
VELS = [0.7, 0.58, 0.45, 0.7, 0.33, 0.6]
for name, paths in {**SYN, **POOL_SETS}.items():
    t0 = time.time()
    vel = VELS[:len(paths)]
    analysis = em.analyze_event_paths(paths)
    put(f"event_{name}_analysis", repr(analysis))
    reports = []
    for a in range(len(paths)):
        for b in range(a + 1, len(paths)):
            a1, a2, si, sj = mp.detect_collision_pairs_for_agent_pair(paths[a], paths[b], a, b, threshold=em.ROBOT_DIAMETER)
            reports.append({"a1": a1, "a2": a2, "segment_i": si, "segment_j": sj})
    put(f"event_{name}_analysis_from_reports", repr(em.analyze_event_paths_from_segment_reports(paths, reports)))
    for norm in (1, np.inf):
        pl = em.EventMultiAgentSimultaneousPlanner(None, paths=paths, norm=norm, v=vel)
        with capture() as buf:
            prob, ts, nz = pl.build_problem(analysis=analysis, print_constraints=True)
        put(f"event_{name}_sim_norm{norm}_print", buf.getvalue())
        put(f"event_{name}_sim_norm{norm}_data", problem_data_text(prob) + f"\nnum_z={nz} mutex={pl.mutex_constraint_count}")
        if norm == 1:
            res, out = attempt(lambda: times_text(pl.plan()))
            put(f"event_{name}_sim_solution", res + "\n" + out)
            res, out = attempt(lambda: times_text(pl.plan_from_analysis(analysis)))
            put(f"event_{name}_sim_solution_from_analysis", res + "\n" + out)
    seq = em.EventMultiAgentSequentialPlanner(None, other_agent_paths=paths[1:], other_agent_times=[[0.0]] * (len(paths) - 1), path=paths[0], v=vel[0])
    sa = seq.analyze()
    seq.other_agent_times = [min_time(sa.agents[k], vel[k]) for k in range(1, len(paths))]
    with capture() as buf:
        prob, ts, nz = seq.build_problem(analysis=sa, print_constraints=True)
    put(f"event_{name}_seq_print", buf.getvalue())
    put(f"event_{name}_seq_data", problem_data_text(prob) + f"\nnum_z={nz} mutex={seq.mutex_constraint_count}")
    res, out = attempt(lambda: times_text([seq.plan()]))
    put(f"event_{name}_seq_solution", res + "\n" + out)
    if len(paths) >= 3:
        cb = em.EventMultiAgentCombinedPlanner(None, other_agent_paths=paths[:1], other_agent_times=[[0.0]], paths=paths[1:], v=vel[1:], norm=1)
        ca = cb.analyze()
        cb.other_agent_times = [min_time(ca.agents[0], em.MAX_VELOCITY)]
        with capture() as buf:
            prob, ts, nz = cb.build_problem(analysis=ca, print_constraints=True)
        put(f"event_{name}_comb_print", buf.getvalue())
        put(f"event_{name}_comb_data", problem_data_text(prob) + f"\nnum_z={nz} mutex={cb.mutex_constraint_count}")
        res, out = attempt(lambda: times_text(cb.plan()))
        put(f"event_{name}_comb_solution", res + "\n" + out)
    print(f"event {name} done {time.time() - t0:.1f}s", file=sys.stderr)

put("event_err_sim_nopaths", attempt(lambda: em.EventMultiAgentSimultaneousPlanner(None).plan())[0])
put("event_err_seq_nopath", attempt(lambda: em.EventMultiAgentSequentialPlanner(None, [], []).plan())[0])
put("event_err_comb_nopaths", attempt(lambda: em.EventMultiAgentCombinedPlanner(None, [], []).plan())[0])
put("event_err_comb_empty", attempt(lambda: em.EventMultiAgentCombinedPlanner(None, [], [], paths=[]).build_problem(analysis=em.analyze_event_paths([])))[0])


# ---------------- dense MILP: problem data only (stop before solving) ----------------
class _Stop(Exception):
    pass


_captured = {}
_orig_solve = cp.Problem.solve


def _fake_solve(self, *a, **k):
    _captured["prob"] = self
    _captured["kw"] = repr(sorted((kk, str(vv)) for kk, vv in k.items()))
    raise _Stop()


cp.Problem.solve = _fake_solve
_viz_flag = mp.VIZ_MILP_Z_ASSIGNMENT
mp.VIZ_MILP_Z_ASSIGNMENT = True  # also cover the z-assignment plots
mp.VIZ_MILP_Z_OUTPUT_DIR = os.path.join(OUT, "milpz")
dense_sets = {"syn_three_way": SYN["three_way"], "syn_headon": SYN["headon"]}
if not QUICK:
    dense_sets["pool_n3"] = [p[::2] for p in POOL_SETS["pool_n3"]]
for name, paths in dense_sets.items():
    shutil.rmtree(mp.VIZ_MILP_Z_OUTPUT_DIR, ignore_errors=True)
    vel = VELS[:len(paths)]
    for kind in ("sim", "seq", "comb"):
        if kind == "sim":
            pl = mp.MultiAgentSimultaneousPlanner(None, paths=paths, norm=1)
            pl.assign_velocities(list(vel))
        elif kind == "seq":
            others = paths[1:]
            ot = [[i * 0.2 / 0.35 for i in range(len(p))] for p in others]
            pl = mp.MultiAgentSequentialPlanner(None, others, ot, path=paths[0], v=vel[0])
        else:
            if len(paths) < 3:
                continue
            ot = [[i * 0.2 / 0.45 for i in range(len(paths[0]))]]
            pl = mp.MultiAgentCombinedPlanner(None, paths[:1], ot, paths=paths[1:], v=list(vel[1:]))
        with capture() as buf:
            try:
                pl.plan()
                res = "no-solve?"
            except _Stop:
                res = problem_data_text(_captured["prob"]) + "\nsolve_kwargs=" + _captured["kw"]
        out = "\n".join(l for l in buf.getvalue().splitlines() if not l.startswith("Saved MILP z visualization"))
        put(f"dense_{name}_{kind}", res + "\n--stdout--\n" + out)
        if kind == "sim" and os.path.isdir(mp.VIZ_MILP_Z_OUTPUT_DIR):
            for fn in sorted(os.listdir(mp.VIZ_MILP_Z_OUTPUT_DIR)):
                put_file(f"dense_{name}_vizpng_{fn}", os.path.join(mp.VIZ_MILP_Z_OUTPUT_DIR, fn))
        if kind == "sim":
            put(f"dense_{name}_timing_records_keys", repr([(r["robot_i"], r["robot_j"], r["collision_tuple_count"], r["constraint_count"]) for r in pl.constraint_timing_records]))
cp.Problem.solve = _orig_solve
mp.VIZ_MILP_Z_ASSIGNMENT = _viz_flag
mc, mz = mp.merge_collision_reports(SYN["three_way"], [
    dict(zip(("a1", "a2", "segment_i", "segment_j"), mp.detect_collision_pairs_for_agent_pair(SYN["three_way"][a], SYN["three_way"][b], a, b)))
    for a, b in ((0, 1), (0, 2), (1, 2))])
put("dense_merge_reports", repr((mc, mz)))
put("dense_subsample", repr([mp.subsample_path_by_stride(POOL[0], s) for s in (1, 2, 7)]))

# ---------------- grids / helpers ----------------
from social_path_planning import benchmark_mapf_ensemble as bme  # noqa: E402
from social_path_planning import benchmark_sparse_snapshots as bss  # noqa: E402
from social_path_planning import compare_astar as ca_mod  # noqa: E402
from social_path_planning.common import wall_distance_cache_path  # noqa: E402


def grid_text(g):
    return "\n".join([
        f"res={g.resolution} w={g.width} h={g.height} ox={g.origin_x} oy={g.origin_y} robot_d={g.robot_d} thresh={g.thresh}",
        "probs=" + h(np.ascontiguousarray(g.probs).tobytes()),
        "dmap=" + h(np.ascontiguousarray(g.distance_map).tobytes()),
        "d_right=" + ("None" if g._d_right is None else h(np.ascontiguousarray(g._d_right).tobytes())),
    ])


for scen in (["sample2_default", "y2e2"] if not QUICK else ["sample2_default"]):
    for label, fn in (("compare_astar", ca_mod.build_occ_grid), ("bss", bss.build_occ_grid), ("bme", bme.build_occ_grid)):
        with capture() as buf:
            g, ms, mr, hi = fn(scen)
        put(f"grid_{scen}_{label}", grid_text(g) + f"\nmap_size={ms} res={mr} hi={hi}\n--stdout--\n" + buf.getvalue().replace(str(REPO), "<REPO>"))
occ_grid, map_size, map_res, hi = ca_mod.build_occ_grid("sample2_default")
for label, rfn, lfn in (("compare_astar", ca_mod.generate_random_free_point, ca_mod.compute_path_length),
                        ("bss", bss.generate_random_free_point, bss.compute_path_length)):
    r = np.random.default_rng(1)
    put(f"randpts_{label}", repr([rfn(occ_grid, r) for _ in range(25)]))
    put(f"pathlen_{label}", repr([lfn(p) for p in POOL[:20]]))
put("route_keys", repr([ca_mod.route_pair_key(r["x_init"], r["x_goal"]) for r in pool[:5]]))
from social_path_planning.heat_map import HeatMap2DVector  # noqa: E402

hm = HeatMap2DVector(occ_grid)
for p in POOL[:60]:
    hm.add_path(p)
put("heatmap_from_pool", h(np.ascontiguousarray(hm.heatmap).tobytes()))
hm_path = os.path.join(OUT, "_heatmap_tmp.npy")
np.save(hm_path, hm.heatmap)
with capture() as buf:
    G = ca_mod.build_sparse_graph_from_heatmap(occ_grid, heatmap_path=hm_path, sparse_graph_threshold=2.0, sparse_min_component_size=15)
put("sparse_graph_edges", repr(list(G.edges(data=True))) + "\n" + repr(list(G.nodes())) + "\n--stdout--\n" + buf.getvalue().replace(hm_path, "<HM>"))
import matplotlib.pyplot as plt  # noqa: E402

plt.close("all")

# ---------------- A* (compiled kernel and Python loop) ----------------
from social_path_planning import astar_kernel  # noqa: E402
from social_path_planning.a_star import AStar, AStar_With_Graph  # noqa: E402
from social_path_planning.utils import snap_to_grid  # noqa: E402


def astar_text(p, ok):
    tel = dict(p.last_solve_telemetry)
    tel.pop("solve_time_s", None)
    return "\n".join([
        f"ok={ok}", "path=" + repr(p.path), f"closed={len(p.closed_set)} came_from={len(p.came_from)}",
        "telemetry=" + repr(sorted(tel.items())),
        "smoothed=" + h(repr([(float(a), float(b)) for a, b in (p.smoothed_path or [])]).encode()),
    ])


a, b = snap_to_grid([75, 2.5], map_res), snap_to_grid([50, 50], map_res)
for compiled in (True, False):
    suffix = "" if compiled else "_python"
    for mode in (["vanilla", "modified"] if not QUICK else ["vanilla"]):
        p = AStar([0, 0], hi, a, b, occ_grid, resolution=map_res)
        p.use_compiled = compiled
        with capture():
            ok = p.solve(mode=mode)
        put(f"astar_{mode}{suffix}", astar_text(p, ok))
    p = AStar_With_Graph([0, 0], hi, a, b, occ_grid, G, resolution=map_res)
    p.use_compiled = compiled
    with capture():
        ok = p.solve(mode="modified")
    put(f"astar_with_graph{suffix}", astar_text(p, ok))

# ---------------- viz ----------------
vdir = os.path.join(OUT, "viz")
shutil.rmtree(vdir, ignore_errors=True)
os.makedirs(vdir)
paths = SYN["three_way"]
pl = em.EventMultiAgentSimultaneousPlanner(None, paths=paths, v=[0.7, 0.5, 0.6])
with capture():
    et = pl.plan()
ap, at = em.build_constant_velocity_schedules(pl.analysis, et)
put("viz_cv_schedules", repr((ap, at)))
put("viz_expand_times", repr([em.expand_event_times_to_original_path(ag, t) for ag, t in zip(pl.analysis.agents, et)]))
with capture():
    tv.create_space_time_plot(ap, at, output_file=os.path.join(vdir, "st.png"), title="T")
    tv.create_map_context_plot(ap, occ_grid=occ_grid, times=at, snapshot_time=5.0, output_file=os.path.join(vdir, "mc.png"), title="M")
    tv.create_map_context_plot(ap, occ_grid=None, times=None, output_file=os.path.join(vdir, "mc2.png"))
    tv.viz_event_waypoints(pl.analysis, occ_grid=None, output_path=os.path.join(vdir, "wp.png"), also_write_pair_panels=True)
    if not QUICK:
        tv.create_map_context_video(ap, at, occ_grid=None, output_file=os.path.join(vdir, "mcv.gif"), total_frames=6)
        tv.create_event_video(pl.analysis, et, output_file=os.path.join(vdir, "ev.gif"), occ_grid=None, title="E")
        tv.create_video(ap, at, output_file=os.path.join(vdir, "v.gif"))
put("viz_get_position", repr([tv.get_position_at_time(t, ap[0], at[0]) for t in np.linspace(-1, 30, 25)]))
for fn in sorted(os.listdir(vdir)):
    put_file("vizfile_" + fn, os.path.join(vdir, fn))

# ---------------- write ----------------
with open(os.path.join(OUT, "_manifest.json"), "w") as f:
    json.dump(REC, f, indent=1, sort_keys=True)


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


env = {
    "quick": QUICK,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "cvxpy": cp.__version__,
    "cvxpy_solvers": sorted(cp.installed_solvers()),
    "matplotlib": matplotlib.__version__,
    "networkx": _version("networkx"),
    "numba": _version("numba"),
    "astar_kernel": bool(astar_kernel.kernel_ready()),
    "y2e2_wall_distance_cache": wall_distance_cache_path("y2e2").is_file(),
    "inputs_sha256": h(DATA.read_bytes()),
}
with open(os.path.join(OUT, "_environment.json"), "w") as f:
    json.dump(env, f, indent=1, sort_keys=True)
print(f"captured {len(REC)} items -> {OUT}", file=sys.stderr)
