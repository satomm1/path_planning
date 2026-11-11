"""Survey: how many homotopy-class alternatives each map offers, and what they cost.

For every scenario this samples random start/goal pairs (deterministic per seed and
scenario) and, for each pair:

* runs the paper's social A* (``AStar``) once as the timing and path baseline;
* runs the class-aware search (``HomotopyAStar``) with ``alpha = max(--alphas)``;
* counts the classes within ``(1 + alpha) x best`` for each requested alpha. One search
  answers every alpha: classes come out in cost order and the bound only prunes classes
  above the largest alpha.

Outputs (default ``results/homotopy_survey/<tag>/``):

* ``rows.jsonl``: one JSON row per (scenario, pair): classes with word, cost, length,
  social penalty and overlap; timings; explored states; the baseline's class.
* ``summary.json`` / ``summary.csv``: per scenario and alpha, the class-count distribution
  and timing statistics.
* ``class_counts.png``, ``timing.png``.

``--resume`` skips pairs already in ``rows.jsonl`` (the settings must match ``meta.json``;
raising ``--pairs`` extends the same pair sequence). Long runs belong in the background.
Use the ``path_planning`` conda env so the compiled searches are used.

Example::

    python -m social_path_planning.survey_homotopy_classes --pairs 30 --alphas 0.1,0.3,0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np

from social_path_planning.common import WALL_CACHE_POLICIES, build_occ_grid, generate_random_free_point

DEFAULT_SCENARIOS = (
    "sample1_default", "sample2_default", "sample3_default", "sample4_default", "sample5_default",
    "circular_default", "hall_with_room_default", "y2e2_aligned",
)
DEFAULT_OUTPUT_ROOT = Path("results/homotopy_survey")
# Settings that must match for --resume (pairs may grow).
META_KEYS = ("seed", "min_separation", "k", "alphas", "min_area", "merge_overlap", "merge_distance",
             "mode", "wall_cache", "time_limit", "baseline", "heatmaps")

# Chart colors (validated with the dataviz palette checks): ordinal blue ramp for class
# counts 1, 2, 3, >=4, and categorical slots 1-2 for the two timing series.
COUNT_COLORS = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
SERIES_COLORS = {"homotopy": "#2a78d6", "baseline": "#eb6834"}
SURFACE = "#fcfcfb"
TEXT = "#2b2b28"
MUTED = "#6b6a63"
GRID = "#e4e3de"


# ----------------------------------------------------------------------
# Pairs and planning
# ----------------------------------------------------------------------

def scenario_pairs(occ, scenario, seed, count, min_separation):
    """``count`` free start/goal pairs; a larger count extends the same sequence."""
    rng = np.random.default_rng([int(seed), zlib.crc32(scenario.encode("utf-8"))])
    pairs = []
    while len(pairs) < count:
        start = generate_random_free_point(occ, rng)
        goal = generate_random_free_point(occ, rng)
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) >= min_separation:
            pairs.append((start, goal))
    return pairs


def survey_pair(occ, res, statespace_hi, start, goal, settings, obstacles=None, graph=None):
    """Plan one pair with the baseline and the class-aware search; return a JSON row.

    With a social ``graph`` both use the graph-biased cost (Eq. 4): ``AStar_With_Graph`` as
    the baseline and ``HomotopyAStar_With_Graph`` for the classes.
    """
    from social_path_planning.a_star import AStar, AStar_With_Graph
    from social_path_planning.homotopy_astar import HomotopyAStar, HomotopyAStar_With_Graph

    lo = [0, 0]
    row = {"start": [float(v) for v in start], "goal": [float(v) for v in goal]}
    if settings["baseline"]:
        if graph is None:
            base = AStar(lo, statespace_hi, start, goal, occ, resolution=res)
        else:
            base = AStar_With_Graph(lo, statespace_hi, start, goal, occ, graph, resolution=res)
        t0 = time.time()
        ok = bool(base.solve(mode=settings["mode"]))
        row["baseline"] = {"success": ok, "time_s": time.time() - t0}
        if ok:
            row["baseline"]["length_m"] = float(sum(
                base.distance(a, b) for a, b in zip(base.path[:-1], base.path[1:])
            ))
            if graph is not None:
                row["baseline"]["fraction_on_graph"] = base.last_solve_telemetry.get("path_fraction_on_graph")
                row["baseline"]["explored_states"] = int(base.last_solve_telemetry["explored_nodes"])
                # Plain social A* on the same pair, in the same process, so the effect of H on
                # planning time is a paired comparison (timings across separate runs vary with
                # machine load).
                plain = AStar(lo, statespace_hi, start, goal, occ, resolution=res)
                t0 = time.time()
                plain.solve(mode=settings["mode"])
                row["baseline_no_graph"] = {
                    "time_s": time.time() - t0,
                    "explored_states": int(plain.last_solve_telemetry["explored_nodes"]),
                }
            base_path = base.path
        else:
            return row  # unreachable pair: the class-aware search can't succeed either
    else:
        base_path = None

    planner_args = (lo, statespace_hi, start, goal, occ) + (() if graph is None else (graph,))
    planner_cls = HomotopyAStar if graph is None else HomotopyAStar_With_Graph
    planner = planner_cls(
        *planner_args, resolution=res,
        obstacles=obstacles, min_area_m2=settings["min_area"],
        k=settings["k"], alpha=max(settings["alphas"]),
        merge_overlap=settings["merge_overlap"], merge_distance_m=settings["merge_distance"],
        time_limit_s=settings["time_limit"],
    )
    t0 = time.time()
    planner.solve(mode=settings["mode"])
    elapsed = time.time() - t0
    row["homotopy"] = {
        "success": bool(planner.alternatives),
        "time_s": elapsed,
        "engine": "compiled" if planner.used_compiled else "python",
        "explored_states": int(planner.explored_states),
        "timed_out": bool(not planner.alternatives and elapsed >= settings["time_limit"]),
        "classes": [
            {"word": list(a.word), "cost": float(a.cost), "length_m": float(a.length_m),
             "social_penalty": float(a.social_penalty), "max_overlap": float(a.max_overlap),
             "fraction_on_graph": None if a.fraction_on_graph is None else float(a.fraction_on_graph)}
            for a in planner.alternatives
        ],
        "merged": len(planner.merged_alternatives),
    }
    if base_path is not None:
        row["baseline"]["word"] = list(planner.obstacles.lattice_path_word(base_path))
    return row


def classes_within(row, alpha):
    classes = (row.get("homotopy") or {}).get("classes") or []
    if not classes:
        return 0
    bound = (1.0 + alpha) * classes[0]["cost"]
    return sum(1 for c in classes if c["cost"] <= bound)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

def _stats(values):
    if not values:
        return {"median": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {"median": float(np.median(arr)), "p90": float(np.percentile(arr, 90)), "max": float(arr.max())}


def summarize(rows_by_scenario, obstacles_by_scenario, settings):
    summary = {}
    k = settings["k"]
    for scenario, rows in rows_by_scenario.items():
        solved = [r for r in rows if (r.get("homotopy") or {}).get("success")]
        entry = {
            "pairs": len(rows),
            "solved": len(solved),
            "unreachable": sum(1 for r in rows if r.get("baseline") and not r["baseline"]["success"]),
            "timed_out": sum(1 for r in rows if (r.get("homotopy") or {}).get("timed_out")),
            "num_obstacles": obstacles_by_scenario.get(scenario),
            "python_engine_runs": sum(1 for r in solved if r["homotopy"]["engine"] != "compiled"),
            "alphas": {},
        }
        for alpha in settings["alphas"]:
            counts = [classes_within(r, alpha) for r in solved]
            hist = {str(n): counts.count(n) for n in range(1, k + 1)}
            entry["alphas"][f"{alpha:g}"] = {
                "mean_classes": float(np.mean(counts)) if counts else None,
                "median_classes": float(np.median(counts)) if counts else None,
                "frac_ge2": float(np.mean([c >= 2 for c in counts])) if counts else None,
                "frac_ge3": float(np.mean([c >= 3 for c in counts])) if counts else None,
                "frac_at_k": float(np.mean([c >= k for c in counts])) if counts else None,
                "histogram": hist,
            }
        h_times = [r["homotopy"]["time_s"] for r in solved]
        entry["homotopy_time_s"] = _stats(h_times)
        entry["explored_states"] = _stats([r["homotopy"]["explored_states"] for r in solved])
        if settings["baseline"]:
            b_times = [r["baseline"]["time_s"] for r in solved]
            entry["baseline_time_s"] = _stats(b_times)
            entry["time_ratio"] = _stats([h / max(b, 1e-6) for h, b in zip(h_times, b_times)])
            same = [r["baseline"].get("word") == r["homotopy"]["classes"][0]["word"] for r in solved]
            entry["best_class_is_baseline_class"] = float(np.mean(same)) if same else None
            # Cost of the 2nd class relative to the best, where one exists.
            ratios = [r["homotopy"]["classes"][1]["cost"] / r["homotopy"]["classes"][0]["cost"]
                      for r in solved if len(r["homotopy"]["classes"]) >= 2]
            entry["second_class_cost_ratio"] = _stats(ratios)
        # Effect of H on the baseline, paired per pair within this run (graph-biased runs only).
        paired = [r for r in solved if r.get("baseline_no_graph")]
        if paired:
            entry["baseline_no_graph_time_s"] = _stats([r["baseline_no_graph"]["time_s"] for r in paired])
            entry["graph_speedup"] = _stats([r["baseline_no_graph"]["time_s"] / max(r["baseline"]["time_s"], 1e-6)
                                             for r in paired])
            entry["graph_explored_ratio"] = _stats([r["baseline"]["explored_states"]
                                                    / max(r["baseline_no_graph"]["explored_states"], 1)
                                                    for r in paired])
            entry["graph_faster_fraction"] = float(np.mean([r["baseline"]["time_s"] < r["baseline_no_graph"]["time_s"]
                                                            for r in paired]))
        # Share of each path on the social graph H (graph-biased runs only).
        first_on_h = [r["homotopy"]["classes"][0].get("fraction_on_graph") for r in solved]
        if any(v is not None for v in first_on_h):
            entry["fraction_on_graph"] = {
                "best_class": _stats([v for v in first_on_h if v is not None]),
                "other_classes": _stats([c["fraction_on_graph"] for r in solved
                                         for c in r["homotopy"]["classes"][1:]
                                         if c.get("fraction_on_graph") is not None]),
                "baseline": _stats([r["baseline"]["fraction_on_graph"] for r in solved
                                    if (r.get("baseline") or {}).get("fraction_on_graph") is not None]),
            }
        summary[scenario] = entry
    return summary


def write_summary_csv(path, summary, settings):
    fields = ["scenario", "alpha", "pairs", "solved", "num_obstacles", "mean_classes", "median_classes",
              "frac_ge2", "frac_ge3", "frac_at_k", "homotopy_time_median_s", "homotopy_time_p90_s",
              "baseline_time_median_s", "time_ratio_median", "explored_states_median",
              "best_class_is_baseline_class"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scenario, entry in summary.items():
            for alpha, a in entry["alphas"].items():
                writer.writerow({
                    "scenario": scenario, "alpha": alpha, "pairs": entry["pairs"], "solved": entry["solved"],
                    "num_obstacles": entry["num_obstacles"], "mean_classes": a["mean_classes"],
                    "median_classes": a["median_classes"], "frac_ge2": a["frac_ge2"],
                    "frac_ge3": a["frac_ge3"], "frac_at_k": a["frac_at_k"],
                    "homotopy_time_median_s": entry["homotopy_time_s"]["median"],
                    "homotopy_time_p90_s": entry["homotopy_time_s"]["p90"],
                    "baseline_time_median_s": (entry.get("baseline_time_s") or {}).get("median"),
                    "time_ratio_median": (entry.get("time_ratio") or {}).get("median"),
                    "explored_states_median": entry["explored_states"]["median"],
                    "best_class_is_baseline_class": entry.get("best_class_is_baseline_class"),
                })


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_class_counts(path, summary, settings):
    """Share of pairs with 1, 2, 3, >=4 classes; one panel per alpha."""
    import matplotlib.pyplot as plt

    scenarios = list(summary)
    alphas = list(next(iter(summary.values()))["alphas"])
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.6 * len(alphas) + 1.6, 0.45 * len(scenarios) + 1.6),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    labels = ["1 class", "2", "3", ">= 4"]
    for ax, alpha in zip(axes, alphas):
        _style_axis(ax)
        for yi, scenario in enumerate(scenarios):
            a = summary[scenario]["alphas"][alpha]
            n = summary[scenario]["solved"]
            if not n:
                continue
            hist = a["histogram"]
            shares = [hist.get("1", 0), hist.get("2", 0), hist.get("3", 0),
                      sum(v for key, v in hist.items() if int(key) >= 4)]
            left = 0.0
            for bucket, count in enumerate(shares):
                width = count / n
                if width > 0:
                    ax.barh(yi, width, left=left, height=0.62, color=COUNT_COLORS[bucket],
                            edgecolor=SURFACE, linewidth=2, label=labels[bucket] if yi == 0 else None)
                left += width
            ax.text(1.02, yi, f"{a['mean_classes']:.1f}", va="center", ha="left", fontsize=8, color=TEXT)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_title(f"alpha = {alpha}", fontsize=9, color=TEXT)
    axes[0].set_yticks(range(len(scenarios)))
    axes[0].set_yticklabels([s.replace("_default", "") for s in scenarios])
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COUNT_COLORS]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), labelcolor=TEXT)
    merge = settings["merge_overlap"]
    fig.suptitle(
        f"Homotopy classes within (1+alpha) x best cost per random pair (K={settings['k']}"
        + (f", merge {merge:g}" if merge is not None else "")
        + (", graph-biased cost" if settings.get("heatmaps") else "") + "); numbers = mean",
        fontsize=9, color=TEXT, y=1.06)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_timing(path, rows_by_scenario, settings):
    """Per-pair solve time: class-aware search vs the baseline social A* (log scale)."""
    import matplotlib.pyplot as plt

    scenarios = list(rows_by_scenario)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(scenarios) + 1.4), facecolor=SURFACE)
    _style_axis(ax)
    rng = np.random.default_rng(0)
    series = [("homotopy", -0.14, f"class-aware search (K={settings['k']})")]
    if settings["baseline"]:
        series.append(("baseline", 0.14,
                       "graph-biased social A* baseline" if settings.get("heatmaps") else "social A* baseline"))
    for key, offset, label in series:
        for yi, scenario in enumerate(scenarios):
            times = [r[key]["time_s"] for r in rows_by_scenario[scenario]
                     if (r.get("homotopy") or {}).get("success")]
            if not times:
                continue
            jitter = rng.uniform(-0.07, 0.07, len(times))
            ax.scatter(times, yi + offset + jitter, s=12, color=SERIES_COLORS[key], alpha=0.55,
                       linewidths=0, label=label if yi == 0 else None)
            ax.plot([np.median(times)] * 2, [yi + offset - 0.13, yi + offset + 0.13],
                    color=SERIES_COLORS[key], lw=2.5, solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_xlabel("solve time per pair [s] (tick = median)", fontsize=8, color=TEXT)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.replace("_default", "") for s in scenarios])
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=8,
              labelcolor=TEXT, markerscale=1.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--pairs", type=int, default=30, help="random pairs per scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-separation", type=float, default=30.0, help="minimum start-goal distance [m]")
    parser.add_argument("--k", type=int, default=8, help="class cap per pair (counts saturate at k)")
    parser.add_argument("--alphas", default="0.1,0.3,0.5", help="comma list; the search uses the largest")
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--merge-overlap", type=float, default=None)
    parser.add_argument("--merge-distance", type=float, default=0.5)
    parser.add_argument("--mode", choices=["modified", "vanilla"], default="modified")
    parser.add_argument("--wall-cache", choices=WALL_CACHE_POLICIES, default="if_present")
    parser.add_argument("--time-limit", type=float, default=300.0, help="per-pair search limit [s]")
    parser.add_argument("--no-baseline", action="store_true", help="skip the social A* baseline runs")
    parser.add_argument("--heatmap", action="append", default=[], metavar="SCENARIO=PATH",
                        help="directional heatmap for a scenario; that scenario then uses the graph-biased "
                             "cost (Eq. 4) for both the baseline and the class-aware search. Repeatable.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: results/homotopy_survey/<tag from settings>")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="rebuild summary and plots from rows.jsonl")
    args = parser.parse_args(argv)
    args.alphas = sorted(float(v) for v in args.alphas.split(",") if v.strip())
    args.scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if not args.alphas:
        parser.error("--alphas is empty")
    heatmaps = {}
    for item in args.heatmap:
        scenario, sep, path = item.partition("=")
        if not sep or not scenario.strip() or not path.strip():
            parser.error(f"--heatmap expects SCENARIO=PATH, got {item!r}")
        heatmaps[scenario.strip()] = str(Path(path.strip()).resolve())
    unknown = sorted(set(heatmaps) - set(args.scenarios))
    if unknown:
        parser.error(f"--heatmap for scenarios not in --scenarios: {unknown}")
    args.heatmaps = heatmaps
    return args


def settings_from_args(args):
    return {
        "seed": args.seed, "min_separation": args.min_separation, "k": args.k, "alphas": args.alphas,
        "min_area": args.min_area, "merge_overlap": args.merge_overlap, "merge_distance": args.merge_distance,
        "mode": args.mode, "wall_cache": args.wall_cache, "time_limit": args.time_limit,
        "baseline": not args.no_baseline,
        "heatmaps": dict(sorted(getattr(args, "heatmaps", {}).items())),
    }


def default_output_dir(settings):
    tag = f"{settings['mode']}_k{settings['k']}_a{max(settings['alphas']):g}_seed{settings['seed']}"
    if settings["merge_overlap"] is not None:
        tag += f"_merge{settings['merge_overlap']:g}"
    if settings["heatmaps"]:
        tag += "_graph"
    return DEFAULT_OUTPUT_ROOT / tag


def load_rows(path):
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return rows


def run(args):
    import matplotlib

    matplotlib.use("Agg")
    from social_path_planning.homotopy import extract_obstacles

    settings = settings_from_args(args)
    out = args.output_dir or default_output_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    meta_path, rows_path = out / "meta.json", out / "rows.jsonl"

    if meta_path.exists() and (args.resume or args.plot_only):
        meta = json.loads(meta_path.read_text())
        meta.setdefault("heatmaps", {})  # folders from before --heatmap existed
        diffs = {key: (meta.get(key), settings[key]) for key in META_KEYS if meta.get(key) != settings[key]}
        if diffs:
            raise SystemExit(f"--resume settings differ from {meta_path}: {diffs}")
    elif rows_path.exists() and not args.plot_only:
        raise SystemExit(f"{rows_path} exists; pass --resume or a different --output-dir")
    meta_path.write_text(json.dumps({**settings, "scenarios": args.scenarios, "pairs": args.pairs}, indent=1))

    done = {(r["scenario"], r["pair"]) for r in load_rows(rows_path)}
    obstacles_by_scenario = {}
    graph_nodes_by_scenario = {}
    for scenario in args.scenarios:
        occ, _, res, statespace_hi = build_occ_grid(scenario, wall_cache=args.wall_cache)
        obstacles = extract_obstacles(occ, min_area_m2=args.min_area, statespace_lo=[0, 0],
                                      statespace_hi=statespace_hi, astar_resolution=res)
        obstacles_by_scenario[scenario] = obstacles.num_obstacles
        if args.plot_only:
            continue
        graph = None
        heatmap = settings["heatmaps"].get(scenario)
        if heatmap:
            from social_path_planning.compare_astar import build_sparse_graph_from_heatmap

            graph = build_sparse_graph_from_heatmap(occ, heatmap_path=heatmap)
            graph_nodes_by_scenario[scenario] = graph.number_of_nodes()
        pairs = scenario_pairs(occ, scenario, args.seed, args.pairs, args.min_separation)
        todo = [i for i in range(len(pairs)) if (scenario, i) not in done]
        print(f"{scenario}: {obstacles.num_obstacles} obstacles"
              + (f", social graph H with {graph.number_of_nodes()} nodes" if graph is not None else "")
              + f", {len(todo)}/{len(pairs)} pairs to plan", flush=True)
        t_scen = time.time()
        for i in todo:
            start, goal = pairs[i]
            row = survey_pair(occ, res, statespace_hi, start, goal, settings, obstacles=obstacles, graph=graph)
            row.update({"scenario": scenario, "pair": i})
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            h = row.get("homotopy") or {}
            print(f"  pair {i:3d}: {len(h.get('classes', []))} classes, "
                  f"{h.get('time_s', 0.0):6.2f} s ({h.get('engine', '-')})", flush=True)
        print(f"  {scenario} done in {time.time() - t_scen:.0f} s", flush=True)

    rows = [r for r in load_rows(rows_path) if r["scenario"] in args.scenarios and r["pair"] < args.pairs]
    rows_by_scenario = {s: sorted((r for r in rows if r["scenario"] == s), key=lambda r: r["pair"])
                        for s in args.scenarios}
    summary = summarize(rows_by_scenario, obstacles_by_scenario, settings)
    for scenario, entry in summary.items():
        entry["heatmap"] = settings["heatmaps"].get(scenario)
        if scenario in graph_nodes_by_scenario:
            entry["graph_nodes"] = graph_nodes_by_scenario[scenario]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_summary_csv(out / "summary.csv", summary, settings)
    plot_class_counts(out / "class_counts.png", summary, settings)
    plot_timing(out / "timing.png", rows_by_scenario, settings)
    print(f"wrote {out}")
    return summary


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
