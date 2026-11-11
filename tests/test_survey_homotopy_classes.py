"""Tests for survey_homotopy_classes (milestone 4)."""
import json

import pytest

from social_path_planning import survey_homotopy_classes as survey


def _row(costs, engine="compiled"):
    return {
        "baseline": {"success": True, "time_s": 0.5, "word": []},
        "homotopy": {"success": bool(costs), "time_s": 1.0, "engine": engine, "explored_states": 10,
                     "timed_out": False, "merged": 0,
                     "classes": [{"word": [i] if i else [], "cost": c, "length_m": c, "social_penalty": 0.0,
                                  "max_overlap": 0.0} for i, c in enumerate(costs)]},
    }


def test_classes_within_counts_by_alpha():
    row = _row([100.0, 105.0, 125.0, 149.0])
    assert survey.classes_within(row, 0.0) == 1
    assert survey.classes_within(row, 0.1) == 2
    assert survey.classes_within(row, 0.3) == 3
    assert survey.classes_within(row, 0.5) == 4
    assert survey.classes_within({"baseline": {"success": False}}, 0.5) == 0


def test_summary_histogram_and_ratios():
    settings = survey.settings_from_args(survey.parse_args(["--alphas", "0.1,0.5", "--k", "4"]))
    rows = {"m": [_row([100.0]), _row([100.0, 120.0]), _row([100.0, 105.0, 140.0])]}
    summary = survey.summarize(rows, {"m": 3}, settings)["m"]
    assert summary["solved"] == 3 and summary["num_obstacles"] == 3
    assert summary["alphas"]["0.1"]["histogram"] == {"1": 2, "2": 1, "3": 0, "4": 0}
    assert summary["alphas"]["0.5"]["histogram"] == {"1": 1, "2": 1, "3": 1, "4": 0}
    assert summary["alphas"]["0.5"]["frac_ge2"] == pytest.approx(2 / 3)
    assert summary["time_ratio"]["median"] == pytest.approx(2.0)
    assert summary["best_class_is_baseline_class"] == 1.0
    assert summary["second_class_cost_ratio"]["median"] == pytest.approx(1.125)


def test_pairs_are_deterministic_and_extend():
    from social_path_planning.common import build_occ_grid

    occ, *_ = build_occ_grid("circular_default", wall_cache="none")
    a = survey.scenario_pairs(occ, "circular_default", 42, 3, 30.0)
    b = survey.scenario_pairs(occ, "circular_default", 42, 5, 30.0)
    assert b[:3] == a
    assert survey.scenario_pairs(occ, "sample2_default", 42, 3, 30.0) != a
    for start, goal in b:
        assert ((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2) ** 0.5 >= 30.0


def test_heatmap_argument_parsing(tmp_path):
    args = survey.parse_args(["--scenarios", "sample2_default", "--heatmap", f"sample2_default={tmp_path / 'h.npy'}"])
    assert args.heatmaps == {"sample2_default": str((tmp_path / "h.npy").resolve())}
    settings = survey.settings_from_args(args)
    assert survey.default_output_dir(settings).name.endswith("_graph")
    with pytest.raises(SystemExit):  # scenario not surveyed
        survey.parse_args(["--scenarios", "sample2_default", "--heatmap", "circular_default=x.npy"])
    with pytest.raises(SystemExit):  # malformed
        survey.parse_args(["--scenarios", "sample2_default", "--heatmap", "sample2_default"])


def test_graph_biased_pair_records_fraction_on_graph():
    import networkx as nx

    from social_path_planning.a_star import AStar
    from social_path_planning.common import build_occ_grid

    occ, _, res, hi = build_occ_grid("circular_default", wall_cache="none")
    start, goal = (50.0, 2.0), (2.0, 50.0)
    seed = AStar([0, 0], hi, start, goal, occ, resolution=res)
    assert seed.solve()
    graph = nx.DiGraph()
    for x1, x2 in zip(seed.path[:-1], seed.path[1:]):
        graph.add_edge(seed.get_index(x1), seed.get_index(x2), weight=seed.distance(x1, x2))
    settings = survey.settings_from_args(survey.parse_args(["--k", "3", "--alphas", "0.5"]))
    row = survey.survey_pair(occ, res, hi, start, goal, settings, graph=graph)
    assert row["baseline"]["fraction_on_graph"] is not None
    assert row["baseline_no_graph"]["time_s"] > 0 and row["baseline_no_graph"]["explored_states"] > 0
    classes = row["homotopy"]["classes"]
    assert classes and all(c["fraction_on_graph"] is not None for c in classes)
    # The graph is exactly a social path, so the best class rides on it almost entirely.
    assert classes[0]["fraction_on_graph"] > 0.9
    summary = survey.summarize({"circular_default": [row]}, {"circular_default": 5}, settings)
    assert summary["circular_default"]["fraction_on_graph"]["best_class"]["median"] > 0.9
    entry = summary["circular_default"]
    assert entry["graph_speedup"]["median"] > 0 and 0 < entry["graph_explored_ratio"]["median"]
    assert entry["graph_faster_fraction"] in (0.0, 1.0)


def test_end_to_end_with_resume(tmp_path):
    argv = ["--scenarios", "circular_default", "--pairs", "2", "--k", "3", "--alphas", "0.3,0.5",
            "--wall-cache", "none", "--output-dir", str(tmp_path)]
    summary = survey.run(survey.parse_args(argv))
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert [r["pair"] for r in rows] == [0, 1]
    assert summary["circular_default"]["solved"] + summary["circular_default"]["unreachable"] == 2
    for name in ("summary.json", "summary.csv", "class_counts.png", "timing.png", "meta.json"):
        assert (tmp_path / name).stat().st_size > 0

    with pytest.raises(SystemExit):  # rows exist: refuse to overwrite without --resume
        survey.run(survey.parse_args(argv))
    with pytest.raises(SystemExit):  # resume with different settings
        survey.run(survey.parse_args(argv[:-2] + ["--k", "4", "--output-dir", str(tmp_path), "--resume"]))

    survey.run(survey.parse_args(argv + ["--resume", "--pairs", "3"]))
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert [r["pair"] for r in rows] == [0, 1, 2]  # only the new pair was planned
