"""Smoke tests for demo_homotopy_alternatives (milestone 3)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from social_path_planning import demo_homotopy_alternatives as demo
from social_path_planning.homotopy_astar import HomotopyAStar

from test_homotopy import RES, make_grid


def test_format_word():
    assert demo.format_word(()) == "[ ]"
    assert demo.format_word((1, -3)) == "[1+ 3-]"


def test_parse_args_requires_a_pair_or_seed():
    with pytest.raises(SystemExit):
        demo.parse_args(["--scenario", "circular_default"])
    with pytest.raises(SystemExit):
        demo.parse_args(["--start", "1", "2"])
    args = demo.parse_args(["--seed", "3"])
    assert args.seed == 3 and args.k == 4 and args.alpha == 0.5


def test_plot_and_json_on_synthetic_grid():
    occ = make_grid([(15, 25, 12, 18)])
    p = HomotopyAStar([0, 0], [40 * RES, 30 * RES], (1.0, 3.0), (7.0, 3.0), occ, resolution=RES,
                      min_area_m2=0.0, k=3, alpha=2.0)
    alts = p.solve_alternatives()
    fig, ax = plt.subplots()
    demo.plot_alternatives(ax, p, alts, title="t")
    labels = [line.get_label() for line in ax.get_lines() if line.get_label().startswith("#")]
    assert len(labels) == len(alts) == 2
    plt.close(fig)
    rows = demo.alternatives_to_json(alts)
    assert [r["rank"] for r in rows] == [1, 2]
    assert json.loads(json.dumps(rows))[0]["path"][0] == [1.0, 3.0]


def test_end_to_end_writes_figure_and_summary(tmp_path):
    summary = demo.run(demo.parse_args([
        "--scenario", "circular_default", "--start", "50", "2", "--goal", "2", "50",
        "--k", "2", "--alpha", "0.5", "--wall-cache", "none", "--output-dir", str(tmp_path), "--name", "t",
    ]))
    assert (tmp_path / "t.png").stat().st_size > 0
    saved = json.loads((tmp_path / "t.json").read_text())
    assert saved["alternatives"] == summary["alternatives"]
    assert len(saved["alternatives"]) == 2
    assert len({tuple(a["word"]) for a in saved["alternatives"]}) == 2
