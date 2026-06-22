"""MAPF comparison visualization (plots, animations, reports)."""

from social_path_planning.mapf_comparison.viz.animations import (
    create_mapf_animation,
    create_multi_panel_animation,
    create_side_by_side_animation,
)
from social_path_planning.mapf_comparison.viz.demo import build_demo_open_scenario
from social_path_planning.mapf_comparison.viz.plots import (
    plot_routes_overlay,
    plot_side_by_side_snapshots,
)
from social_path_planning.mapf_comparison.viz.report import write_verification_report

__all__ = [
    "build_demo_open_scenario",
    "create_mapf_animation",
    "create_multi_panel_animation",
    "create_side_by_side_animation",
    "plot_routes_overlay",
    "plot_side_by_side_snapshots",
    "write_verification_report",
]
