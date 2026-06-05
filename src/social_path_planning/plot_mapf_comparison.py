"""CLI shim for MAPF comparison plots."""

from social_path_planning.mapf_comparison.plot import main, plot_example_map, plot_metrics_bars

__all__ = ["main", "plot_metrics_bars", "plot_example_map"]

if __name__ == "__main__":
    raise SystemExit(main())
