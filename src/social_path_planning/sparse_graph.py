import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import argparse
import json
import glob
import re
import pickle

from social_path_planning.compare_astar import build_occ_grid, generate_random_free_point
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.heat_map import HeatMap2DVector
from social_path_planning.a_star import AStar, AStar_With_Graph
from social_path_planning.utils import *

CHECKPOINT_FILENAME = "heatmap_checkpoint.npy"
METADATA_FILENAME = "run_metadata.json"


class _GlobalNumpyUniformRng:
    """Adapter: `generate_random_free_point` expects a Generator-like object; timeline uses `np.random` (seed/resume)."""

    __slots__ = ()

    def uniform(self, low, high):
        return np.random.uniform(low=low, high=high)

class FrequentSubgraph:

    def __init__(self, occ_grid: StochOccupancyGrid2D, heat_map_filename: str = None):
        self.occ_grid = occ_grid
        self.heat_map_object = HeatMap2DVector(occ_grid)
        if heat_map_filename is not None:
            self.heat_map_object.load_heatmap(heat_map_filename)
        self.heat_map = self.heat_map_object.heatmap

        self.graph = nx.DiGraph()

    def set_heat_map(self, heat_map: np.ndarray):
        self.heat_map = heat_map

    def build_graph(self, threshold: float, reset_graph: bool = True):
        """
        Build a directed graph from the heat map, only including nodes/edges that are frequently traversed
        """
        if reset_graph:
            self.graph = nx.DiGraph()

        directions = [
            (-1,  1),  # NW
            (0,   1),  # N
            (1,   1),  # NE
            (1,   0),  # E
            (-1,  0),  # W
            (-1, -1),  # SW
            (0,  -1),  # S
            (1,  -1)   # SE
        ]

        for x in range(self.occ_grid.width):
            for y in range(self.occ_grid.height):
                for dir_idx, (dx, dy) in enumerate(directions):
                    # heat_map is (height, width, 8); x=col, y=row in grid indices
                    if self.heat_map[y, x, dir_idx] >= threshold:
                        from_node = (x, y)
                        to_node = (x + dx, y + dy)

                        if (0 <= to_node[0] < self.occ_grid.width) and (0 <= to_node[1] < self.occ_grid.height):
                            if self.occ_grid.is_free((to_node[0] * self.occ_grid.resolution + self.occ_grid.origin_x,
                                                      to_node[1] * self.occ_grid.resolution + self.occ_grid.origin_y)):
                                self.graph.add_edge(from_node, to_node, weight=np.linalg.norm(self.occ_grid.resolution * np.array([dx, dy])))

    def prune_graph(self, min_component_size: int = 15):
        # Remove components with less than a certain number of nodes
        components = list(nx.weakly_connected_components(self.graph))

        for component in components:
            if len(component) < min_component_size:
                self.graph.remove_nodes_from(component)

    def visualize_graph(
        self,
        ax=None,
        show=True,
        title="Sparse Graph",
        xlabel="X (m)",
        ylabel="Y (m)",
        title_fontsize=16,
        label_fontsize=16,
        tick_fontsize=36,
        edge_color="red",
        edge_linewidth=2,
        edge_alpha=0.75,
        figsize=(2.33, 2.45),
        heatmap_layer=None,
        min_visible_intensity=5.0,
        heatmap_legend=False,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if heatmap_layer is not None:
            heatmap_layer.plot_heatmap(
                ax=ax,
                show=False,
                min_visible_intensity=min_visible_intensity,
                add_legend=heatmap_legend,
            )
        else:
            self.occ_grid.plot_grid(ax=ax)
        for edge in self.graph.edges():
            from_node = edge[0]
            to_node = edge[1]
            ax.plot(
                [
                    from_node[0] * self.occ_grid.resolution + self.occ_grid.origin_x,
                    to_node[0] * self.occ_grid.resolution + self.occ_grid.origin_x,
                ],
                [
                    from_node[1] * self.occ_grid.resolution + self.occ_grid.origin_y,
                    to_node[1] * self.occ_grid.resolution + self.occ_grid.origin_y,
                ],
                color=edge_color,
                linewidth=edge_linewidth,
                alpha=edge_alpha,
            )
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        if show:
            plt.show()
        return ax

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b), ord=1)

def _latest_frame_index(output_dir: str) -> int:
    frame_paths = glob.glob(os.path.join(output_dir, "frame_*.png"))
    latest_idx = 0
    for frame_path in frame_paths:
        match = re.search(r"frame_(\d+)\.png$", os.path.basename(frame_path))
        if match:
            latest_idx = max(latest_idx, int(match.group(1)))
    return latest_idx

def _serialize_rng_state(rng_state):
    return {
        "bit_generator": rng_state[0],
        "state": rng_state[1].tolist(),
        "pos": int(rng_state[2]),
        "has_gauss": int(rng_state[3]),
        "cached_gaussian": float(rng_state[4]),
    }

def _deserialize_rng_state(serialized):
    return (
        serialized["bit_generator"],
        np.array(serialized["state"], dtype=np.uint32),
        int(serialized["pos"]),
        int(serialized["has_gauss"]),
        float(serialized["cached_gaussian"]),
    )


def _extract_snapshot_idx(path_obj):
    basename = os.path.basename(str(path_obj))
    match = re.search(r"graph_(\d+)\.pkl$", basename)
    if not match:
        return -1
    return int(match.group(1))


def _load_timeline_metadata(timeline_dir):
    metadata_path = os.path.join(timeline_dir, METADATA_FILENAME)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _discover_graph_snapshots(timeline_dir):
    timeline_dir = os.path.abspath(timeline_dir)
    metadata = _load_timeline_metadata(timeline_dir)

    graph_paths = []
    if isinstance(metadata.get("graph_snapshots"), list) and metadata["graph_snapshots"]:
        for p in metadata["graph_snapshots"]:
            pp = p if os.path.isabs(p) else os.path.join(timeline_dir, p)
            if os.path.exists(pp):
                graph_paths.append(os.path.abspath(pp))

    if not graph_paths:
        graph_paths = sorted(
            glob.glob(os.path.join(timeline_dir, "graph_*.pkl")),
            key=_extract_snapshot_idx,
        )

    if not graph_paths:
        raise FileNotFoundError(f"No graph snapshots found in {timeline_dir}")

    return graph_paths, metadata


def _resolve_graph_snapshot_path(timeline_dir, snapshot_idx=None, graph_file=None):
    if graph_file is not None:
        graph_file = os.path.abspath(graph_file)
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        return graph_file

    if snapshot_idx is None:
        raise ValueError("Either snapshot_idx or graph_file must be provided")

    graph_paths, _metadata = _discover_graph_snapshots(timeline_dir)
    target_name = f"graph_{snapshot_idx:04d}.pkl"
    for path in graph_paths:
        if os.path.basename(path) == target_name:
            return path

    fallback = os.path.join(os.path.abspath(timeline_dir), target_name)
    if os.path.exists(fallback):
        return os.path.abspath(fallback)

    available = [os.path.basename(p) for p in graph_paths]
    raise FileNotFoundError(
        f"No graph snapshot {target_name} in {timeline_dir}. Available: {available}"
    )


def _load_heatmap_layer(occ_grid, heatmap_prefix=None, heatmap_path=None):
    """Load a directional heatmap for optional underlay (prefix -> <prefix>_heatmap.npy)."""
    if not heatmap_prefix and not heatmap_path:
        return None

    heatmap_layer = HeatMap2DVector(occ_grid)
    if heatmap_path:
        heatmap_path = os.path.abspath(heatmap_path)
        if not os.path.exists(heatmap_path):
            raise FileNotFoundError(f"Heatmap file not found: {heatmap_path}")
        heatmap_layer.heatmap = np.load(heatmap_path)
    else:
        heatmap_layer.load_heatmap(heatmap_prefix)

    expected = (heatmap_layer.height, heatmap_layer.width, 8)
    if heatmap_layer.heatmap.shape != expected:
        raise ValueError(
            f"Heatmap shape mismatch. Expected {expected}, got {heatmap_layer.heatmap.shape}"
        )
    return heatmap_layer


def plan_random_social_paths(
    occ_grid,
    statespace_hi,
    map_resolution,
    graph,
    num_paths: int,
    *,
    seed: int = 0,
    max_attempts: int = 200,
):
    """Plan up to ``num_paths`` random start/goal routes with ``AStar_With_Graph``."""
    if num_paths <= 0:
        return []

    rng = np.random.default_rng(seed)
    paths = []
    attempts = 0
    while len(paths) < num_paths and attempts < max_attempts:
        attempts += 1
        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        if x_init == x_goal:
            continue
        problem = AStar_With_Graph(
            [0, 0],
            statespace_hi,
            x_init,
            x_goal,
            occ_grid,
            graph,
            resolution=map_resolution,
        )
        if problem.solve(mode="modified") and problem.path and len(problem.path) >= 2:
            paths.append(problem.path)

    if len(paths) < num_paths:
        print(
            f"Warning: planned {len(paths)}/{num_paths} overlay paths "
            f"after {attempts} attempts (max_attempts={max_attempts})"
        )
    else:
        print(f"Planned {len(paths)} overlay path(s) in {attempts} attempt(s)")
    return paths


def _plot_paths_on_ax(ax, paths, *, linewidth=1.8, alpha=0.9):
    cmap = plt.cm.tab10
    for i, path in enumerate(paths):
        xs, ys = zip(*path)
        ax.plot(
            xs,
            ys,
            color=cmap(i % 10),
            linewidth=linewidth,
            alpha=alpha,
            linestyle="-",
            zorder=6,
        )


def plot_graph_snapshot(
    timeline_dir="outputs_timeline",
    snapshot_idx=None,
    graph_file=None,
    scenario_name=None,
    graph_threshold=1.0,
    min_component_size=15,
    title="Social Graph",
    xlabel="X (m)",
    ylabel="Y (m)",
    title_fontsize=16,
    label_fontsize=16,
    tick_fontsize=30,
    edge_color="red",
    edge_linewidth=2,
    edge_alpha=0.75,
    figsize=(10, 10),
    heatmap_prefix=None,
    heatmap_path=None,
    min_visible_intensity=5.0,
    heatmap_legend=False,
    save_fig=None,
    show=True,
    dpi=600,
    num_overlay_paths=0,
    overlay_seed=0,
    max_overlay_attempts=200,
    overlay_linewidth=1.8,
    overlay_alpha=0.9,
):
    has_graph_source = snapshot_idx is not None or graph_file is not None
    has_heatmap_source = bool(heatmap_prefix or heatmap_path)
    if not has_graph_source and not has_heatmap_source:
        raise ValueError(
            "Provide a saved graph (--snapshot-idx or --graph-file) and/or a heatmap "
            "(--heatmap-prefix or --heatmap-file to build the graph from the heatmap)."
        )

    timeline_dir = os.path.abspath(timeline_dir)
    metadata = _load_timeline_metadata(timeline_dir)
    if scenario_name is None:
        scenario_name = metadata.get("scenario_name")
    if not scenario_name:
        raise ValueError(
            "scenario_name is required when run_metadata.json is missing or has no scenario_name"
        )

    occ_grid, _, map_resolution, statespace_hi = build_occ_grid(scenario_name)
    frequent_graph = FrequentSubgraph(occ_grid)
    heatmap_layer = _load_heatmap_layer(
        occ_grid, heatmap_prefix=heatmap_prefix, heatmap_path=heatmap_path
    )

    graph_path = None
    built_from_heatmap = False
    if has_graph_source:
        graph_path = _resolve_graph_snapshot_path(
            timeline_dir, snapshot_idx=snapshot_idx, graph_file=graph_file
        )
        with open(graph_path, "rb") as graph_fp:
            frequent_graph.graph = pickle.load(graph_fp)
    else:
        frequent_graph.set_heat_map(heatmap_layer.heatmap)
        frequent_graph.build_graph(threshold=graph_threshold, reset_graph=True)
        frequent_graph.prune_graph(min_component_size=min_component_size)
        built_from_heatmap = True

    ax = frequent_graph.visualize_graph(
        show=False,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        edge_color=edge_color,
        edge_linewidth=edge_linewidth,
        edge_alpha=edge_alpha,
        figsize=figsize,
        heatmap_layer=heatmap_layer,
        min_visible_intensity=min_visible_intensity,
        heatmap_legend=heatmap_legend,
    )

    overlay_paths = []
    if num_overlay_paths > 0:
        overlay_paths = plan_random_social_paths(
            occ_grid,
            statespace_hi,
            map_resolution,
            frequent_graph.graph,
            num_overlay_paths,
            seed=overlay_seed,
            max_attempts=max_overlay_attempts,
        )
        if overlay_paths:
            _plot_paths_on_ax(
                ax,
                overlay_paths,
                linewidth=overlay_linewidth,
                alpha=overlay_alpha,
            )

    ax.set_axis_off()
    fig = ax.figure
    fig.tight_layout()

    if save_fig:
        save_parent = os.path.dirname(os.path.abspath(save_fig))
        if save_parent:
            os.makedirs(save_parent, exist_ok=True)
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure: {save_fig}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    graph = frequent_graph.graph
    return {
        "graph_path": graph_path,
        "built_from_heatmap": built_from_heatmap,
        "scenario_name": scenario_name,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "save_fig": save_fig,
        "overlay_paths": len(overlay_paths),
    }


def plot_all_graph_snapshots(
    timeline_dir="outputs_timeline",
    scenario_name=None,
    title="Social Graph",
    xlabel="X (m)",
    ylabel="Y (m)",
    title_fontsize=10,
    label_fontsize=8,
    tick_fontsize=8,
    edge_color="red",
    edge_linewidth=2.5,
    edge_alpha=0.75,
    figsize=(10, 10),
    heatmap_prefix=None,
    heatmap_path=None,
    min_visible_intensity=5.0,
    heatmap_legend=False,
    save_dir=None,
    dpi=300,
    num_overlay_paths=0,
    overlay_seed=0,
    max_overlay_attempts=200,
    overlay_linewidth=1.8,
    overlay_alpha=0.9,
):
    timeline_dir = os.path.abspath(timeline_dir)
    if save_dir is None:
        save_dir = os.path.join(timeline_dir, "social_graph_replots")
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    graph_paths, _metadata = _discover_graph_snapshots(timeline_dir)
    results = []
    for graph_path in graph_paths:
        snapshot_idx = _extract_snapshot_idx(graph_path)
        save_fig = os.path.join(save_dir, f"social_graph_{snapshot_idx:04d}.svg")
        result = plot_graph_snapshot(
            timeline_dir=timeline_dir,
            graph_file=graph_path,
            scenario_name=scenario_name,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            edge_color=edge_color,
            edge_linewidth=edge_linewidth,
            edge_alpha=edge_alpha,
            figsize=figsize,
            heatmap_prefix=heatmap_prefix,
            heatmap_path=heatmap_path,
            min_visible_intensity=min_visible_intensity,
            heatmap_legend=heatmap_legend,
            save_fig=save_fig,
            show=False,
            dpi=dpi,
            num_overlay_paths=num_overlay_paths,
            overlay_seed=overlay_seed,
            max_overlay_attempts=max_overlay_attempts,
            overlay_linewidth=overlay_linewidth,
            overlay_alpha=overlay_alpha,
        )
        results.append(result)
    print(f"Replotted {len(results)} snapshots to {save_dir}")
    return results


def save_side_by_side_timeline(
    scenario_name: str = "sample2_default",
    num_paths: int = 100,
    batch_k: int = 10,
    graph_threshold: float = 1.0,
    min_component_size: int = 15,
    output_dir: str = "outputs_timeline",
    seed: int = 0,
    resume: bool = False,
):
    if batch_k <= 0:
        raise ValueError("batch_k must be > 0")
    os.makedirs(output_dir, exist_ok=True)

    occ_grid, map_size, map_resolution, statespace_hi = build_occ_grid(scenario_name)

    heatmap = HeatMap2DVector(occ_grid)
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILENAME)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)

    if resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Cannot resume: checkpoint not found at {checkpoint_path}")
        heatmap.heatmap = np.load(checkpoint_path)
        if heatmap.heatmap.shape != (heatmap.height, heatmap.width, 8):
            raise ValueError(
                f"Checkpoint shape mismatch. Expected {(heatmap.height, heatmap.width, 8)}, "
                f"got {heatmap.heatmap.shape}"
            )
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = {}
        rng_state_data = existing_metadata.get("rng_state", None)
        if rng_state_data is not None:
            np.random.set_state(_deserialize_rng_state(rng_state_data))
            rng_source = "restored"
        else:
            np.random.seed(seed)
            rng_source = "seeded_fallback"
    else:
        np.random.seed(seed)
        heatmap.heatmap = np.zeros_like(heatmap.heatmap)
        existing_metadata = {}
        rng_source = "seeded"

    frequent_graph = FrequentSubgraph(occ_grid)
    frequent_graph.set_heat_map(heatmap.heatmap)

    solved_paths_before = int(existing_metadata.get("cumulative_solved_paths", 0))
    attempted_before = int(existing_metadata.get("cumulative_attempted_paths", 0))
    snapshot_idx = int(existing_metadata.get("snapshots_saved", _latest_frame_index(output_dir)))
    graph_snapshots = list(existing_metadata.get("graph_snapshots", []))

    solved_paths_this_run = 0
    attempted_this_run = 0

    _sample_rng = _GlobalNumpyUniformRng()

    while solved_paths_this_run < num_paths:
        attempted_this_run += 1
        x_init = generate_random_free_point(occ_grid, _sample_rng)
        x_goal = generate_random_free_point(occ_grid, _sample_rng)
        problem = AStar([0, 0], statespace_hi, x_init, x_goal, occ_grid, resolution=map_resolution)

        if not problem.solve():
            continue

        heatmap.add_path(problem.path, increment=1.0)
        solved_paths_this_run += 1
        cumulative_solved_paths = solved_paths_before + solved_paths_this_run
        cumulative_attempted_paths = attempted_before + attempted_this_run

        if cumulative_solved_paths % batch_k != 0 and solved_paths_this_run != num_paths:
            continue

        snapshot_idx += 1
        frequent_graph.build_graph(threshold=graph_threshold, reset_graph=True)
        frequent_graph.prune_graph(min_component_size=min_component_size)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        heatmap.plot_heatmap(ax=axes[0], show=False, add_legend=False)
        axes[0].set_title(f"Directional Heatmap | solved_paths={cumulative_solved_paths}")

        frequent_graph.visualize_graph(ax=axes[1], show=False)
        axes[1].set_title(
            "Social Graph", fontsize=16
        )

        fig.tight_layout()
        filename = f"frame_{snapshot_idx:04d}.png"
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        graph_filename = f"graph_{snapshot_idx:04d}.pkl"
        graph_output_path = os.path.join(output_dir, graph_filename)
        with open(graph_output_path, "wb") as graph_file:
            pickle.dump(frequent_graph.graph, graph_file)
        graph_snapshots.append(graph_output_path)

        np.save(checkpoint_path, heatmap.heatmap)
        print(
            f"[snapshot {snapshot_idx:04d}] solved_paths={cumulative_solved_paths} attempted={cumulative_attempted_paths} "
            f"nodes={frequent_graph.graph.number_of_nodes()} edges={frequent_graph.graph.number_of_edges()} "
            f"file={output_path} graph_file={graph_output_path}"
        )

    result = {
        "scenario_name": scenario_name,
        "num_paths": num_paths,
        "batch_k": batch_k,
        "graph_threshold": graph_threshold,
        "min_component_size": min_component_size,
        "output_dir": output_dir,
        "seed": seed,
        "resume": resume,
        "paths_added_this_run": solved_paths_this_run,
        "attempted_this_run": attempted_this_run,
        "cumulative_solved_paths": solved_paths_before + solved_paths_this_run,
        "cumulative_attempted_paths": attempted_before + attempted_this_run,
        "snapshots_saved": snapshot_idx,
        "checkpoint_file": checkpoint_path,
        "graph_snapshots": graph_snapshots,
        "rng_source": rng_source,
        "rng_state": _serialize_rng_state(np.random.get_state()),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved run metadata: {metadata_path}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sparse graph utilities and side-by-side timeline exporter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "timeline", "plot-snapshot"],
        default="timeline",
        help=(
            "Run mode: 'timeline' saves progressive heatmap+graph snapshots; "
            "'plot-snapshot' replots a saved graph_*.pkl; 'demo' runs a one-off graph planning demo."
        ),
    )
    parser.add_argument(
        "--scenario-name",
        default=None,
        help=(
            "Scenario key from grid_scenarios.json. "
            "Defaults to sample2_default for timeline/demo; for plot-snapshot, uses run_metadata.json when omitted."
        ),
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="Number of successful new paths to add in this run (when --resume is set, this is additional paths).",
    )
    parser.add_argument(
        "--batch-k",
        type=int,
        default=10,
        help="Save one frame+graph every K cumulative solved paths.",
    )
    parser.add_argument(
        "--graph-threshold",
        type=float,
        default=1.0,
        help="Minimum directional heat count required to include an edge in the sparse graph.",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=15,
        help="Prune weakly-connected graph components smaller than this node count.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_timeline",
        help="Directory for timeline artifacts (frames, graph_*.pkl snapshots, checkpoint, and metadata).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to initialize RNG for a fresh run; on resume, saved RNG state is restored when available.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume timeline from heatmap checkpoint and metadata in --output-dir instead of starting from scratch.",
    )
    plot_group = parser.add_argument_group("plot-snapshot mode")
    plot_group.add_argument(
        "--snapshot-idx",
        type=int,
        default=None,
        help=(
            "Snapshot index for graph_XXXX.pkl in --output-dir (plot-snapshot mode). "
            "Not needed if you only pass --heatmap-prefix/--heatmap-file (graph is built from heatmap)."
        ),
    )
    plot_group.add_argument(
        "--graph-file",
        default=None,
        help="Direct path to a graph pickle (plot-snapshot mode; overrides --snapshot-idx).",
    )
    plot_group.add_argument(
        "--save-fig",
        default=None,
        help="Output PNG path for plot-snapshot mode.",
    )
    plot_group.add_argument(
        "--save-dir",
        default=None,
        help="Output directory when --all-snapshots is set (default: <output-dir>/social_graph_replots).",
    )
    plot_group.add_argument(
        "--all-snapshots",
        action="store_true",
        help="Replot every graph_*.pkl in --output-dir with the same style (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a matplotlib window (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--title",
        default="Social Graph",
        help="Figure title (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--xlabel",
        default="",
        help="X-axis label (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--ylabel",
        default="",
        help="Y-axis label (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--title-fontsize",
        type=float,
        default=45,
        help="Title font size (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--label-fontsize",
        type=float,
        default=16,
        help="Axis label font size (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--tick-fontsize",
        type=float,
        default=30,
        help="Tick label font size (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--edge-linewidth",
        type=float,
        default=4,
        help="Graph edge line width (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--edge-alpha",
        "--transparency",
        type=float,
        default=1.0,
        dest="edge_alpha",
        metavar="ALPHA",
        help="Graph edge transparency in [0, 1] (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--heatmap-prefix",
        default=None,
        help=(
            "Optional heatmap prefix P; loads P_heatmap.npy under the occupancy grid "
            "(e.g. y2e2_routes for y2e2_routes_heatmap.npy)."
        ),
    )
    plot_group.add_argument(
        "--heatmap-file",
        default=None,
        help="Optional path to a directional heatmap .npy (overrides --heatmap-prefix).",
    )
    plot_group.add_argument(
        "--min-heatmap-intensity",
        type=float,
        default=5.0,
        help="Minimum heat count shown when a heatmap underlay is used.",
    )
    plot_group.add_argument(
        "--heatmap-legend",
        action="store_true",
        help="Show direction hue legend when a heatmap underlay is used.",
    )
    plot_group.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 10],
        metavar=("W", "H"),
        help="Figure size in inches (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--plot-dpi",
        type=int,
        default=600,
        help="DPI for saved figures (plot-snapshot mode).",
    )
    plot_group.add_argument(
        "--num-overlay-paths",
        type=int,
        default=0,
        help="Number of random social A* paths to overlay on the graph plot (0 = none).",
    )
    plot_group.add_argument(
        "--overlay-seed",
        type=int,
        default=0,
        help="RNG seed for sampling random overlay start/goal pairs.",
    )
    plot_group.add_argument(
        "--max-overlay-attempts",
        type=int,
        default=200,
        help="Max random start/goal tries when planning overlay paths.",
    )
    plot_group.add_argument(
        "--overlay-linewidth",
        type=float,
        default=1.8,
        help="Line width for overlay paths.",
    )
    plot_group.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.9,
        help="Line alpha for overlay paths.",
    )
    args = parser.parse_args()

    if args.mode == "plot-snapshot":
        style_kwargs = {
            "timeline_dir": args.output_dir,
            "scenario_name": args.scenario_name,
            "title": args.title,
            "xlabel": args.xlabel,
            "ylabel": args.ylabel,
            "title_fontsize": args.title_fontsize,
            "label_fontsize": args.label_fontsize,
            "tick_fontsize": args.tick_fontsize,
            "edge_linewidth": args.edge_linewidth,
            "edge_alpha": args.edge_alpha,
            "heatmap_prefix": args.heatmap_prefix,
            "heatmap_path": args.heatmap_file,
            "min_visible_intensity": args.min_heatmap_intensity,
            "heatmap_legend": args.heatmap_legend,
            "graph_threshold": args.graph_threshold,
            "min_component_size": args.min_component_size,
            "figsize": tuple(args.figsize),
            "dpi": args.plot_dpi,
            "num_overlay_paths": args.num_overlay_paths,
            "overlay_seed": args.overlay_seed,
            "max_overlay_attempts": args.max_overlay_attempts,
            "overlay_linewidth": args.overlay_linewidth,
            "overlay_alpha": args.overlay_alpha,
        }
        if args.all_snapshots:
            results = plot_all_graph_snapshots(
                save_dir=args.save_dir,
                **style_kwargs,
            )
            print("Plot-snapshot batch completed:", len(results), "figures")
        else:
            has_graph = args.snapshot_idx is not None or args.graph_file is not None
            has_heatmap = bool(args.heatmap_prefix or args.heatmap_file)
            if not has_graph and not has_heatmap:
                parser.error(
                    "plot-snapshot mode requires --snapshot-idx, --graph-file, "
                    "--heatmap-prefix/--heatmap-file, or --all-snapshots"
                )
            result = plot_graph_snapshot(
                snapshot_idx=args.snapshot_idx,
                graph_file=args.graph_file,
                save_fig=args.save_fig,
                show=not args.no_show,
                **style_kwargs,
            )
            print("Plot-snapshot completed:", result)
    elif args.mode == "timeline":
        result = save_side_by_side_timeline(
            scenario_name=args.scenario_name or "sample2_default",
            num_paths=args.num_paths,
            batch_k=args.batch_k,
            graph_threshold=args.graph_threshold,
            min_component_size=args.min_component_size,
            output_dir=args.output_dir,
            seed=args.seed,
            resume=args.resume,
        )
        print("Timeline export completed:", result)
    else:
        scenario_name = args.scenario_name or "sample2_default"
        occ_grid, map_size, map_resolution, statespace_hi = build_occ_grid(scenario_name)

        frequent_graph = FrequentSubgraph(occ_grid, "y2e2_routes")
        frequent_graph.build_graph(threshold=args.graph_threshold)
        frequent_graph.prune_graph(min_component_size=args.min_component_size)

        print("Number of nodes in the graph:", frequent_graph.graph.number_of_nodes())
        print("Number of edges in the graph:", frequent_graph.graph.number_of_edges())

        x_init = snap_to_grid([5, 38], map_resolution)
        x_goal = snap_to_grid([40, 40], map_resolution)
        problem = AStar_With_Graph([0, 0], statespace_hi, x_init, x_goal, occ_grid, frequent_graph.graph, resolution=map_resolution)

        problem_status = problem.solve()
        if problem_status:
            print("Path found!")
            plt.figure(2)
            occ_grid.plot_grid_and_path(problem.path)
            plt.scatter(x_init[0], x_init[1], c='green', s=100, label='Start')
            plt.scatter(x_goal[0], x_goal[1], c='gold', marker="*", s=100, label='Goal')
            plt.show()
        else:
            print("No path found.")

        plt.figure(3)
        occ_grid.plot_grid()
        plt.scatter(x_init[0], x_init[1], c='green', s=100, label='Start', zorder=5)
        plt.scatter(x_goal[0], x_goal[1], c='gold', marker="*", s=100, label='Goal')

        closed_set = problem.closed_set
        xs, ys = zip(*closed_set)  # unzip into two sequences
        plt.scatter(xs, ys, c='red', marker="o", s=0.25, label='Explored Nodes')
        plt.show()

        problem.show_path_on_graph()