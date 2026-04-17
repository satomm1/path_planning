import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import argparse
import json
import glob
import re
import pickle

from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.heat_map import HeatMap2DVector, generate_random_free_point
from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.a_star import AStar, AStar_With_Graph
from social_path_planning.utils import *

CHECKPOINT_FILENAME = "heatmap_checkpoint.npy"
METADATA_FILENAME = "run_metadata.json"

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
                    if self.heat_map[x, y, dir_idx] >= threshold:
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

    def visualize_graph(self, ax=None, show=True):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        self.occ_grid.plot_grid(ax=ax)
        for edge in self.graph.edges():
            from_node = edge[0]
            to_node = edge[1]
            ax.plot([from_node[0] * self.occ_grid.resolution + self.occ_grid.origin_x,
                     to_node[0] * self.occ_grid.resolution + self.occ_grid.origin_x],
                    [from_node[1] * self.occ_grid.resolution + self.occ_grid.origin_y,
                     to_node[1] * self.occ_grid.resolution + self.occ_grid.origin_y],
                    color='red', linewidth=2)
        ax.set_title("Sparse Graph")
        ax.set_xlabel("X (m)", fontsize=16)
        ax.set_ylabel("Y (m)", fontsize=16)
        # Set x and y tick size
        ax.tick_params(axis='both', which='major', labelsize=16)
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

    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    occ_grid = StochOccupancyGrid2D(
        map_resolution,
        round(map_size[0] / map_resolution),
        round(map_size[1] / map_resolution),
        0,
        0,
        10,
        occ.T
    )

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

    while solved_paths_this_run < num_paths:
        attempted_this_run += 1
        x_init = generate_random_free_point(occ_grid)
        x_goal = generate_random_free_point(occ_grid)
        problem = AStar([0, 0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)

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
        choices=["demo", "timeline"],
        default="timeline",
        help="Run mode: 'timeline' saves progressive heatmap+graph snapshots, 'demo' runs a one-off graph planning demo.",
    )
    parser.add_argument(
        "--scenario-name",
        default="sample2_default",
        help="Scenario key from grid_scenarios.json used to build the occupancy map.",
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
    args = parser.parse_args()

    if args.mode == "timeline":
        result = save_side_by_side_timeline(
            scenario_name=args.scenario_name,
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
        scenario_name = args.scenario_name
        occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
        occ_grid = StochOccupancyGrid2D(map_resolution, round(map_size[0] / map_resolution),
                                        round(map_size[1] / map_resolution), 0, 0, 10, occ.T)

        frequent_graph = FrequentSubgraph(occ_grid, "vector_incomplete")
        frequent_graph.build_graph(threshold=args.graph_threshold)
        frequent_graph.prune_graph(min_component_size=args.min_component_size)

        print("Number of nodes in the graph:", frequent_graph.graph.number_of_nodes())
        print("Number of edges in the graph:", frequent_graph.graph.number_of_edges())

        x_init = snap_to_grid([2, 2], map_resolution)
        x_goal = snap_to_grid([75, 97], map_resolution)
        problem = AStar_With_Graph([0, 0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, frequent_graph.graph, resolution=map_resolution)

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