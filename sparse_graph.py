import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from occupancy_grid import StochOccupancyGrid2D
from heat_map import HeatMap2DVector
from path_planning import generate_sample_grid2
from a_star import AStar_With_Graph
from utils import *

class FrequentSubgraph:

    def __init__(self, occ_grid: StochOccupancyGrid2D, heat_map_filename: str):
        self.occ_grid = occ_grid
        self.heat_map_object = HeatMap2DVector(occ_grid)
        self.heat_map_object.load_heatmap(heat_map_filename)
        self.heat_map = self.heat_map_object.heatmap

        self.graph = nx.DiGraph()

    def build_graph(self, threshold: float):
        """
        Build a directed graph from the heat map, only including nodes/edges that are frequently traversed
        """
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

        total_heatmap = np.sum(self.heat_map, axis=2)
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

    def visualize_graph(self):
        # Iterate through each edge and plot it
        plt.figure(figsize=(10, 10))
        # self.occ_grid.plot_grid()
        for edge in self.graph.edges():
            from_node = edge[0]
            to_node = edge[1]
            plt.plot([from_node[0] * self.occ_grid.resolution + self.occ_grid.origin_x,
                      to_node[0] * self.occ_grid.resolution + self.occ_grid.origin_x],
                     [from_node[1] * self.occ_grid.resolution + self.occ_grid.origin_y,
                      to_node[1] * self.occ_grid.resolution + self.occ_grid.origin_y],
                     color='red', linewidth=1)
        plt.show()

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b), ord=1)

if __name__ == "__main__":
    map_size = [100, 100]
    map_resolution = 0.2

    occ = generate_sample_grid2(map_size, map_resolution, plot=False)
    occ_grid = StochOccupancyGrid2D(map_resolution, round(map_size[0] / map_resolution),
                                    round(map_size[1] / map_resolution), 0, 0, 10, occ.T)

    frequent_graph = FrequentSubgraph(occ_grid, "vector_incomplete")
    frequent_graph.build_graph(threshold=4)

    frequent_graph.prune_graph()

    print("Number of nodes in the graph:", frequent_graph.graph.number_of_nodes())
    print("Number of edges in the graph:", frequent_graph.graph.number_of_edges())

    x_init = snap_to_grid([2, 2], map_resolution)
    x_goal = snap_to_grid([75, 97], map_resolution)
    problem = AStar_With_Graph([0, 0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, frequent_graph.graph, resolution=map_resolution)

    problem_status = problem.solve(plot=False)
    if problem_status:
        print("Path found!")

        plt.figure(2)
        occ_grid.plot_grid_and_path(problem.path)
        # occ_grid.plot_smoothed_path(problem.smoothed_path)
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