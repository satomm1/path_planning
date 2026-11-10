"""route_entropy.load_social_graph builds the graph from a heatmap file (``--heatmap-file``)."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from social_path_planning.heat_map import HeatMap2DVector  # noqa: E402
from social_path_planning.occupancy_grid import StochOccupancyGrid2D  # noqa: E402


class TestLoadSocialGraphFromHeatmap(unittest.TestCase):
    def test_heatmap_file_returns_pruned_digraph(self):
        from social_path_planning import route_entropy

        probs = np.zeros((40, 60))
        grid = StochOccupancyGrid2D(0.2, 60, 40, 0.0, 0.0, 10, probs)
        heat = HeatMap2DVector(grid)
        corridor = [(0.2 * k, 4.0) for k in range(5, 55)]
        for _ in range(3):
            heat.add_path(corridor)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hm.npy"
            np.save(path, heat.heatmap)
            with contextlib.redirect_stdout(io.StringIO()):
                graph = route_entropy.load_social_graph(
                    grid, heatmap_file=str(path), sparse_graph_threshold=2.0, sparse_min_component_size=5
                )
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(graph.number_of_edges(), len(corridor) - 1)


if __name__ == "__main__":
    unittest.main()
