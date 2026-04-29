from __future__ import annotations

import json
import time
from queue import PriorityQueue
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from social_path_planning.occupancy_grid import StochOccupancyGrid2D


def _empty_solve_telemetry():
    return {
        "mode": "unknown",
        "success": False,
        "solve_time_s": None,
        "explored_nodes": 0,
        "cost_eval_count": 0,
        "social_cost_sum": 0.0,
        "social_cost_max": 0.0,
        "social_nonzero_edges": 0,
        "path_geometric_length_m": None,
        "path_social_penalty_sum": None,
        "graph_edge_cost_evals": 0,
        "off_graph_social_evals": 0,
        "graph_weight_sum": 0.0,
        "off_graph_edge_cost_sum": 0.0,
        "path_segments_on_graph": 0,
        "path_segments_total": 0,
        "path_fraction_on_graph": None,
    }


def _telemetry_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _telemetry_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_telemetry_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""
    SUPPORTED_SOLVER_MODES = {"modified", "vanilla"}

    def __init__(
        self,
        statespace_lo,
        statespace_hi,
        x_init,
        x_goal,
        occupancy: StochOccupancyGrid2D,
        resolution=1,
        robot_d=0.4,
        desired_dist_right_extra=0.25,
    ):
        self.statespace_lo = np.array(statespace_lo)  # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)  # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy  # occupancy grid (a StochOccupancyGrid2D object)
        self.resolution = resolution  # resolution of the discretization of state space (cell/m)
        self.x_init = self.snap_to_grid(x_init)  # initial state
        self.x_goal = self.snap_to_grid(x_goal)  # goal state
        self.robot_d = robot_d  # robot diameter (m)
        self.desired_dist_right_extra = desired_dist_right_extra  # added to robot_d/2 for desired standoff from right wall (m)

        self.closed_set = set()  # the set containing the states that have been visited
        self.open_set = set()  # the set containing the states that are condidate for future expension
        self.came_from = {}  # dictionary keeping track of each state's parent to reconstruct the path
        self.est_cost_through = {}
        self.cost_to_arrive = {}
        self.dist_to_right = {}
        self.dist_to_right[self.x_init] = 0

        self.priority_queue = PriorityQueue()
        self.priority_queue.put((self.manhattan_distance(self.x_init, self.x_goal), self.x_init))

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.manhattan_distance(self.x_init, self.x_goal)

        self.path = None  # the final path as a list of states
        self.smoothed_path = None
        self.pp_plan = None  # the post-processed plan
        self.last_solve_time = None
        self.last_solve_telemetry = _empty_solve_telemetry()
        self._telemetry_collecting = False

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        """
        if self.occupancy.is_free(x) and self.statespace_lo[0] <= x[0] < self.statespace_hi[0] and self.statespace_lo[1] <= x[1] < self.statespace_hi[1]:
            return True
        else:
            return False

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance
        """
        return np.linalg.norm(np.array(x1) - np.array(x2))

    def manhattan_distance(self, x1, x2):
        """
        Computes the Manhattan distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Manhattan distance
        """
        return np.sum(np.abs(np.array(x1) - np.array(x2)))

    def h(self, x):
        return self.manhattan_distance(x, self.x_goal)

    def vanilla_cost(self, x1, x2):
        return self.distance(x1, x2), 0

    def cost(self, x1, x2, dist2right_prev=0):
        social_cost, dist2right = self.rightness_penalty(x1, x2, dist2right_prev)
        dist = self.distance(x1, x2)
        edge_total = dist + social_cost
        self._telemetry_record_edge_cost(dist, edge_total)
        return edge_total, dist2right

    def _telemetry_record_edge_cost(self, dist, edge_total):
        if not getattr(self, "_telemetry_collecting", False):
            return
        t = self.last_solve_telemetry
        t["cost_eval_count"] += 1
        social = float(edge_total - dist)
        t["social_cost_sum"] += social
        t["social_cost_max"] = max(t["social_cost_max"], social)
        if social > 1e-9:
            t["social_nonzero_edges"] += 1

    def _begin_solve_telemetry(self, mode):
        self.last_solve_telemetry = _empty_solve_telemetry()
        self.last_solve_telemetry["mode"] = mode
        self._telemetry_collecting = True

    def _finalize_solve_telemetry(self, success, mode, elapsed):
        self._telemetry_collecting = False
        t = self.last_solve_telemetry
        t["mode"] = mode
        t["success"] = bool(success)
        t["solve_time_s"] = float(elapsed)
        t["explored_nodes"] = int(len(self.closed_set))
        if success and self.path is not None and len(self.path) >= 2:
            path_len = 0.0
            social_pen = 0.0
            prev_d2r = 0.0
            for i in range(len(self.path) - 1):
                x1, x2 = self.path[i], self.path[i + 1]
                path_len += float(self.distance(x1, x2))
                pen, d2r = self.rightness_penalty(x1, x2, prev_d2r)
                social_pen += float(pen)
                prev_d2r = float(d2r)
            t["path_geometric_length_m"] = path_len
            t["path_social_penalty_sum"] = social_pen
            self._append_graph_path_metrics(t)
        else:
            t["path_geometric_length_m"] = None
            t["path_social_penalty_sum"] = None
            self._append_graph_path_metrics(t)

    def _append_graph_path_metrics(self, t):
        """Override in AStar_With_Graph to fill graph path counters."""
        pass

    def _solve_return(self, success, elapsed, return_timing, return_telemetry):
        if return_timing and return_telemetry:
            return success, elapsed, self.last_solve_telemetry
        if return_timing:
            return success, elapsed
        if return_telemetry:
            return success, self.last_solve_telemetry
        return success

    def _emit_solve_return(self, success, elapsed, mode, return_timing, return_telemetry, log_telemetry):
        self._finalize_solve_telemetry(success, mode, elapsed)
        if log_telemetry:
            self.log_last_solve_telemetry_ros()
        return self._solve_return(success, elapsed, return_timing, return_telemetry)

    def log_last_solve_telemetry_ros(self):
        """Emit ``last_solve_telemetry`` via ``rospy.loginfo`` if ROS is initialized; otherwise no-op."""
        if self.last_solve_telemetry is None:
            return
        try:
            import rospy
        except ImportError:
            return
        if not rospy.core.is_initialized():
            return
        payload = _telemetry_json_safe(self.last_solve_telemetry)
        rospy.loginfo(
            "social_path_planning AStar telemetry: %s",
            json.dumps(payload, sort_keys=True),
        )

    def rightness_penalty(self, x1, x2, dist2right_prev=0):
        """
        Computes the heuristic distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float: heuristic distance
        """
        if self.distance(x2, self.x_goal) < 2:
            # Don't penalize when near goal, may need to take non-social behavior to be able to get to goal
            return 0, 0
        elif self.distance(x2, self.x_init) < 2:
            # Don't penalize when near start, may need to take non-social behavior to be able to get to socially compliant path later
            return 0, 0

        penalty = 0

        travel_dir = np.array(x2) - np.array(x1)
        travel_dir /= np.linalg.norm(travel_dir)
        dist_to_right = self.occupancy.dist_to_wall_right(x2, travel_dir)

        if dist_to_right > 10:
            penalty += self.resolution
            # Get distance to left
            dist_to_left = self.occupancy.dist_to_wall_left(x2, travel_dir)

            if dist_to_left > 3:
                dist_to_left_prev = self.occupancy.dist_to_wall_left(x1, travel_dir)
                delta_dist_to_left = dist_to_left - dist_to_left_prev

                # To account for when you just enter an intersection and the distance to left wall jumps up dramatically
                if delta_dist_to_left < 0 or delta_dist_to_left > 10:
                    delta_dist_to_left = 0
                penalty += 5*delta_dist_to_left
            else:
                # If far from right side, we should just penalize being close to left side (and we also want to penalize
                # moving closer to the left side)
                penalty =  max(0, (4 - dist_to_left))
        else:
            dist_to_right_prev = dist2right_prev # self.occupancy.dist_to_wall_right(x1, travel_dir)
            delta_raw = dist_to_right - dist_to_right_prev

            desired_dist_right = self.robot_d / 2 + self.desired_dist_right_extra
            # Penalize being both too far and too close to the desired standoff from the right wall
            penalty = abs(dist_to_right - desired_dist_right)

            # When too far from the right wall: penalize drifting even farther (same as before, with spike suppression)
            delta_far = delta_raw
            if delta_far > 15 or delta_far < 0:
                delta_far = 0
            if dist_to_right > desired_dist_right:
                penalty += 2 * delta_far
            # When too close: penalize moving still closer to the wall (negative delta along the ray)
            elif dist_to_right < desired_dist_right and -15 < delta_raw < 0:
                penalty += 2 * (-delta_raw)
        return penalty, dist_to_right

    def leftness_penalty(self, x1, x2):
        travel_dir = (np.array(x2) - np.array(x1)) / np.linalg.norm(np.array(x2) - np.array(x1))
        dist_to_left = self.occupancy.dist_to_wall_left(x2, travel_dir)
        return dist_to_left
    
    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (self.resolution * round(x[0] / self.resolution), self.resolution * round(x[1] / self.resolution))

    def get_index(self, x):
        return int(np.round((x[0] - self.occupancy.origin_x) / self.resolution)), int(np.round((x[1] - self.occupancy.origin_y) / self.resolution))

    def get_neighbors(self, x, step_resolution=1):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES
        """
        neighbors = []
        for ii in [1, 0, -1]:
            for jj in [1, 0, -1]:
                if ii != 0 or jj != 0:
                    x0 = x[0]
                    x1 = x[1]
                    x0 += ii * self.resolution   # /(np.linalg.norm(np.array((ii, jj))))
                    x1 += jj * self.resolution   # /(np.linalg.norm(np.array((ii, jj))))
                    state = self.snap_to_grid((x0, x1))
                    if self.is_free(state):
                        neighbors.append(state)
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]

        return list(reversed(path))

    def solve(self, mode="modified", return_timing=False, return_telemetry=False, log_telemetry=True):
        if mode not in self.SUPPORTED_SOLVER_MODES:
            raise ValueError(f"Unsupported solver mode '{mode}'. Supported modes: {sorted(self.SUPPORTED_SOLVER_MODES)}")

        if mode == "vanilla":
            return self.vanilla_solve(
                return_timing=return_timing,
                return_telemetry=return_telemetry,
                log_telemetry=log_telemetry,
            )
        if mode == "modified":
            return self.modified_solve(
                return_timing=return_timing,
                return_telemetry=return_telemetry,
                log_telemetry=log_telemetry,
            )
        raise ValueError(f"Unsupported solver mode '{mode}'. Supported modes: {sorted(self.SUPPORTED_SOLVER_MODES)}")

    def modified_solve(self, return_timing=False, return_telemetry=False, log_telemetry=False):
        self._begin_solve_telemetry("modified")
        t_start = time.time()
        while self.priority_queue.qsize() > 0:
            current_cost, x_current = self.priority_queue.get()

            if x_current == self.x_goal:
                t_end = time.time()
                elapsed = t_end - t_start
                self.last_solve_time = elapsed
                self.path = self.reconstruct_path()
                self.smooth_path()
                print(f"Social A* found a path in {elapsed:.2f} seconds.")
                return self._emit_solve_return(True, elapsed, "modified", return_timing, return_telemetry, log_telemetry)

            if time.time() - t_start > 60:
                elapsed = time.time() - t_start
                self.last_solve_time = elapsed
                print("A* took too long.")
                return self._emit_solve_return(False, elapsed, "modified", return_timing, return_telemetry, log_telemetry)

            self.closed_set.add(x_current)

            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue

                # cost_x_x_neigh = self.cost(x_current, x_neigh)
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                if x_current == self.x_init:
                    dist2right_prev = 0
                else:
                    dist2right_prev = self.dist_to_right[self.came_from[x_current]]
                if x_neigh not in self.cost_to_arrive or tentative_cost_to_arrive < self.cost_to_arrive[x_neigh]:
                    cost_x_x_neigh, dist2right = self.cost(x_current, x_neigh, dist2right_prev)
                    self.dist_to_right[x_neigh] = dist2right
                    self.came_from[x_neigh] = x_current
                    self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                    self.priority_queue.put(
                        (current_cost + cost_x_x_neigh + self.h(x_neigh)
                         - self.h(x_current),
                         x_neigh)
                    )
        elapsed = time.time() - t_start
        self.last_solve_time = elapsed
        return self._emit_solve_return(False, elapsed, "modified", return_timing, return_telemetry, log_telemetry)

    def vanilla_solve(self, return_timing=False, return_telemetry=False, log_telemetry=False):
        self._begin_solve_telemetry("vanilla")
        t_start = time.time()
        while self.priority_queue.qsize() > 0:
            current_cost, x_current = self.priority_queue.get()

            if x_current == self.x_goal:
                t_end = time.time()
                elapsed = t_end - t_start
                self.last_solve_time = elapsed
                self.path = self.reconstruct_path()
                self.smooth_path()
                print(f"Vanilla A* found a path in {elapsed:.2f} seconds.")
                return self._emit_solve_return(True, elapsed, "vanilla", return_timing, return_telemetry, log_telemetry)

            if time.time() - t_start > 150:
                elapsed = time.time() - t_start
                self.last_solve_time = elapsed
                print("A* took too long.")
                return self._emit_solve_return(False, elapsed, "vanilla", return_timing, return_telemetry, log_telemetry)

            self.closed_set.add(x_current)

            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue

                # cost_x_x_neigh = self.cost(x_current, x_neigh)
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                if x_neigh not in self.cost_to_arrive or tentative_cost_to_arrive < self.cost_to_arrive[x_neigh]:
                    cost_x_x_neigh = self.distance(x_current, x_neigh)
                    self.came_from[x_neigh] = x_current
                    self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                    self.priority_queue.put(
                        (current_cost + cost_x_x_neigh + self.h(x_neigh)
                         - self.h(x_current),
                         x_neigh)
                    )
        elapsed = time.time() - t_start
        self.last_solve_time = elapsed
        return self._emit_solve_return(False, elapsed, "vanilla", return_timing, return_telemetry, log_telemetry)

    def postprocess(self):
        """
        Function for postprocessing the planned path. It examines each planned point, computes the nearest point to the right,
        and if close enough, moves the point to the right. The post-processed plan is placed into the self.pp_plan variable.
        Input: 
            None
        Output: 
            None
        """
        self.pp_path = []
        self.pp_path.append(self.path[0])

        for ii in range(len(self.path)-1):
            dist_to_right = self.rightness_penalty(self.path[ii], self.path[ii+1])
            if dist_to_right < 10.1:
                # 1. Get the start and end points of the segment
                p1 = np.array(self.path[ii])
                p2 = np.array(self.path[ii+1]) # The point we are conceptually moving
        
                # 2. Calculate the direction vector of the path segment
                direction_vec = p2 - p1
                
                # Avoid division by zero if points are identical
                dir_magnitude = np.linalg.norm(direction_vec)
                if dir_magnitude < 1e-6:
                    continue

                # 3. Get the perpendicular "RIGHT" vector
                # If direction is (dx, dy), the vector to the right is (dy, -dx)
                right_vec = np.array([direction_vec[1], -direction_vec[0]])
                
                # 4. Normalize the "right" vector to get a pure direction (unit vector)
                unit_right_vec = right_vec / np.linalg.norm(right_vec)
                
                # 5. Calculate the new point by shifting the original point to the right
                new_point = p2 + unit_right_vec * dist_to_right
                
                self.pp_path.append(self.snap_to_grid((new_point[0], new_point[1])))
            else:
                self.pp_path.append(self.path[ii+1])
        return 

    def postprocess2(self):
        """
        Function for postprocessing the planned path. It examines each planned point, computes the nearest point to the right,
        and if close enough, moves the point to the right. The post-processed plan is placed into the self.pp_plan variable.
        Input: 
            None
        Output: 
            None
        """
        self.pp_path = []
        self.pp_path.append(self.path[0])

        for ii in range(len(self.path)-1):
            dist_to_right = self.rightness_penalty(self.path[ii], self.path[ii+1])
            if dist_to_right < 10.1:
                # 1. Get the start and end points of the segment
                p1 = np.array(self.path[ii])
                p2 = np.array(self.path[ii+1]) # The point we are conceptually moving
        
                # 2. Calculate the direction vector of the path segment
                direction_vec = p2 - p1
                
                # Avoid division by zero if points are identical
                dir_magnitude = np.linalg.norm(direction_vec)
                if dir_magnitude < 1e-6:
                    continue

                # 3. Get the perpendicular "RIGHT" vector
                # If direction is (dx, dy), the vector to the right is (dy, -dx)
                right_vec = np.array([direction_vec[1], -direction_vec[0]])
                
                # 4. Normalize the "right" vector to get a pure direction (unit vector)
                unit_right_vec = right_vec / np.linalg.norm(right_vec)
                
                # 5. Calculate the new point by shifting the original point to the right
                new_point = p2 + unit_right_vec * dist_to_right
                
                self.pp_path.append((new_point[0], new_point[1]))
            else:
                self.pp_path.append(self.path[ii+1])
        return

    def smooth_path(self):
        """
        Function for smoothing the planned path by interpolating between points using a spline
        """
        path = np.array(self.path)
        t = np.linspace(0, 1, len(path))
        cs_x = CubicSpline(t, path[:, 0])
        cs_y = CubicSpline(t, path[:, 1])
        t_new = np.linspace(0, 1, num=5*len(path))
        smoothed_path = [(cs_x(ti), cs_y(ti)) for ti in t_new]
        self.smoothed_path = smoothed_path
        return

class AStar_With_Graph(AStar):
    def __init__(
        self,
        statespace_lo,
        statespace_hi,
        x_init,
        x_goal,
        occupancy: StochOccupancyGrid2D,
        graph: nx.DiGraph,
        resolution=1,
        robot_d=0.4,
        desired_dist_right_extra=0.25,
    ):
        super().__init__(
            statespace_lo,
            statespace_hi,
            x_init,
            x_goal,
            occupancy,
            resolution,
            robot_d,
            desired_dist_right_extra,
        )
        self.graph = graph

    def cost(self, x1, x2, dist2right_prev=0):
        """
        This modified cost function penalizes depending on whether x1->x2 is in
        self.graph or not. If in the graph, the cost is the nominal distance.
        If not in the graph, we follow the usual rightness_penalty function to
        compute a social cost.
        """
        dist = self.distance(x1, x2)
        x1x, x1y = self.get_index(x1)
        x2x, x2y = self.get_index(x2)
        if self.graph.has_edge((x1x, x1y), (x2x, x2y)):
            w = float(self.graph[(x1x, x1y)][(x2x, x2y)]["weight"])
            if getattr(self, "_telemetry_collecting", False):
                t = self.last_solve_telemetry
                t["cost_eval_count"] += 1
                t["graph_edge_cost_evals"] += 1
                t["graph_weight_sum"] += w
                social = float(w - dist)
                t["social_cost_sum"] += social
                t["social_cost_max"] = max(t["social_cost_max"], social)
                if social > 1e-9:
                    t["social_nonzero_edges"] += 1
            return w, 0
        social_cost, dist2right = self.rightness_penalty(x1, x2, dist2right_prev)
        edge_total = dist + social_cost + 0.01
        if getattr(self, "_telemetry_collecting", False):
            t = self.last_solve_telemetry
            t["cost_eval_count"] += 1
            t["off_graph_social_evals"] += 1
            t["off_graph_edge_cost_sum"] += edge_total
            social = float(edge_total - dist)
            t["social_cost_sum"] += social
            t["social_cost_max"] = max(t["social_cost_max"], social)
            if social > 1e-9:
                t["social_nonzero_edges"] += 1
        return edge_total, dist2right

    def _append_graph_path_metrics(self, t):
        if not self.path or len(self.path) < 2:
            t["path_segments_on_graph"] = 0
            t["path_segments_total"] = 0
            t["path_fraction_on_graph"] = None
            return
        on = 0
        tot = 0
        for i in range(len(self.path) - 1):
            x1, x2 = self.path[i], self.path[i + 1]
            i1 = self.get_index(x1)
            i2 = self.get_index(x2)
            tot += 1
            if self.graph.has_edge(i1, i2):
                on += 1
        t["path_segments_on_graph"] = on
        t["path_segments_total"] = tot
        t["path_fraction_on_graph"] = (on / tot) if tot else None

    def show_path_on_graph(self):
        """
        Displays the planned path. Path nodes that are in the graph are shown in purple,
        while those not in the graph are shown in red.
        """
        self.occupancy.plot_grid()
        for ii in range(len(self.path)-1):
            x1 = self.path[ii]
            x2 = self.path[ii+1]
            x1x, x1y = self.get_index(x1)
            x2x, x2y = self.get_index(x2)
            if self.graph.has_edge((x1x, x1y), (x2x, x2y)):
                plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color='purple', linewidth=1)
            else:
                plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color='red', linewidth=1)
        plt.scatter(self.x_init[0], self.x_init[1], c='green', s=100, label='Start', zorder=5)
        plt.scatter(self.x_goal[0], self.x_goal[1], c='gold', marker="*", s=100, label='Goal', zorder=5)
        plt.show()
