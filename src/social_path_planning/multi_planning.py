import cvxpy as cp
import time
import pickle
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.a_star import AStar
from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.utils import *

NOMINAL_VELOCITY = 0.35  # m/s
TIME_STEP = 5  # seconds
MAX_VELOCITY = 0.7 # m/s
ROBOT_DIAMETER = 0.5  # meters
M = 1e6  # Big-M constant for constraints
DELTA = 1  # Safety margin in seconds
MAX_ACCELERATION = 0.2  # m/s^2 cap on |v_{k+1} - v_k| / dt


def _scalar_times_from_solver(prob, time_var, context):
    """Read one cvxpy time vector after ``prob.solve()``; raise if infeasible / no primal."""
    val = time_var.value
    if val is None:
        raise RuntimeError(
            f"{context}: no solution (cvxpy status={getattr(prob, 'status', None)!r}); likely infeasible or unbounded"
        )
    return val.tolist()


def _multi_agent_times_from_solver(prob, agent_times, context):
    """Read per-agent time vectors after ``prob.solve()``; raise if any primal is missing."""
    out = []
    for i, at in enumerate(agent_times):
        val = at.value
        if val is None:
            raise RuntimeError(
                f"{context}: no solution for agent {i} (cvxpy status={getattr(prob, 'status', None)!r}); likely infeasible or unbounded"
            )
        out.append(val.tolist())
    return out


def _add_acceleration_constraints(constraints, time_var, path, v_max, a_max):
    """
    Add convex acceleration constraints using per-segment speed variables.

    For segment k:
      dt_k = t_{k+1} - t_k
      d_k  = ||p_{k+1} - p_k||
      v_k  = average segment speed (decision variable)

    Constraints:
      - 0 <= v_k <= v_max
      - d_k <= v_k * dt_k (convex via geo_mean form)
      - |v_{k+1} - v_k| <= a_max * 0.5 * (dt_k + dt_{k+1})
    """
    num_segments = len(path) - 1
    if num_segments <= 0:
        return None

    seg_speeds = cp.Variable(num_segments, nonneg=True)
    constraints += [seg_speeds <= v_max]

    dists = [
        float(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])))
        for i in range(num_segments)
    ]
    for i in range(num_segments):
        dt_k = time_var[i + 1] - time_var[i]
        if dists[i] <= 0.0:
            continue
        # Enforce dt_k * v_k >= d_k in a DCP-compliant way.
        constraints += [cp.geo_mean(cp.hstack([dt_k, seg_speeds[i]])) >= np.sqrt(dists[i])]

    for i in range(num_segments - 1):
        dt_k = time_var[i + 1] - time_var[i]
        dt_kp1 = time_var[i + 2] - time_var[i + 1]
        dt_mid = 0.5 * (dt_k + dt_kp1)
        constraints += [cp.abs(seg_speeds[i + 1] - seg_speeds[i]) <= a_max * dt_mid]

    return seg_speeds


def _segment_endpoint_conflict(p0, p1, q0, q1, threshold):
    """
    True if any endpoint of segment (p0, p1) is within threshold of any endpoint of (q0, q1).

    Endpoint-only C-space overlap; does not detect crossing segments whose interiors overlap
    while all four endpoints remain farther apart than threshold.
    """
    for pa in (p0, p1):
        for qb in (q0, q1):
            if np.linalg.norm(np.array(pa) - np.array(qb)) <= threshold:
                return True
    return False


def _segment_shares_exact_vertex(p0, p1, q0, q1):
    """True if any segment endpoint equals any endpoint of the other segment."""
    for pa in (p0, p1):
        for qb in (q0, q1):
            if pa == qb:
                return True
    return False


def _iter_conflicting_segment_pairs(path_a, path_b, threshold):
    """Yield (segment_index_i, segment_index_j) for conflicting segment pairs."""
    if len(path_a) < 2 or len(path_b) < 2:
        return
    for i in range(len(path_a) - 1):
        for j in range(len(path_b) - 1):
            if _segment_endpoint_conflict(
                path_a[i], path_a[i + 1], path_b[j], path_b[j + 1], threshold
            ):
                yield i, j


def _collect_segment_collision_pairs(
    collision_pairs, path_a, path_b, threshold, prefix=(), start_z_index=0
):
    """
    Append segment collision tuples with z_index, reusing z for consecutive shared-vertex segments.

    Returns:
        num_z: total number of binary z variables required after this batch
    """
    z_index = start_z_index
    prev_reuse = False
    prev_i = None
    for i, j in _iter_conflicting_segment_pairs(path_a, path_b, threshold):
        p0, p1 = path_a[i], path_a[i + 1]
        q0, q1 = path_b[j], path_b[j + 1]
        current_reuse = _segment_shares_exact_vertex(p0, p1, q0, q1)
        if current_reuse and prev_reuse and prev_i is not None and i == prev_i + 1:
            z_index -= 1
        collision_pairs.append(prefix + (i, j, z_index))
        z_index += 1
        prev_reuse = current_reuse
        prev_i = i
    return z_index


class MultiAgentPlanner:

    def __init__(self, occupancy_grid: StochOccupancyGrid2D, v):
        """
        Initialize the multi-agent planner.

        Args:
            occupancy_grid (StochOccupancyGrid2D): The occupancy grid of the environment.
        """
        self.occupancy_grid = occupancy_grid
        self.v = v if v is not None else MAX_VELOCITY

    def assign_path(self, path):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def assign_velocities(self, v):
        """
        Assign the velocities for all agents.

        Args:
            v (list of floats): Max velocities for each agent.
        Returns:
            None
        """
        self.v = v

    def find_collision_pairs(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def plan(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class MultiAgentSequentialPlanner(MultiAgentPlanner):

    def __init__(self, occupancy_grid: StochOccupancyGrid2D, other_agent_paths, other_agent_times, path=None, v=None):
        """
        Initialize the Sequential multi-agent planner.

        Args:
            occupancy_grid (StochOccupancyGrid2D): The occupancy grid of the environment.
            other_agent_paths (list of list of tuples): Paths of other agents in the environment.
            other_agent_times (list of list of floats): Time steps corresponding to other agent paths.
            path (list of tuples, optional): The path for the agent to follow. Defaults to None.
        """
        super().__init__(occupancy_grid, v)

        self.path = path  # Ego path
        self.other_agent_paths = other_agent_paths  # List of other agent paths
        self.other_agent_times = other_agent_times  # List of other agent arrival times

    def assign_path(self, path):
        """
        Assign the ego path for the planner.
        """
        self.path = path

    def find_collision_pairs(self):
        """
        Identify conflicting segment pairs with fixed other-agent schedules.

        Returns:
            collision_pairs: list of (other_idx, i, j, z_index)
                other_idx: index into other_agent_paths / other_agent_times
                i: ego segment index (occupancy [t_i, t_{i+1}])
                j: other-agent segment index (fixed [other_times[j], other_times[j+1]])
                z_index: binary variable for the ordering disjunction
            num_z: total number of z variables needed
        """
        collision_pairs = []
        z_index = 0
        for other_idx, other_path in enumerate(self.other_agent_paths):
            z_index = _collect_segment_collision_pairs(
                collision_pairs,
                self.path,
                other_path,
                ROBOT_DIAMETER,
                prefix=(other_idx,),
                start_z_index=z_index,
            )
        return collision_pairs, z_index

    def plan(self, verbose=False):
        if self.path is None:
            raise ValueError("Path not assigned. Please assign a path before planning.")

        n = len(self.path)
        if n < 1:
            raise ValueError("ego path is empty for sequential planner")
        if n == 1:
            return [0.0]

        t = cp.Variable(n)  # CP variable for time to reach each waypoint
        constraints = []
        constraints += [t[0] == 0]  # Start at time 0

        # Max velocity constraints (also enforces t_i+1 >= t_i)
        for i in range(n - 1):
            delta_pos = np.linalg.norm(np.array(self.path[i + 1]) - np.array(self.path[i]))
            constraints += [t[i + 1] - t[i] >= delta_pos / self.v]
        _add_acceleration_constraints(constraints, t, self.path, self.v, MAX_ACCELERATION)

        # Collision Avoiding Constraints using Big-M method (segment occupancy intervals)
        collision_pairs, max_z = self.find_collision_pairs()
        z = cp.Variable(max_z, boolean=True) if max_z > 0 else None
        for (other_idx, i, j, z_index) in collision_pairs:
            if z is None:
                continue
            other_times = self.other_agent_times[other_idx]
            constraints += [t[i + 1] <= other_times[j] - DELTA + M * z[z_index]]
            constraints += [other_times[j + 1] <= t[i] - DELTA - M * (1 - z[z_index])]

        objective = cp.Minimize(t[-1])  # Minimize time to reach final point
        prob = cp.Problem(objective, constraints)
        print("Starting to solve multi-agent planning problem...")
        prob.solve(verbose=verbose, solver=cp.ECOS_BB)
        return _scalar_times_from_solver(prob, t, "MultiAgentSequentialPlanner")


class MultiAgentSimultaneousPlanner(MultiAgentPlanner):

    def __init__(self, occupancy_grid: StochOccupancyGrid2D, paths=None, norm=1, v=None):
        """
        Initialize the Simultaneous multi-agent planner.

        Args:
            occupancy_grid (StochOccupancyGrid2D): The occupancy grid of the environment.
            paths (list of list of tuples, optional): Paths for all agents. Defaults to None.
            norm (int, optional): Norm to minimize (1, 2, or inf). Defaults to 1.
            v (list of floats, optional): Max velocities for each agent. Defaults to None.
        """
        super().__init__(occupancy_grid, v)
        self.paths = paths  # List of paths for all agents
        self.norm = norm  # Norm to minimize (1, 2, or inf)
        self.constraint_timing_records = []

    def assign_path(self, paths):
        """
        Assign the paths for all agents.

        Args:
            paths (list of list of tuples): Paths for all agents.
                Each path is a list of (x, y) tuples.
                paths is a list of such paths.

        Returns:
            None
        """
        self.paths = paths

    def find_collision_pairs(self):
        """
        Identify conflicting segment pairs between agents (Section 3.3 interval occupancy).

        Returns:
            collision_pairs: list of (a1, a2, i, j, z_index)
                a1, a2: agent indices
                i, j: segment indices (occupancy [t_i, t_{i+1}], [t_j, t_{j+1}])
                z_index: binary variable for the ordering disjunction
            num_z: total number of z variables needed
        """
        collision_pairs = []
        z_index = 0
        num_agents = len(self.paths)
        pair_detection_stats = {}
        count_before = 0

        for a1 in range(num_agents):
            for a2 in range(a1 + 1, num_agents):
                pair_t0 = time.perf_counter()
                count_before = len(collision_pairs)
                z_index = _collect_segment_collision_pairs(
                    collision_pairs,
                    self.paths[a1],
                    self.paths[a2],
                    0.5,
                    prefix=(a1, a2),
                    start_z_index=z_index,
                )
                pair_detection_stats[(a1, a2)] = {
                    "robot_i": a1,
                    "robot_j": a2,
                    "pair_detect_time_s": float(time.perf_counter() - pair_t0),
                    "collision_tuple_count": int(len(collision_pairs) - count_before),
                }

        self._pair_detection_stats = pair_detection_stats
        return collision_pairs, z_index

    def plan(self, verbose=False):
        if self.paths is None:
            raise ValueError("Paths not assigned. Please assign paths before planning.")

        if len(self.v) == 1:
            self.v = [self.v[0] for _ in range(len(self.paths))]

        # Create CP variables for each agent's time to reach each waypoint
        agent_times = [cp.Variable(len(path)) for path in self.paths]
        constraints = []
        for agent_time in agent_times:
            constraints += [agent_time[0] == 0]  # All agents start at same time (t=0)

        # Max velocity constraints (also enforces t_i+1 >= t_i)
        for agent_idx, path in enumerate(self.paths):
            for i in range(len(path) - 1):
                delta_pos = np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
                constraints += [agent_times[agent_idx][i + 1] - agent_times[agent_idx][i] >= delta_pos / self.v[agent_idx]]
            _add_acceleration_constraints(
                constraints,
                agent_times[agent_idx],
                path,
                self.v[agent_idx],
                MAX_ACCELERATION,
            )

        # Collision Avoiding Constraints using Big-M method
        collision_pairs, max_z = self.find_collision_pairs()  # Get all the collision pairs
        z = cp.Variable(max_z, boolean=True) if max_z > 0 else None
        pair_constraint_build_time = defaultdict(float)
        pair_constraint_count = defaultdict(int)
        for (a1, a2, i, j, z_index) in collision_pairs:
            if z is None:
                continue
            build_t0 = time.perf_counter()
            constraints += [agent_times[a1][i + 1] <= agent_times[a2][j] - DELTA + M * z[z_index]]
            constraints += [agent_times[a2][j + 1] <= agent_times[a1][i] - DELTA - M * (1 - z[z_index])]
            pair_key = (a1, a2)
            pair_constraint_build_time[pair_key] += float(time.perf_counter() - build_t0)
            pair_constraint_count[pair_key] += 2

        detection_stats = getattr(self, "_pair_detection_stats", {})
        all_pair_keys = sorted(
            set(detection_stats.keys()) | set(pair_constraint_build_time.keys()),
            key=lambda pair: (pair[0], pair[1])
        )
        self.constraint_timing_records = []
        for pair_key in all_pair_keys:
            detect_row = detection_stats.get(pair_key, {})
            self.constraint_timing_records.append(
                {
                    "robot_i": int(pair_key[0]),
                    "robot_j": int(pair_key[1]),
                    "pair_detect_time_s": float(detect_row.get("pair_detect_time_s", 0.0)),
                    "pair_constraint_build_time_s": float(pair_constraint_build_time.get(pair_key, 0.0)),
                    "collision_tuple_count": int(detect_row.get("collision_tuple_count", 0)),
                    "constraint_count": int(pair_constraint_count.get(pair_key, 0)),
                }
            )

        final_time_vars = cp.hstack([agent_time[-1] for agent_time in agent_times])
        objective = cp.Minimize(cp.norm(final_time_vars, p=self.norm))  # Minimize norm of final times
        prob = cp.Problem(objective, constraints)
        print("Starting to solve multi-agent planning problem...")
        prob.solve(verbose=verbose, solver=cp.ECOS_BB)
        return _multi_agent_times_from_solver(prob, agent_times, "MultiAgentSimultaneousPlanner")

class MultiAgentCombinedPlanner(MultiAgentPlanner):

    def __init__(self, occupancy_grid: StochOccupancyGrid2D, other_agent_paths, other_agent_times, paths=None, v=None, norm=1):
        super().__init__(occupancy_grid, v)

        self.other_agent_paths = other_agent_paths  # List of other agent paths
        self.other_agent_times = other_agent_times  # List of other agent arrival times
        self.paths = paths  # List of paths to find control sequences for
        self.norm = norm

    def assign_path(self, paths):
        """
        Assign the paths for all agents.

        Args:
            paths (list of list of tuples): Paths for all agents.
                Each path is a list of (x, y) tuples.
                paths is a list of such paths.

        Returns:
            None
        """
        self.paths = paths

    def find_collision_pairs(self):
        z_index = 0
        num_agents = len(self.paths)

        sequential_pairs = []
        for other_idx, other_path in enumerate(self.other_agent_paths):
            for a, path in enumerate(self.paths):
                z_index = _collect_segment_collision_pairs(
                    sequential_pairs,
                    path,
                    other_path,
                    ROBOT_DIAMETER,
                    prefix=(other_idx, a),
                    start_z_index=z_index,
                )

        simultaneous_pairs = []
        for a1 in range(num_agents):
            for a2 in range(a1 + 1, num_agents):
                z_index = _collect_segment_collision_pairs(
                    simultaneous_pairs,
                    self.paths[a1],
                    self.paths[a2],
                    ROBOT_DIAMETER,
                    prefix=(a1, a2),
                    start_z_index=z_index,
                )

        return (sequential_pairs, simultaneous_pairs), z_index

    def plan(self, verbose=False):
        if self.paths is None:
            raise ValueError("Paths not assigned. Please assign paths before planning.")

        if len(self.v) == 1:
            self.v = [self.v[0] for _ in range(len(self.paths))]

        # Create CP variables for each agent's time to reach each waypoint
        agent_times = [cp.Variable(len(path)) for path in self.paths]
        constraints = []
        for agent_time in agent_times:
            constraints += [agent_time[0] == 0]  # All agents start at same time (t=0)

        # Max velocity constraints (also enforces t_i+1 >= t_i)
        for agent_idx, path in enumerate(self.paths):
            for i in range(len(path) - 1):
                delta_pos = np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
                constraints += [
                    agent_times[agent_idx][i + 1] - agent_times[agent_idx][i] >= delta_pos / self.v[agent_idx]]
            _add_acceleration_constraints(
                constraints,
                agent_times[agent_idx],
                path,
                self.v[agent_idx],
                MAX_ACCELERATION,
            )

        # Collision Avoiding Constraints using Big-M method
        collision_pairs, max_z = self.find_collision_pairs()  # Get all the collision pairs
        sequential_pairs, simultaneous_pairs = collision_pairs
        z = cp.Variable(max_z, boolean=True) if max_z > 0 else None
        for (other_idx, a, i, j, z_index) in sequential_pairs:
            if z is None:
                continue
            other_times = self.other_agent_times[other_idx]
            constraints += [agent_times[a][i + 1] <= other_times[j] - DELTA + M * z[z_index]]
            constraints += [other_times[j + 1] <= agent_times[a][i] - DELTA - M * (1 - z[z_index])]
        for (a1, a2, i, j, z_index) in simultaneous_pairs:
            if z is None:
                continue
            constraints += [agent_times[a1][i + 1] <= agent_times[a2][j] - DELTA + M * z[z_index]]
            constraints += [agent_times[a2][j + 1] <= agent_times[a1][i] - DELTA - M * (1 - z[z_index])]

        final_time_vars = cp.hstack([agent_time[-1] for agent_time in agent_times])
        objective = cp.Minimize(cp.norm(final_time_vars, p=self.norm))  # Minimize norm of final times
        prob = cp.Problem(objective, constraints)
        print("Starting to solve multi-agent planning problem...")
        prob.solve(verbose=verbose, solver=cp.ECOS_BB)
        return _multi_agent_times_from_solver(prob, agent_times, "MultiAgentCombinedPlanner")

def get_position_at_time(t, path, time_points):
    """
    Returns (x, y) at time t using linear interpolation.
    If t is outside the range of time_points, it returns the start or end pos.
    """
    # Extract x and y lists
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    # Check if the path hasn't started or has finished
    # (Optional: return None if you want points to disappear)
    if t < time_points[0]:
        return xs[0], ys[0]
    if t > time_points[-1]:
        return xs[-1], ys[-1]

    # Interpolate
    x = np.interp(t, time_points, xs)
    y = np.interp(t, time_points, ys)
    return x, y


def path_to_arc_length(path):
    """
    Convert a path [(x0, y0), (x1, y1), ...] to cumulative distance values.
    """
    if path is None or len(path) == 0:
        return []

    arc_lengths = [0.0]
    for i in range(1, len(path)):
        segment = np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        arc_lengths.append(arc_lengths[-1] + segment)
    return arc_lengths


def create_space_time_plot(paths, times, output_file="space_time_sequential.png", title="Space-Time Plot", dpi=300):
    """
    Create and save a static space-time plot for multiple agents.
    x-axis: time [s], y-axis: distance along path [m].
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.brg(np.linspace(0, 0.9, len(paths)))
    title_fs = 22
    label_fs = 20
    tick_fs = 20
    legend_fs = 20

    for i, (path, time_seq, color) in enumerate(zip(paths, times, colors)):
        s_values = path_to_arc_length(path)
        if len(s_values) != len(time_seq):
            raise ValueError(
                f"Length mismatch for path {i + 1}: "
                f"{len(s_values)} arc-length points vs {len(time_seq)} time points."
            )
        ax.plot(time_seq, s_values, color=color, linewidth=4, label=f"Robot {i + 1}")

    ax.set_xlabel("Time [s]", fontsize=label_fs)
    ax.set_ylabel("Distance Along Path [m]", fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=legend_fs)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    print(f"Saved space-time plot to {output_file}")


def create_map_context_plot(
        paths,
        occ_grid=None,
        times=None,
        snapshot_time=None,
        output_file="map_context.png",
        title="Path Context Map",
        dpi=300):
    """
    Create and save a static map plot with robot paths.
    Optionally overlays robot positions at snapshot_time.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    title_fs = 18
    label_fs = 15
    tick_fs = 15
    legend_fs = 15

    if occ_grid is not None:
        occ_grid.plot_grid(ax=ax)

    colors = plt.cm.brg(np.linspace(0, 0.9, len(paths)))
    path_handles = []
    path_labels = []
    for i, (path, color) in enumerate(zip(paths, colors)):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        path_line, = ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, label=f"Robot {i + 1}")
        path_handles.append(path_line)
        path_labels.append(f"Robot {i + 1}")

        # Start/goal markers for context in the paper figure.
        ax.scatter(xs[0], ys[0], marker="o", s=70, color=color, edgecolors=color, linewidths=0.8)
        ax.scatter(xs[-1], ys[-1], marker="*", s=180, color=color, edgecolors=color, linewidths=0.8)

        if times is not None and snapshot_time is not None:
            rx, ry = get_position_at_time(snapshot_time, path, times[i])
            ax.scatter(rx, ry, marker="s", s=75, color=color, edgecolors=color, linewidths=1.0)

    if snapshot_time is not None:
        ax.text(
            0.02,
            0.98,
            f"Robot positions at t = {snapshot_time:.2f}s",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
            fontsize=legend_fs,
        )

    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel("x [m]", fontsize=label_fs)
    ax.set_ylabel("y [m]", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    path_legend = ax.legend(path_handles, path_labels, loc="upper right", fontsize=legend_fs)
    marker_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="green",
               markeredgecolor="gray", markersize=8, label="Start"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="blue",
               markeredgecolor="gray", markersize=8, label="Current"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="gold",
               markeredgecolor="gray", markersize=12, label="Goal"),
    ]
    marker_legend = ax.legend(handles=marker_handles, loc="lower right", fontsize=legend_fs)
    path_legend.get_title().set_fontsize(legend_fs)
    marker_legend.get_title().set_fontsize(legend_fs)
    ax.add_artist(path_legend)
    ax.add_artist(marker_legend)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    print(f"Saved map context plot to {output_file}")


def create_video(paths, times, output_file="video.gif", occ_grid=None):
    """
    Create a video visualizing the multi-agent paths over time.
    """
    # --- 1. Determine global time bounds ---
    all_times = [t for sublist in times for t in sublist]
    start_time = min(all_times)
    end_time = max(all_times)

    # --- 2. Setup Figure ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Path Visualization")
    ax.grid(True, linestyle='--', alpha=0.6)
    if occ_grid is not None:
        occ_grid.plot_grid(ax=ax)

    # Determine axis limits automatically based on all coordinates
    all_coords = [p for sublist in paths for p in sublist]
    all_xs = [p[0] for p in all_coords]
    all_ys = [p[1] for p in all_coords]
    pad = 1

    # --- 3. Initialize lines (trails) and points ---
    colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
    lines = []
    points = []

    for i, color in enumerate(colors):
        # The trail line
        line, = ax.plot([], [], color=color, alpha=0.5, linewidth=1)
        lines.append(line)

        # The current head point
        point, = ax.plot([], [], marker='o', color=color, markersize=8, label=f'Path {i + 1}')
        points.append(point)

    ax.legend(loc='upper right')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # --- 4. Define Animation Update Functions ---
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]

    def update(frame):
        current_time = frame

        for i, (path, time_seq) in enumerate(zip(paths, times)):
            # Get current head position
            curr_x, curr_y = get_position_at_time(current_time, path, time_seq)
            points[i].set_data([curr_x], [curr_y])

            # Draw trail (history up to current time)
            history_x = []
            history_y = []

            # Add past fixed waypoints
            for (px, py), pt in zip(path, time_seq):
                if pt <= current_time:
                    history_x.append(px)
                    history_y.append(py)

            # Add current interpolated position to connect the line smoothly
            history_x.append(curr_x)
            history_y.append(curr_y)

            lines[i].set_data(history_x, history_y)

        time_text.set_text(f'Time: {current_time:.2f}')
        return lines + points + [time_text]

    # --- 5. Run Animation ---
    total_frames = 200
    interval_ms = 50
    frames = np.linspace(start_time, end_time, total_frames)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=interval_ms
    )

    if output_file:
        print(f"Saving animation to {output_file}...")
        # Note: Saving as .mp4 requires ffmpeg to be installed.
        # Saving as .gif requires Pillow.
        try:
            ani.save(output_file, fps=1000 / interval_ms)
            print("Save complete.")
        except Exception as e:
            print(f"Could not save file: {e}")
            print("Ensure ffmpeg is installed for video, or try saving as .gif")

if __name__ == "__main__":

    # Test the MultiAgentPlanner with dummy data
    scenario_name = "sample2_default"
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    occ_grid = StochOccupancyGrid2D(map_resolution, round(map_size[0]/map_resolution), round(map_size[1]/map_resolution), 0, 0, 10, occ.T)

    # Load path1.pkl if it exists
    try:
        with open("path1.pkl", "rb") as f:
            path1 = pickle.load(f)
    except FileNotFoundError:
        # Generate a path for the agent 1
        x_init = snap_to_grid([2, 25], map_resolution)
        x_goal = snap_to_grid([97, 50], map_resolution)
        problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)
        problem_status = problem.solve()
        path1 = problem.path if problem_status else None
        with open("path1.pkl", "wb") as f:
            pickle.dump(path1, f)
        # occ_grid.plot_grid_and_path(path1)
        # plt.show()

    # Load path2.pkl if it exists
    try:
        with open("path2.pkl", "rb") as f:
            path2 = pickle.load(f)
    except FileNotFoundError:
        # Agent 2 paths and times
        # x_init = snap_to_grid([25, 2], map_resolution)
        # x_goal = snap_to_grid([50, 97], map_resolution)
        x_init = snap_to_grid([2, 40], map_resolution)
        x_goal = snap_to_grid([97, 50], map_resolution)
        problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)
        problem_status = problem.solve()
        path2 = problem.path if problem_status else None
        with open("path2.pkl", "wb") as f:
            pickle.dump(path2, f)
        # occ_grid.plot_grid_and_path(path2)
        # plt.show()

    # Load path3.pkl if it exists
    try:
        with open("path3.pkl", "rb") as f:
            path3 = pickle.load(f)
    except FileNotFoundError:
        # Agent 3 paths and times
        x_init = snap_to_grid([50, 80], map_resolution)
        x_goal = snap_to_grid([97, 20], map_resolution)
        problem = AStar([0,0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid, resolution=map_resolution)
        problem_status = problem.solve()
        path3 = problem.path if problem_status else None
        with open("path3.pkl", "wb") as f:
            pickle.dump(path3, f)
        occ_grid.plot_grid_and_path(path3)
        plt.show()

    # Load path4.pkl if it exists
    try:
        with open("path4.pkl", "rb") as f:
            path4 = pickle.load(f)
    except FileNotFoundError:
        # Agent 4 paths and times
        x_init = snap_to_grid([50, 20], map_resolution)
        x_goal = snap_to_grid([97, 75], map_resolution)
        problem = AStar([0, 0], snap_to_grid(map_size, map_resolution), x_init, x_goal, occ_grid,
                        resolution=map_resolution)
        problem_status = problem.solve()
        path4 = problem.path if problem_status else None
        with open("path4.pkl", "wb") as f:
            pickle.dump(path4, f)
        occ_grid.plot_grid_and_path(path4)
        plt.show()

    ############## Sequential Path Planning Example ##############
    # Assign uniform time steps for the other agent (path2)
    path2_times = [i * map_resolution * (1 / NOMINAL_VELOCITY) for i in range(len(path2))]
    path3_times = [i * map_resolution * (1 / (NOMINAL_VELOCITY/1.5)) for i in range(len(path3))]
    path4_times = [i * map_resolution * (1 / (NOMINAL_VELOCITY)) for i in range(len(path4))]

    # Create the planner
    planner = MultiAgentSequentialPlanner(occ_grid, [path2, path3, path4], [path2_times, path3_times, path4_times], path=path1)

    # Plan and visualize
    # times = planner.plan()
    # create_video([path1, path2, path3, path4], [times, path2_times, path3_times, path4_times], occ_grid=occ_grid, output_file="video_sequential.gif")

    ############## Simultaneous Path Planning Example ##############
    # Reduce granularity of paths for faster solving
    # path1 = path1[::5]
    # path2 = path2[::5]

    # Make the planner
    planner = MultiAgentSimultaneousPlanner(occ_grid, paths=[path1, path2, path3, path4])
    planner.assign_velocities([MAX_VELOCITY, MAX_VELOCITY/1.2, MAX_VELOCITY/1.5, MAX_VELOCITY/1.65])

    # Plan and visualize
    times = planner.plan()
    create_space_time_plot(
        [path1, path2, path3, path4],
        times,
        output_file="space_time_simultaneous.png",
        title="Simultaneous Planning Space-Time Plot"
    )
    snapshot_time = 0.345 * max(t_seq[-1] for t_seq in times)
    create_map_context_plot(
        [path1, path2, path3, path4],
        occ_grid=occ_grid,
        times=times,
        snapshot_time=snapshot_time,
        output_file="map_context_simultaneous.png",
        title="Simultaneous Planning Paths on Map"
    )
    # create_video([path1, path2, path3, path4], times, occ_grid=occ_grid, output_file="video_simultaneous.gif")

    ############## Combined Path Planning Example ##############
    # Assign uniform time steps for the already planned paths (path2/path3)
    path2_times = [i * map_resolution * (1 / (1.3*NOMINAL_VELOCITY)) for i in range(len(path2))]
    path3_times = [i * map_resolution * (1 / (NOMINAL_VELOCITY / 1.2)) for i in range(len(path3))]

    # Create the planner
    planner = MultiAgentCombinedPlanner(occ_grid, [path2, path3], [path2_times, path3_times], paths=[path1, path4])
    planner.assign_velocities([MAX_VELOCITY, MAX_VELOCITY / 2])

    # Plan and visualize
    # times = planner.plan()
    # create_video([path1, path2, path3, path4], [times[0], path2_times, path3_times, times[1]], occ_grid=occ_grid, output_file="video_combined.gif")
