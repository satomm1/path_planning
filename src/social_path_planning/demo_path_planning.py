import matplotlib.pyplot as plt

from social_path_planning.a_star import AStar
from social_path_planning.grid_loader import load_grid_scenario
from social_path_planning.occupancy_grid import StochOccupancyGrid2D
from social_path_planning.utils import snap_to_grid


def _make_problem(occ_grid, statespace_hi, x_init, x_goal, map_resolution):
    return AStar(
        [0, 0],
        statespace_hi,
        x_init,
        x_goal,
        occ_grid,
        resolution=map_resolution,
    )


def _plot_path(ax, path, *, color, linestyle, label):
    xs, ys = zip(*path)
    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        linewidth=2,
        marker=",",
        zorder=5,
        alpha=0.85,
        label=label,
    )


if __name__ == "__main__":
    scenario_name = "sample2_default"
    occ, map_size, map_resolution = load_grid_scenario(scenario_name, plot=False)
    map_dim = [round(map_size[i] / map_resolution) for i in range(len(map_size))]

    occ_grid = StochOccupancyGrid2D(map_resolution, map_dim[0], map_dim[1], 0, 0, 10, occ.T)
    statespace_hi = snap_to_grid(map_size, map_resolution)

    x_init = snap_to_grid([75, 2.5], map_resolution)
    x_goal = snap_to_grid([25, 97.5], map_resolution)

    vanilla = _make_problem(occ_grid, statespace_hi, x_init, x_goal, map_resolution)
    vanilla_ok = vanilla.solve(mode="vanilla")

    modified = _make_problem(occ_grid, statespace_hi, x_init, x_goal, map_resolution)
    modified_ok = modified.solve(mode="modified")

    print(f"Vanilla A*: {'found' if vanilla_ok else 'no path'}")
    print(f"Social A*:  {'found' if modified_ok else 'no path'}")

    fig, ax = plt.subplots(figsize=(6, 6))
    occ_grid.plot_grid(ax=ax)

    if vanilla_ok and vanilla.path:
        _plot_path(
            ax,
            vanilla.path,
            color="red",
            linestyle="--",
            label="Shortest Path",
        )

    if modified_ok and modified.path:
        _plot_path(
            ax,
            modified.path,
            color="darkorange",
            linestyle="-",
            label="Social Path",
        )

    ax.scatter(x_init[0], x_init[1], c="green", s=100, label="Start", zorder=6)
    ax.scatter(x_goal[0], x_goal[1], c="gold", marker="*", s=100, label="Goal", zorder=6)

    if modified_ok:
        xs, ys = zip(*modified.closed_set)
        ax.scatter(xs, ys, c="red", marker="o", s=2, label="Explored (social)", zorder=3)

    ax.set_axis_off()

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=16)
    fig.subplots_adjust(right=0.65)
    plt.tight_layout()
    plt.savefig("./results/media/fig2a.png", dpi=600)
    plt.show()
