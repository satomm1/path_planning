"""
Demonstrate temporary-obstacle protocol: local overlay, tagged detour paths,
separate temporary heatmap, and adapted social-graph preview.

Example (build baseline inline, then accumulate temporary detours):

    python -m social_path_planning.demo_temporary_obstacles \\
        --scenario sample2_default \\
        --num-baseline-paths 80 \\
        --num-temp-paths 40 \\
        --batch-k 5 \\
        --output-dir outputs_temp_obstacles

Use a saved social graph as baseline (recommended):

    python -m social_path_planning.demo_temporary_obstacles \\
        --scenario sample2_default \\
        --baseline-graph results/grid2_timeline/graph_0007.pkl \\
        --num-temp-paths 40

Use a pre-built timeline heatmap checkpoint (rebuilds H from heatmap):

    python -m social_path_planning.demo_temporary_obstacles \\
        --scenario sample2_default \\
        --baseline-dir outputs_timeline_test \\
        --num-temp-paths 40 \\
        --output-dir outputs_temp_obstacles

Revisit instantly from cache (no replanning); add more paths later:

    python -m social_path_planning.demo_temporary_obstacles \\
        --baseline-dir outputs_timeline_test \\
        --num-temp-paths 10 --replay-only

    python -m social_path_planning.demo_temporary_obstacles \\
        --baseline-dir outputs_timeline_test \\
        --num-temp-paths 20
        # plans only paths 11-20, reuses cached 1-10

Manual obstacle placement via CLI (overrides config file):

    python -m social_path_planning.demo_temporary_obstacles \\
        --temp-rect 46,54,18,22 \\
        --num-temp-paths 30
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from social_path_planning.a_star import AStar, AStar_With_Graph
from social_path_planning.compare_astar import build_occ_grid, generate_random_free_point
from social_path_planning.heat_map import HeatMap2DVector
from social_path_planning.planning_grid import PlanningGrid
from social_path_planning.social_knowledge import SocialKnowledgeStore
from social_path_planning.sparse_graph import CHECKPOINT_FILENAME, METADATA_FILENAME
from social_path_planning.temp_demo_path_cache import (
    build_cache_fingerprint,
    default_cache_path,
    load_path_cache,
    make_cache_payload,
    obstacle_specs_from_overlay,
    records_from_cache,
    restore_planning_rng,
    save_path_cache,
    serialize_planning_rng,
    validate_cache_fingerprint,
)
from social_path_planning.temporary_overlay import (
    DEFAULT_PROXIMITY_RADIUS_M,
    TemporaryObstacleOverlay,
    apply_obstacle_config,
    load_temporary_obstacle_config,
)

METADATA_FILENAME_DEMO = "run_metadata.json"
TEMP_OBSTACLE_OVERLAY_COLOR = "#E53935"
TEMP_OBSTACLE_OVERLAY_ALPHA = 0.975

# --- Standalone bottom-right (adapted graph) figure: edit these as needed ---
SAVE_STANDALONE_ADAPTED_GRAPH = True
ADAPTED_GRAPH_FIGSIZE = (2.33, 2.45)
ADAPTED_GRAPH_DPI = 450
ADAPTED_GRAPH_EDGE_COLOR = "darkorange"
ADAPTED_GRAPH_EDGE_LINEWIDTH = 0.75
ADAPTED_GRAPH_EDGE_ALPHA = 0.85
ADAPTED_GRAPH_SHOW_TEMP_HEATMAP = True
ADAPTED_GRAPH_HEATMAP_MIN_INTENSITY = 5.0
ADAPTED_GRAPH_HEATMAP_LEGEND = False
ADAPTED_GRAPH_TITLE_FONTSIZE = 18
ADAPTED_GRAPH_LABEL_FONTSIZE = 28
ADAPTED_GRAPH_TICK_FONTSIZE = 28
ADAPTED_GRAPH_LEGEND_FONTSIZE = 8


class _NumpyUniformRng:
    def __init__(self, seed: int, *, rng_state: dict[str, Any] | None = None):
        if rng_state is not None:
            self._rng = restore_planning_rng(rng_state)
        else:
            self._rng = np.random.default_rng(seed)

    def uniform(self, low, high):
        return self._rng.uniform(low=low, high=high)

    def state_dict(self) -> dict[str, Any]:
        return serialize_planning_rng(self._rng)


def _load_baseline_knowledge(
    store: SocialKnowledgeStore,
    baseline_dir: str | None,
    *,
    baseline_graph_file: str | None = None,
    load_heatmap_for_viz: bool = False,
    graph_threshold: float,
    min_component_size: int,
) -> tuple[int, str | None]:
    """
    Load permanent baseline knowledge.

    Returns ``(path_count_proxy, resolved_graph_path)``.

    When ``baseline_graph_file`` is set, loads the pickled social graph directly
    (does not rebuild H from heatmap).
    """
    graph_path: str | None = None

    if baseline_graph_file is not None:
        graph_path = os.path.abspath(baseline_graph_file)
        if not os.path.isfile(graph_path):
            raise FileNotFoundError(f"Baseline graph not found: {graph_path}")
        with open(graph_path, "rb") as graph_fp:
            store.social_graph.graph = pickle.load(graph_fp)

        if load_heatmap_for_viz and baseline_dir is not None:
            checkpoint = os.path.join(baseline_dir, CHECKPOINT_FILENAME)
            if os.path.isfile(checkpoint):
                store.load_permanent_checkpoint(checkpoint)

        timeline_dir = baseline_dir or os.path.dirname(graph_path)
        metadata_path = os.path.join(timeline_dir, METADATA_FILENAME)
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            path_proxy = int(meta.get("cumulative_solved_paths", 0))
        else:
            path_proxy = store.social_graph.graph.number_of_edges()
        return path_proxy, graph_path

    if baseline_dir is None:
        return 0, None

    checkpoint = os.path.join(baseline_dir, CHECKPOINT_FILENAME)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {checkpoint}. "
            "Use --baseline-graph for a saved social graph, "
            "or run sparse_graph timeline first."
        )
    store.load_permanent_checkpoint(checkpoint)
    store.rebuild_social_graph(threshold=graph_threshold, min_component_size=min_component_size)
    metadata_path = os.path.join(baseline_dir, METADATA_FILENAME)
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("cumulative_solved_paths", 0)), None
    return int(np.sum(store.permanent_heatmap.heatmap)), None


def _has_permanent_heatmap(store: SocialKnowledgeStore) -> bool:
    return bool(np.any(store.permanent_heatmap.heatmap))


def _build_baseline(
    store: SocialKnowledgeStore,
    occ_grid,
    statespace_hi,
    map_resolution: float,
    num_paths: int,
    seed: int,
    *,
    graph_threshold: float,
    min_component_size: int,
) -> int:
    rng = _NumpyUniformRng(seed)
    solved = 0
    while solved < num_paths:
        x_init = generate_random_free_point(occ_grid, rng)
        x_goal = generate_random_free_point(occ_grid, rng)
        problem = AStar([0, 0], statespace_hi, x_init, x_goal, occ_grid, resolution=map_resolution)
        if not problem.solve(mode="modified"):
            continue
        store.add_path(problem.path, temporary=False)
        solved += 1
    store.rebuild_social_graph(threshold=graph_threshold, min_component_size=min_component_size)
    return solved


def _parse_temp_rect(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--temp-rect expects xmin,xmax,ymin,ymax")
    return {
        "id": "cli_rect",
        "shape": "rect",
        "x": [parts[0], parts[1]],
        "y": [parts[2], parts[3]],
        "active": True,
    }


def _parse_temp_circle(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--temp-circle expects cx,cy,radius")
    return {
        "id": "cli_circle",
        "shape": "circle",
        "center": [parts[0], parts[1]],
        "radius": parts[2],
        "active": True,
    }


def _plot_temp_obstacle_overlay(
    ax,
    occ_grid,
    overlay: TemporaryObstacleOverlay,
    *,
    zorder: int = 5,
) -> None:
    """Draw temporary obstacle mask on an axis that already has map content."""
    if not overlay.has_active_obstacles():
        return
    mask = overlay.mask.astype(float)
    mask[mask == 0] = np.nan
    obstacle_cmap = ListedColormap([TEMP_OBSTACLE_OVERLAY_COLOR])
    ax.imshow(
        mask,
        origin="lower",
        extent=occ_grid.extent,
        cmap=obstacle_cmap,
        vmin=0.5,
        vmax=1.5,
        alpha=TEMP_OBSTACLE_OVERLAY_ALPHA,
        interpolation="nearest",
        zorder=zorder,
    )


def _plot_overlay_hatch(ax, occ_grid, overlay: TemporaryObstacleOverlay) -> None:
    occ_grid.plot_grid(ax=ax)
    _plot_temp_obstacle_overlay(ax, occ_grid, overlay, zorder=3)
    ax.set_title("Map + Temporary Overlay", fontsize=13)


def _plot_temp_paths(ax, paths: list[list[tuple[float, float]]], *, alpha: float = 0.7) -> None:
    _plot_classified_paths(ax, affected_paths=paths, unaffected_paths=[], alpha=alpha)


def _plot_classified_paths(
    ax,
    *,
    affected_paths: list[list[tuple[float, float]]],
    unaffected_paths: list[list[tuple[float, float]]],
    alpha: float = 0.85,
) -> None:
    cmap = plt.cm.tab10
    for idx, path in enumerate(affected_paths):
        if len(path) < 2:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "--", color=cmap(idx % 10), linewidth=1.4, alpha=alpha, zorder=4)
    for path in unaffected_paths:
        if len(path) < 2:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(
            xs,
            ys,
            ":",
            color="#555555",
            linewidth=1.6,
            alpha=min(alpha, 0.9),
            zorder=3,
        )


def _adapted_graph_legend_handles(
    *,
    edge_color: str,
    edge_linewidth: float,
    edge_alpha: float,
) -> list:
    return [
        Patch(
            facecolor=TEMP_OBSTACLE_OVERLAY_COLOR,
            alpha=TEMP_OBSTACLE_OVERLAY_ALPHA,
            label="Obstacle",
        ),
        Line2D(
            [0],
            [0],
            color=edge_color,
            linestyle="-",
            linewidth=edge_linewidth,
            alpha=edge_alpha,
            label="Temporary Paths",
        ),
    ]


def _decorate_adapted_graph_panel(
    ax,
    occ_grid,
    overlay: TemporaryObstacleOverlay,
    *,
    edge_color: str,
    edge_linewidth: float,
    edge_alpha: float,
    legend_fontsize: float = 8,
) -> None:
    _plot_temp_obstacle_overlay(ax, occ_grid, overlay)
    ax.legend(
        handles=_adapted_graph_legend_handles(
            edge_color=edge_color,
            edge_linewidth=edge_linewidth,
            edge_alpha=edge_alpha,
        ),
        loc="upper right",
        fontsize=legend_fontsize,
    )


def _register_replanned_path(
    store: SocialKnowledgeStore,
    overlay: TemporaryObstacleOverlay,
    path: list[tuple[float, float]],
    *,
    robot_id: int,
    step: int,
    proximity_radius_m: float,
    cached_obstacle_affected: bool | None = None,
) -> bool:
    if cached_obstacle_affected is None:
        affected = overlay.path_within_proximity(path, proximity_radius_m)
    else:
        affected = bool(cached_obstacle_affected)
    store.broadcast_path(
        robot_id=robot_id,
        path=path,
        temporary=True,
        obstacle_affected=affected,
    )
    return affected


def save_adapted_graph_figure(
    output_path: str,
    *,
    occ_grid,
    overlay: TemporaryObstacleOverlay,
    store: SocialKnowledgeStore,
    adapted_graph,
    adapted_threshold: float,
    title: str | None = None,
) -> None:
    """Save the adapted social-graph panel as its own figure (separate plot settings)."""
    fig, ax = plt.subplots(figsize=ADAPTED_GRAPH_FIGSIZE)
    heatmap_layer = store.temporary_heatmap if ADAPTED_GRAPH_SHOW_TEMP_HEATMAP else None
    adapted_graph.visualize_graph(
        ax=ax,
        show=False,
        title="",
        edge_color=ADAPTED_GRAPH_EDGE_COLOR,
        edge_linewidth=ADAPTED_GRAPH_EDGE_LINEWIDTH,
        edge_alpha=ADAPTED_GRAPH_EDGE_ALPHA,
        title_fontsize=ADAPTED_GRAPH_TITLE_FONTSIZE,
        label_fontsize=ADAPTED_GRAPH_LABEL_FONTSIZE,
        tick_fontsize=ADAPTED_GRAPH_TICK_FONTSIZE,
        heatmap_layer=heatmap_layer,
        min_visible_intensity=ADAPTED_GRAPH_HEATMAP_MIN_INTENSITY,
        heatmap_legend=ADAPTED_GRAPH_HEATMAP_LEGEND,
    )
    _decorate_adapted_graph_panel(
        ax,
        occ_grid,
        overlay,
        edge_color=ADAPTED_GRAPH_EDGE_COLOR,
        edge_linewidth=ADAPTED_GRAPH_EDGE_LINEWIDTH,
        edge_alpha=ADAPTED_GRAPH_EDGE_ALPHA,
        legend_fontsize=ADAPTED_GRAPH_LEGEND_FONTSIZE,
    )
    if title is None:
        ax.set_title(
            f"Adapted Graph Preview\n(threshold={adapted_threshold}, affected paths only)",
            fontsize=ADAPTED_GRAPH_TITLE_FONTSIZE,
        )
    # Turn off ax ticks and x/y labels
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=ADAPTED_GRAPH_DPI, bbox_inches="tight")
    plt.close(fig)


def save_demo_frame(
    output_path: str,
    *,
    occ_grid,
    overlay: TemporaryObstacleOverlay,
    store: SocialKnowledgeStore,
    adapted_graph,
    recent_affected_paths: list[list[tuple[float, float]]],
    recent_unaffected_paths: list[list[tuple[float, float]]],
    n_affected: int,
    n_unaffected: int,
    cumulative_replans: int,
    step_label: str,
    adapted_threshold: float,
    proximity_radius_m: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    _plot_overlay_hatch(axes[0, 0], occ_grid, overlay)
    _plot_classified_paths(
        axes[0, 0],
        affected_paths=recent_affected_paths,
        unaffected_paths=recent_unaffected_paths,
    )

    heatmap_layer = store.permanent_heatmap if _has_permanent_heatmap(store) else None
    store.social_graph.visualize_graph(
        ax=axes[0, 1],
        show=False,
        title="Permanent Social Graph H",
        edge_color="red",
        edge_linewidth=1.5,
        heatmap_layer=heatmap_layer,
        heatmap_legend=False,
    )
    if heatmap_layer is not None:
        title_suffix = "(unchanged by temp paths)"
    else:
        title_suffix = "(loaded from file; unchanged by temp paths)"
    axes[0, 1].set_title(f"Permanent Social Graph H\n{title_suffix}", fontsize=13)
    _plot_temp_obstacle_overlay(axes[0, 1], occ_grid, overlay)

    store.temporary_heatmap.plot_heatmap(ax=axes[1, 0], show=False, add_legend=False)
    _plot_classified_paths(
        axes[1, 0],
        affected_paths=recent_affected_paths,
        unaffected_paths=[],
        alpha=0.9,
    )
    _plot_temp_obstacle_overlay(axes[1, 0], occ_grid, overlay)
    axes[1, 0].set_title(
        f"Temporary Heatmap\n"
        f"(affected={n_affected}, unaffected replans={n_unaffected}, "
        f"radius={proximity_radius_m:g} m)",
        fontsize=13,
    )

    store.temporary_heatmap.plot_heatmap(ax=axes[1, 1], show=False, add_legend=False)
    adapted_graph.visualize_graph(
        ax=axes[1, 1],
        show=False,
        title="Adapted Social Graph (from temp heatmap)",
        edge_color="darkorange",
        edge_linewidth=1.5,
    )
    _decorate_adapted_graph_panel(
        axes[1, 1],
        occ_grid,
        overlay,
        edge_color="darkorange",
        edge_linewidth=1.5,
        edge_alpha=0.75,
    )
    axes[1, 1].set_title(
        f"Adapted Graph Preview\n(threshold={adapted_threshold}, affected paths only)",
        fontsize=13,
    )

    legend_handles = [
        Patch(
            facecolor=TEMP_OBSTACLE_OVERLAY_COLOR,
            alpha=TEMP_OBSTACLE_OVERLAY_ALPHA,
            label="Temporary obstacle",
        ),
        Line2D([0], [0], color="C0", linestyle="--", linewidth=1.4, label="Affected (in temp heatmap)"),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=":",
            linewidth=1.6,
            label="Unaffected replan (not in temp heatmap)",
        ),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.suptitle(f"{step_label} | replans={cumulative_replans}", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _reset_temporary_knowledge(store: SocialKnowledgeStore) -> None:
    store.temporary_heatmap = HeatMap2DVector(store.occ_grid)
    store.broadcasts = [b for b in store.broadcasts if not b.temporary]


def _plan_temp_detour(
    *,
    planning_grid,
    statespace_hi,
    map_resolution: float,
    social_graph,
    rng: _NumpyUniformRng,
) -> tuple[list[tuple[float, float]], tuple[float, float], tuple[float, float], float] | None:
    x_init = generate_random_free_point(planning_grid, rng)
    x_goal = generate_random_free_point(planning_grid, rng)
    problem = AStar_With_Graph(
        [0, 0],
        statespace_hi,
        x_init,
        x_goal,
        planning_grid,
        social_graph,
        resolution=map_resolution,
    )
    t0 = time.perf_counter()
    if not problem.solve():
        return None
    elapsed = time.perf_counter() - t0
    return list(problem.path), x_init, x_goal, elapsed


def _maybe_save_frame(
    *,
    target: int,
    num_temp_paths: int,
    batch_k: int,
    snapshot_idx: int,
    output_dir: str,
    occ_grid,
    overlay,
    store,
    adapted_threshold: float,
    min_component_size: int,
    recent_affected_paths: list,
    recent_unaffected_paths: list,
    n_affected: int,
    n_unaffected: int,
    attempted: int,
    proximity_radius_m: float,
    frame_paths: list[str],
    adapted_graph_frame_paths: list[str],
) -> int:
    if target % batch_k != 0 and target != num_temp_paths:
        return snapshot_idx
    snapshot_idx += 1
    adapted = store.build_adapted_graph(
        threshold=adapted_threshold,
        min_component_size=min_component_size,
        source="temporary",
    )
    frame_path = os.path.join(output_dir, f"frame_{snapshot_idx:04d}.png")
    save_demo_frame(
        frame_path,
        occ_grid=occ_grid,
        overlay=overlay,
        store=store,
        adapted_graph=adapted,
        recent_affected_paths=recent_affected_paths,
        recent_unaffected_paths=recent_unaffected_paths,
        n_affected=n_affected,
        n_unaffected=n_unaffected,
        cumulative_replans=target,
        step_label=f"Replans: {target}/{num_temp_paths} (attempted={attempted})",
        adapted_threshold=adapted_threshold,
        proximity_radius_m=proximity_radius_m,
    )
    frame_paths.append(frame_path)
    if SAVE_STANDALONE_ADAPTED_GRAPH:
        adapted_path = os.path.join(output_dir, f"adapted_graph_{snapshot_idx:04d}.png")
        save_adapted_graph_figure(
            adapted_path,
            occ_grid=occ_grid,
            overlay=overlay,
            store=store,
            adapted_graph=adapted,
            adapted_threshold=adapted_threshold,
            title=f"Adapted Graph | replans={target}/{num_temp_paths}",
        )
        adapted_graph_frame_paths.append(adapted_path)
    store.save_checkpoint(output_dir)
    print(
        f"[frame {snapshot_idx:04d}] replans={target} affected={n_affected} "
        f"unaffected={n_unaffected} temp_heat_sum={np.sum(store.temporary_heatmap.heatmap):.0f} "
        f"adapted_nodes={adapted.graph.number_of_nodes()} file={frame_path}"
    )
    return snapshot_idx


def run_temporary_obstacle_demo(
    scenario_name: str = "sample2_default",
    baseline_dir: str | None = None,
    baseline_graph_file: str | None = None,
    baseline_heatmap_for_viz: bool = False,
    num_baseline_paths: int = 0,
    num_temp_paths: int = 40,
    batch_k: int = 5,
    graph_threshold: float = 1.0,
    adapted_threshold: float = 1.0,
    min_component_size: int = 15,
    output_dir: str = "outputs_temp_obstacles",
    seed: int = 0,
    obstacle_config_path: str | None = None,
    cli_rect: str | None = None,
    cli_circle: str | None = None,
    promotion_threshold: int | None = None,
    path_cache: str | None = None,
    use_path_cache: bool = True,
    force_replan: bool = False,
    replay_only: bool = False,
    proximity_radius_m: float = DEFAULT_PROXIMITY_RADIUS_M,
) -> dict[str, Any]:
    if batch_k <= 0:
        raise ValueError("batch_k must be > 0")

    os.makedirs(output_dir, exist_ok=True)
    occ_grid, map_size, map_resolution, statespace_hi = build_occ_grid(scenario_name)

    overlay = TemporaryObstacleOverlay(occ_grid)
    cli_rect_spec = _parse_temp_rect(cli_rect)
    cli_circle_spec = _parse_temp_circle(cli_circle)
    if cli_rect_spec:
        overlay.add_obstacle(cli_rect_spec)
    elif cli_circle_spec:
        overlay.add_obstacle(cli_circle_spec)
    else:
        config = load_temporary_obstacle_config(obstacle_config_path)
        if config.get("scenario") and config["scenario"] != scenario_name:
            print(
                f"Warning: obstacle config scenario '{config['scenario']}' "
                f"differs from --scenario '{scenario_name}'."
            )
        apply_obstacle_config(overlay, config)

    planning_grid = PlanningGrid(occ_grid, overlay)
    store = SocialKnowledgeStore(occ_grid)

    baseline_paths, resolved_graph_path = _load_baseline_knowledge(
        store,
        baseline_dir,
        baseline_graph_file=baseline_graph_file,
        load_heatmap_for_viz=baseline_heatmap_for_viz,
        graph_threshold=graph_threshold,
        min_component_size=min_component_size,
    )
    if baseline_paths > 0 or resolved_graph_path is not None:
        print(
            f"Loaded permanent baseline: paths_proxy={baseline_paths}, "
            f"graph nodes={store.social_graph.graph.number_of_nodes()}, "
            f"edges={store.social_graph.graph.number_of_edges()}"
            + (f", graph_file={resolved_graph_path}" if resolved_graph_path else "")
        )
    elif num_baseline_paths <= 0:
        raise ValueError(
            "No permanent social graph available. Provide --baseline-graph, "
            "--baseline-dir with heatmap checkpoint, "
            "or set --num-baseline-paths > 0 (slow)."
        )
    if baseline_paths == 0 and num_baseline_paths > 0:
        print(f"Building baseline with {num_baseline_paths} permanent paths...")
        baseline_paths = _build_baseline(
            store,
            occ_grid,
            statespace_hi,
            map_resolution,
            num_baseline_paths,
            seed,
            graph_threshold=graph_threshold,
            min_component_size=min_component_size,
        )
        baseline_checkpoint = os.path.join(output_dir, "baseline_permanent_heatmap.npy")
        np.save(baseline_checkpoint, store.permanent_heatmap.heatmap)
        print(
            f"Baseline ready: {baseline_paths} paths, "
            f"graph nodes={store.social_graph.graph.number_of_nodes()}, "
            f"edges={store.social_graph.graph.number_of_edges()}"
        )

    obstacle_specs = obstacle_specs_from_overlay(overlay)
    fingerprint = build_cache_fingerprint(
        scenario_name=scenario_name,
        seed=seed,
        obstacle_specs=obstacle_specs,
        permanent_heatmap=store.permanent_heatmap.heatmap,
        baseline_dir=baseline_dir,
        baseline_graph_file=resolved_graph_path,
        proximity_radius_m=proximity_radius_m,
        graph_threshold=graph_threshold,
        min_component_size=min_component_size,
    )
    cache_path = path_cache or default_cache_path(output_dir)
    cached_records: list[dict[str, Any]] = []
    cache_hits = 0
    cache_planned = 0
    cached_payload: dict[str, Any] | None = None
    restored_rng_state: dict[str, Any] | None = None
    restored_attempts = 0

    if use_path_cache and not force_replan:
        cached_payload = load_path_cache(cache_path)
        if cached_payload is not None and validate_cache_fingerprint(cached_payload, fingerprint):
            cached_records = records_from_cache(cached_payload)
            restored_rng_state = cached_payload.get("rng_state")
            restored_attempts = int(cached_payload.get("planning_attempts", 0))
            if cached_records:
                print(
                    f"Loaded {len(cached_records)} cached temporary paths from {cache_path}"
                )
                if restored_rng_state is None:
                    print(
                        "Warning: cache has no RNG state (legacy cache). "
                        "New paths may duplicate earlier start/goal pairs; "
                        "use --force-replan to rebuild the cache."
                    )
        elif cached_payload is not None:
            print(
                f"Path cache at {cache_path} ignored (scenario/obstacle/baseline changed). "
                "Use --force-replan to replace it."
            )

    if replay_only:
        if not use_path_cache or not cached_records:
            raise FileNotFoundError(
                f"--replay-only requires a valid path cache at {cache_path}. "
                "Run once without --replay-only to build the cache."
            )
        if len(cached_records) < num_temp_paths:
            raise ValueError(
                f"Cache has {len(cached_records)} paths but --num-temp-paths={num_temp_paths}. "
                "Run without --replay-only to plan more."
            )

    if force_replan:
        cached_records = []

    if replay_only:
        records = cached_records[:num_temp_paths]
    else:
        records = list(cached_records)

    _reset_temporary_knowledge(store)
    rng = _NumpyUniformRng(seed + 1, rng_state=restored_rng_state if not force_replan else None)
    attempted = restored_attempts if (restored_rng_state is not None and not force_replan) else 0
    snapshot_idx = 0
    recent_affected_paths: list[list[tuple[float, float]]] = []
    recent_unaffected_paths: list[list[tuple[float, float]]] = []
    n_affected = 0
    n_unaffected = 0
    frame_paths: list[str] = []
    adapted_graph_frame_paths: list[str] = []
    promoted = False

    for target in range(1, num_temp_paths + 1):
        cached_affected = None
        newly_planned = False
        if not force_replan and target <= len(records):
            rec = records[target - 1]
            path = rec["path"]
            cached_affected = rec.get("obstacle_affected")
            cache_hits += 1
        else:
            if replay_only:
                raise RuntimeError("Internal error: replay-only requested planning.")
            planned = None
            while planned is None:
                attempted += 1
                planned = _plan_temp_detour(
                    planning_grid=planning_grid,
                    statespace_hi=statespace_hi,
                    map_resolution=map_resolution,
                    social_graph=store.social_graph.graph,
                    rng=rng,
                )
            path, x_init, x_goal, elapsed = planned
            rec = {
                "route_id": target,
                "x_init": x_init,
                "x_goal": x_goal,
                "path": path,
                "solve_time_s": float(elapsed),
                "obstacle_affected": None,
            }
            if target <= len(records):
                records[target - 1] = rec
            else:
                records.append(rec)
            cache_planned += 1
            newly_planned = True

        affected = _register_replanned_path(
            store,
            overlay,
            path,
            robot_id=target,
            step=target,
            proximity_radius_m=proximity_radius_m,
            cached_obstacle_affected=cached_affected,
        )
        rec["obstacle_affected"] = affected

        if newly_planned and use_path_cache:
            save_path_cache(
                cache_path,
                make_cache_payload(
                    fingerprint=fingerprint,
                    scenario_name=scenario_name,
                    seed=seed,
                    obstacle_specs=obstacle_specs,
                    records=records,
                    rng_state=rng.state_dict(),
                    planning_attempts=attempted,
                ),
            )
            status = "affected" if affected else "unaffected"
            print(
                f"Planned replan {target}/{num_temp_paths} in {rec['solve_time_s']:.2f}s "
                f"({rec['x_init']} -> {rec['x_goal']}) [{status}]"
            )

        if affected:
            n_affected += 1
            recent_affected_paths.append(list(path))
            if len(recent_affected_paths) > 8:
                recent_affected_paths.pop(0)
        else:
            n_unaffected += 1
            recent_unaffected_paths.append(list(path))
            if len(recent_unaffected_paths) > 8:
                recent_unaffected_paths.pop(0)

        if promotion_threshold is not None and not promoted:
            n_promoted = store.promote_temporary(promotion_threshold)
            if n_promoted > 0:
                store.rebuild_social_graph(
                    threshold=graph_threshold,
                    min_component_size=min_component_size,
                )
                promoted = True
                print(f"Promoted {n_promoted} temporary heat cells at path {target}.")

        snapshot_idx = _maybe_save_frame(
            target=target,
            num_temp_paths=num_temp_paths,
            batch_k=batch_k,
            snapshot_idx=snapshot_idx,
            output_dir=output_dir,
            occ_grid=occ_grid,
            overlay=overlay,
            store=store,
            adapted_threshold=adapted_threshold,
            min_component_size=min_component_size,
            recent_affected_paths=recent_affected_paths,
            recent_unaffected_paths=recent_unaffected_paths,
            n_affected=n_affected,
            n_unaffected=n_unaffected,
            attempted=attempted,
            proximity_radius_m=proximity_radius_m,
            frame_paths=frame_paths,
            adapted_graph_frame_paths=adapted_graph_frame_paths,
        )

    solved_temp = num_temp_paths

    adapted_final = store.build_adapted_graph(
        threshold=adapted_threshold,
        min_component_size=min_component_size,
        source="temporary",
    )
    summary_path = os.path.join(output_dir, "summary_comparison.png")
    save_demo_frame(
        summary_path,
        occ_grid=occ_grid,
        overlay=overlay,
        store=store,
        adapted_graph=adapted_final,
        recent_affected_paths=recent_affected_paths,
        recent_unaffected_paths=recent_unaffected_paths,
        n_affected=n_affected,
        n_unaffected=n_unaffected,
        cumulative_replans=solved_temp,
        step_label=f"Final: {n_affected} affected, {n_unaffected} unaffected replans",
        adapted_threshold=adapted_threshold,
        proximity_radius_m=proximity_radius_m,
    )

    adapted_graph_summary_path = None
    if SAVE_STANDALONE_ADAPTED_GRAPH:
        adapted_graph_summary_path = os.path.join(output_dir, "adapted_graph_summary.png")
        save_adapted_graph_figure(
            adapted_graph_summary_path,
            occ_grid=occ_grid,
            overlay=overlay,
            store=store,
            adapted_graph=adapted_final,
            adapted_threshold=adapted_threshold,
            title=(
                f"Adapted Graph | {n_affected} affected, "
                f"{n_unaffected} unaffected replans"
            ),
        )

    result = {
        "scenario_name": scenario_name,
        "baseline_paths": baseline_paths,
        "baseline_graph_file": resolved_graph_path,
        "proximity_radius_m": proximity_radius_m,
        "num_temp_paths": num_temp_paths,
        "temp_paths_solved": solved_temp,
        "temp_paths_affected": n_affected,
        "temp_paths_unaffected": n_unaffected,
        "temp_paths_attempted": attempted,
        "batch_k": batch_k,
        "graph_threshold": graph_threshold,
        "adapted_threshold": adapted_threshold,
        "min_component_size": min_component_size,
        "promotion_threshold": promotion_threshold,
        "promoted": promoted,
        "output_dir": output_dir,
        "seed": seed,
        "path_cache": cache_path,
        "cache_hits": cache_hits,
        "cache_planned": cache_planned,
        "replay_only": replay_only,
        "frames": frame_paths,
        "adapted_graph_frames": adapted_graph_frame_paths,
        "adapted_graph_summary": adapted_graph_summary_path,
        "summary_figure": summary_path,
        **store.summary(),
    }
    meta_path = os.path.join(output_dir, METADATA_FILENAME_DEMO)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    store.save_metadata(output_dir, extra=result)
    print(f"Saved metadata: {meta_path}")
    print(f"Summary figure: {summary_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate temporary obstacle protocol with dual heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default="sample2_default", help="Grid scenario name.")
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Optional timeline directory (heatmap checkpoint for legacy mode or viz underlay).",
    )
    parser.add_argument(
        "--baseline-graph",
        default=None,
        help="Path to a saved social graph pickle (e.g. results/grid2_timeline/graph_0007.pkl).",
    )
    parser.add_argument(
        "--baseline-heatmap-for-viz",
        action="store_true",
        help="With --baseline-graph, also load heatmap_checkpoint.npy from --baseline-dir for underlay.",
    )
    parser.add_argument(
        "--num-baseline-paths",
        type=int,
        default=0,
        help="Permanent paths to build if --baseline-dir is not provided (0 = require baseline-dir).",
    )
    parser.add_argument(
        "--num-temp-paths",
        type=int,
        default=40,
        help="Number of temporary detour paths to plan and broadcast.",
    )
    parser.add_argument("--batch-k", type=int, default=5, help="Save a frame every K temp paths.")
    parser.add_argument("--graph-threshold", type=float, default=1.0, help="Permanent graph edge threshold.")
    parser.add_argument(
        "--adapted-threshold",
        type=float,
        default=1.0,
        help="Threshold for adapted preview graph from temporary heatmap.",
    )
    parser.add_argument("--min-component-size", type=int, default=15, help="Graph pruning minimum component size.")
    parser.add_argument("--output-dir", default="outputs_temp_obstacles", help="Output directory for figures.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--obstacle-config",
        default=None,
        help="JSON file with manual temporary obstacle definitions.",
    )
    parser.add_argument(
        "--temp-rect",
        default=None,
        help="CLI rect obstacle: xmin,xmax,ymin,ymax (overrides config file).",
    )
    parser.add_argument(
        "--temp-circle",
        default=None,
        help="CLI circle obstacle: cx,cy,radius (overrides config file).",
    )
    parser.add_argument(
        "--proximity-radius",
        type=float,
        default=DEFAULT_PROXIMITY_RADIUS_M,
        help="Path is temp-heatmap eligible if within this distance (m) of the obstacle.",
    )
    parser.add_argument(
        "--promotion-threshold",
        type=int,
        default=None,
        help="If set, promote temp heat cells >= threshold into permanent H.",
    )
    parser.add_argument(
        "--path-cache",
        default=None,
        help="JSON path cache file (default: <output-dir>/temp_paths_cache.json).",
    )
    parser.add_argument(
        "--no-path-cache",
        action="store_true",
        help="Disable loading/saving planned path cache.",
    )
    parser.add_argument(
        "--force-replan",
        action="store_true",
        help="Ignore existing cache and replan all temporary paths.",
    )
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Regenerate figures from cache only (no A* planning).",
    )
    args = parser.parse_args()

    run_temporary_obstacle_demo(
        scenario_name=args.scenario,
        baseline_dir=args.baseline_dir,
        baseline_graph_file=args.baseline_graph,
        baseline_heatmap_for_viz=args.baseline_heatmap_for_viz,
        num_baseline_paths=args.num_baseline_paths,
        num_temp_paths=args.num_temp_paths,
        batch_k=args.batch_k,
        graph_threshold=args.graph_threshold,
        adapted_threshold=args.adapted_threshold,
        min_component_size=args.min_component_size,
        output_dir=args.output_dir,
        seed=args.seed,
        obstacle_config_path=args.obstacle_config,
        cli_rect=args.temp_rect,
        cli_circle=args.temp_circle,
        promotion_threshold=args.promotion_threshold,
        path_cache=args.path_cache,
        use_path_cache=not args.no_path_cache,
        force_replan=args.force_replan,
        replay_only=args.replay_only,
        proximity_radius_m=args.proximity_radius,
    )


if __name__ == "__main__":
    main()
