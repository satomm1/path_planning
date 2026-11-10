"""Encounters with no safe ordering (both goals, or both starts, inside the window)."""

import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _em():
    # Imported lazily so this module does not load cvxpy before other tests stub it.
    import social_path_planning.event_multi_planning as em

    return em


def _have_real_cvxpy():
    try:
        import cvxpy as cp
    except ImportError:
        return False
    return hasattr(cp, "installed_solvers") and callable(cp.installed_solvers) and not type(cp.Variable).__name__.endswith("Mock")


def line(p0, p1, step=0.2):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    n = int(round(np.linalg.norm(p1 - p0) / step))
    return [tuple(map(float, np.round(p0 + (p1 - p0) * k / n, 6))) for k in range(n + 1)]


# Robot a drives east and stops at x = 5; robot b drives south and stops 0.2 m above it.
BOTH_GOALS = [line((0, 0), (5, 0)), line((5, 8), (5, 0.2))]
# The same routes reversed: both robots start inside the shared window.
BOTH_STARTS = [list(reversed(BOTH_GOALS[0])), list(reversed(BOTH_GOALS[1]))]
# Only robot b parks in a's path: b must arrive last (existing fixed-order rule).
ONE_GOAL = [line((0, 0), (10, 0)), line((5, 8), (5, 0.2))]


def min_separation(em, analysis, event_times):
    paths, times = em.build_constant_velocity_schedules(analysis, event_times)
    ts = np.arange(0.0, max(t[-1] for t in times) + 0.02, 0.02)
    pos = [np.stack([np.interp(ts, t, np.asarray(p)[:, 0]), np.interp(ts, t, np.asarray(p)[:, 1])], 1)
           for p, t in zip(paths, times)]
    return float(np.min(np.linalg.norm(pos[0] - pos[1], axis=1)))


@unittest.skipUnless(_have_real_cvxpy(), "cvxpy not installed")
class TestUnschedulableEncounters(unittest.TestCase):
    def _plan(self, paths):
        em = _em()
        planner = em.EventMultiAgentSimultaneousPlanner(None, paths=paths, norm=1, v=[0.7, 0.7])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                return planner, planner.plan(print_constraints=True), out
            except em.EventMilpInfeasibleError as exc:
                return planner, exc, out

    def test_both_goals_inside_window_is_infeasible(self):
        planner, result, out = self._plan(BOTH_GOALS)
        self.assertIsInstance(result, _em().EventMilpInfeasibleError)
        self.assertIn("both goals are inside the encounter window", str(result))
        self.assertIn("agents (0,1)", str(result))
        self.assertIn("NO SAFE ORDERING", out.getvalue())
        self.assertIsNone(planner.event_times)

    def test_both_starts_inside_window_is_infeasible(self):
        _, result, _ = self._plan(BOTH_STARTS)
        self.assertIsInstance(result, _em().EventMilpInfeasibleError)
        self.assertIn("both starts are inside the encounter window", str(result))

    def test_single_parked_goal_still_solved_safely(self):
        planner, times, _ = self._plan(ONE_GOAL)
        self.assertNotIsInstance(times, Exception)
        self.assertEqual(planner.unschedulable_encounters, [])
        self.assertGreaterEqual(min_separation(_em(), planner.analysis, times), 0.5 - 1e-9)

    def test_sequential_and_combined_planners_detect_it(self):
        em = _em()
        seq = em.EventMultiAgentSequentialPlanner(None, [BOTH_GOALS[1]], [[0.0]], path=BOTH_GOALS[0], v=0.7)
        analysis = seq.analyze()
        wps = analysis.agents[1].interest_waypoints
        seq.other_agent_times = [[0.0] + [(wps[k + 1].s) / 0.7 for k in range(len(wps) - 1)]]
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(em.EventMilpInfeasibleError):
                seq.plan()
        comb = em.EventMultiAgentCombinedPlanner(None, [], [], paths=BOTH_GOALS, v=[0.7, 0.7])
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(em.EventMilpInfeasibleError):
                comb.plan()

    def test_pipeline_records_failure(self):
        from social_path_planning.mapf_comparison.pipeline import _format_solver_error_status

        _, exc, _ = self._plan(BOTH_GOALS)
        self.assertTrue(_format_solver_error_status(exc).startswith("error:EventMilpInfeasibleError:"))


if __name__ == "__main__":
    unittest.main()
