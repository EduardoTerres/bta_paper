"""
Tests for goal-set BTA composition in Skill Machines.

Run from skill_machines/extension/:
    python tests.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest, numpy as np, gymnasium as gym
import envs  # registers Office-v0
from sympy import sympify
from sympy.logic import boolalg
from sm import TaskPrimitive, SkillMachine, evaluate
from rm import Task
import sm_ql
from exp_convergence import _goal_satisfies_exp, GoalSetSkillPrimitive, GoalSetSkillMachine


# ─── Shared fixtures ─────────────────────────────────────────────────────────
# In the SM framework, constraints must be a subset of predicates.
# goal vector shape: (2*N,) — first N = proposition bits, next N = constraint violation bits.

PREDICATES  = ["A", "B", "c", "d"]   # N = 4
CONSTRAINTS = ["d"]                   # "d" is in PREDICATES at index 3
N = len(PREDICATES)


def _make_goal(props=(), cons=()):
    """Build a goal vector over PREDICATES/CONSTRAINTS."""
    g = np.zeros(2 * N, dtype=np.uint8)
    for name in props: g[PREDICATES.index(name)] = 1          # proposition bit
    for name in cons:  g[N + PREDICATES.index(name)] = 1      # constraint violation bit
    return g


class _MockQL:
    """QLAgent stub that returns a fixed (action, value) regardless of state."""
    def __init__(self, val, act=0):
        self._val, self._act = val, act
        self.action_space = gym.spaces.Discrete(4)

    def get_action_value(self, state):
        return np.array([self._act]), np.array([self._val])


# ─── TestGoalSatisfiesExp ────────────────────────────────────────────────────

class TestGoalSatisfiesExp(unittest.TestCase):

    def test_simple_prop_true(self):
        self.assertTrue(_goal_satisfies_exp(sympify("p_A"), _make_goal(props=["A"]),
                                            PREDICATES, CONSTRAINTS))

    def test_simple_prop_false(self):
        self.assertFalse(_goal_satisfies_exp(sympify("p_A"), _make_goal(),
                                             PREDICATES, CONSTRAINTS))

    def test_negated_constraint_satisfied(self):
        # ~c_d: True when constraint d is NOT violated
        self.assertTrue(_goal_satisfies_exp(sympify("~c_d"), _make_goal(),
                                            PREDICATES, CONSTRAINTS))

    def test_negated_constraint_violated(self):
        self.assertFalse(_goal_satisfies_exp(sympify("~c_d"), _make_goal(cons=["d"]),
                                             PREDICATES, CONSTRAINTS))

    def test_conjunction_both_true(self):
        exp = sympify("p_A & ~c_d")
        self.assertTrue(_goal_satisfies_exp(exp, _make_goal(props=["A"]),
                                            PREDICATES, CONSTRAINTS))

    def test_conjunction_constraint_fails(self):
        exp = sympify("p_A & ~c_d")
        self.assertFalse(_goal_satisfies_exp(exp, _make_goal(props=["A"], cons=["d"]),
                                             PREDICATES, CONSTRAINTS))

    def test_disjunction(self):
        exp = sympify("p_A | p_B")
        self.assertTrue(_goal_satisfies_exp(exp, _make_goal(props=["B"]),
                                            PREDICATES, CONSTRAINTS))
        self.assertFalse(_goal_satisfies_exp(exp, _make_goal(),
                                             PREDICATES, CONSTRAINTS))

    def test_boolean_true_literal(self):
        self.assertTrue(_goal_satisfies_exp(boolalg.BooleanTrue(), _make_goal(),
                                            PREDICATES, CONSTRAINTS))

    def test_boolean_false_literal(self):
        self.assertFalse(_goal_satisfies_exp(boolalg.BooleanFalse(), _make_goal(),
                                             PREDICATES, CONSTRAINTS))


# ─── TestGoalSetSkillPrimitive ───────────────────────────────────────────────

class TestGoalSetSkillPrimitive(unittest.TestCase):

    def setUp(self):
        env = gym.make("Office-v0")
        self.penv  = TaskPrimitive(env)
        self.preds = self.penv.predicates   # Office World predicates (10 items)
        self.cons  = self.penv.constraints  # ["d"]
        self.n     = len(self.preds)

    def _goal(self, props=(), cons=()):
        g = np.zeros(2 * self.n, dtype=np.uint8)
        for name in props: g[self.preds.index(name)] = 1
        for name in cons:  g[self.n + self.preds.index(name)] = 1
        return g

    def _states(self):
        return {"env_state":            np.zeros((1, 2), dtype=np.uint8),
                "violated_constraints": np.zeros((1, self.n), dtype=np.uint8)}

    def test_satisfying_goal_uses_Q_U(self):
        g  = self._goal(props=["A"])
        SP = {"1": _MockQL(1.0, act=1), "0": _MockQL(0.0, act=0)}
        gsp = GoalSetSkillPrimitive(SP, {g.tobytes(): g},
                                    self.preds, self.cons, sympify("p_A"))
        _, v = gsp.get_action_value(self._states())
        self.assertAlmostEqual(v[0], 1.0)

    def test_non_satisfying_goal_uses_Q_empty(self):
        g  = self._goal()   # A=0 → fails p_A
        SP = {"1": _MockQL(1.0, act=1), "0": _MockQL(0.0, act=0)}
        gsp = GoalSetSkillPrimitive(SP, {g.tobytes(): g},
                                    self.preds, self.cons, sympify("p_A"))
        _, v = gsp.get_action_value(self._states())
        self.assertAlmostEqual(v[0], 0.0)

    def test_best_goal_selected_across_multiple(self):
        g_bad  = self._goal()              # A=0 → Q_empty, value 0.0
        g_good = self._goal(props=["A"])   # A=1 → Q_U,     value 1.0
        goals  = {g_bad.tobytes(): g_bad, g_good.tobytes(): g_good}
        SP     = {"1": _MockQL(1.0, act=1), "0": _MockQL(0.0, act=0)}
        gsp    = GoalSetSkillPrimitive(SP, goals, self.preds, self.cons, sympify("p_A"))
        _, v   = gsp.get_action_value(self._states())
        self.assertAlmostEqual(v[0], 1.0)   # selects the satisfying goal

    def test_desired_goal_routes_to_Q_U(self):
        SP  = {"1": _MockQL(1.0, act=1), "0": _MockQL(0.0, act=0)}
        g   = self._goal(props=["A"])
        gsp = GoalSetSkillPrimitive(SP, {g.tobytes(): g},
                                    self.preds, self.cons, sympify("p_A"))
        _, v = gsp.get_action_value(self._states(), desired_goal=self._goal(props=["A"]))
        self.assertAlmostEqual(v[0], 1.0)

    def test_desired_goal_routes_to_Q_empty(self):
        SP  = {"1": _MockQL(1.0, act=1), "0": _MockQL(0.0, act=0)}
        g   = self._goal()
        gsp = GoalSetSkillPrimitive(SP, {g.tobytes(): g},
                                    self.preds, self.cons, sympify("p_A"))
        _, v = gsp.get_action_value(self._states(), desired_goal=self._goal())
        self.assertAlmostEqual(v[0], 0.0)


# ─── TestConvergenceParity ───────────────────────────────────────────────────

class TestConvergenceParity(unittest.TestCase):
    """Integration: after 50K training steps both SM and goal-set BTA solve Coffee."""

    @classmethod
    def setUpClass(cls):
        env = gym.make("Office-v0")
        cls.penv = TaskPrimitive(env)
        cls.SP   = sm_ql.learn(cls.penv, None, 50_000,
                               log_dir="/tmp/sm_test_logs/", print_freq=10**9, seed=42)
        cls.ltl  = "(F (c & X (F o))) & (G~d)"

    def _mean_return(self, SM_cls):
        task = Task(gym.make("Office-v0"), self.ltl)
        SM   = SM_cls(self.penv, self.SP, goal_directed=True)
        r, _, _ = evaluate(task, SM, epsilon=0, gamma=0.9, episodes=50, seed=42)
        return r / 50

    def test_sm_positive_return(self):
        self.assertGreater(self._mean_return(SkillMachine), 0,
                           "Standard SM should achieve positive return after 50K steps")

    def test_gsbta_positive_return(self):
        self.assertGreater(self._mean_return(GoalSetSkillMachine), 0,
                           "Goal-set BTA should achieve positive return after 50K steps")


if __name__ == "__main__":
    unittest.main()
