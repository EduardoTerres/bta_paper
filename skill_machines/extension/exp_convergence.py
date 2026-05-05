"""
Convergence experiment: standard Skill Machine composition vs goal-set BTA composition.

Trains WVFs (Q_U and Q_∅) for increasing numbers of steps on Office-v0 and evaluates
zero-shot composition on four Office World tasks (Coffee, Patrol, CoffeeMail, Long).
Produces per-method convergence figures with optimal baselines.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle, numpy as np, gymnasium as gym
import matplotlib.pyplot as plt
import envs  # registers Office-v0 and task variants
from sm import TaskPrimitive, SkillMachine, BaseAgent, evaluate
from rm import Task
import sm_ql
from sympy import sympify
from sympy.logic import boolalg


# ─── Goal-Set BTA components ─────────────────────────────────────────────────

def _goal_satisfies_exp(exp, goal, predicates, constraints):
    """True iff goal vector satisfies sympy boolean expression with p_* / c_* symbols."""
    exp = sympify(exp)
    if isinstance(exp, boolalg.BooleanTrue):  return True
    if isinstance(exp, boolalg.BooleanFalse) or not exp: return False
    n = len(predicates)
    subs = {}
    for sym in exp.free_symbols:
        name = str(sym)
        if name.startswith("p_"):   subs[sym] = bool(goal[predicates.index(name[2:])])
        elif name.startswith("c_"): subs[sym] = bool(goal[n + predicates.index(name[2:])])
    return bool(exp.subs(subs))


class GoalSetSkillPrimitive(BaseAgent):
    """BTA goal-set primitive: routes each goal to Q_U or Q_∅ based on expression satisfaction."""

    def __init__(self, SP, goals, predicates, constraints, exp):
        self.SP, self.goals = SP, goals
        self.predicates, self.constraints, self.exp = predicates, constraints, exp
        self.goal = None
        self.is_discrete = type(SP["1"].action_space) == gym.spaces.Discrete

    def get_action_value(self, states, desired_goal=None, vectorised=False):
        states = states.copy()
        if desired_goal is not None:
            states["desired_goal"] = np.expand_dims(desired_goal, 0)
            key = "1" if _goal_satisfies_exp(self.exp, desired_goal, self.predicates, self.constraints) else "0"
            return self.SP[key].get_action_value(states)
        goals_arr = np.array(list(self.goals.values()))
        actions, values = [], []
        for g in goals_arr:
            states["desired_goal"] = np.expand_dims(g, 0)
            key = "1" if _goal_satisfies_exp(self.exp, g, self.predicates, self.constraints) else "0"
            a, v = self.SP[key].get_action_value(states)
            actions.append(a[0]); values.append(v[0])
        idx = np.argmax(values)
        self.goal = goals_arr[idx]
        return [actions[idx]], [values[idx]]


class GoalSetSkillMachine(SkillMachine):
    """Skill Machine using goal-set BTA composition instead of boolean WVF algebra."""

    def __init__(self, primitive_env, SP, **kwargs):
        super().__init__(primitive_env, SP, **kwargs)
        self.SP = SP  # raw QLAgents {"0": Q_∅, "1": Q_U}
        self.predicates = primitive_env.predicates

    def step(self, rm, true_propositions):
        if self.rm_state != rm.u:
            self.violated_constraints = self.proposition_space.low.copy()
            self.true_propositions = true_propositions.copy()
            self.rm_state, self.exp = rm.u, self.delta_q[rm.u]
            if self.exp not in self.exp_wvf_cache:
                goals = self.skill_primitives["1"].goals  # shared reference to primitive_env.goals
                self.exp_wvf_cache[self.exp] = GoalSetSkillPrimitive(
                    self.SP, goals, self.predicates, self.constraints, self.exp)
            self.wvf = self.exp_wvf_cache[self.exp]
            self.goal = None
        else:
            self.violated_constraints |= ((self.true_propositions ^ true_propositions) & self.constraints_mask)
            self.true_propositions = true_propositions.copy()
            if self.goal_directed: self.goal = self.wvf.goal
        return self.wvf


# ─── Experiment configuration ─────────────────────────────────────────────────

ENV_NAME = "Office-v0"
TASKS = {
    "Coffee":
        "(F (c & X (F o))) & (G~d)",
    "Patrol":
        "(F (A & X (F (B & X (F (C & X (F D))))))) & (G~d)",
    "CoffeeMail":
        "((F (m & X (F (c & X (F o))))) | (F (c & X (F (m & X (F o)))))) & (G~d)",
    "Long":
        "(F (m & X (F (o & X (~ m U (~ tm & m & X (F (c & X (~ o U (~ to & o & X "
        "(F (A & X (F (B & X (F (C & X (F (D & X (F A))))))))))))))))))) & (G~d)",
}
MAX_ITERS     = [50_000, 100_000, 200_000]
OPTIMAL_ITERS = 300_000
N_SEEDS       = 1
EVAL_EPISODES = 50
OUT_DIR = os.path.join(os.path.dirname(__file__), "exps_data_extension")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _train(total_steps, seed):
    """Fresh primitive training run; returns (primitive_env, SP)."""
    primitive_env = TaskPrimitive(gym.make(ENV_NAME))
    log_dir = f"/tmp/sm_conv_logs/seed{seed}_iter{total_steps}/"
    SP = sm_ql.learn(primitive_env, None, total_steps,
                     log_dir=log_dir, print_freq=10**9, seed=seed)
    return primitive_env, SP


def _eval(primitive_env, SP, ltl, SM_cls, seed=None):
    """Evaluate one SM variant on a task; returns success rate in [0, 1]."""
    task_env = Task(gym.make(ENV_NAME), ltl)
    SM = SM_cls(primitive_env, SP, goal_directed=True)
    _, success_rate, _ = evaluate(task_env, SM, epsilon=0, gamma=1,
                                  episodes=EVAL_EPISODES, seed=seed)
    return success_rate


# ─── Main experiment ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    all_iters = MAX_ITERS + [OPTIMAL_ITERS]
    sm_data    = {t: {it: [] for it in all_iters} for t in TASKS}
    gsbta_data = {t: {it: [] for it in all_iters} for t in TASKS}

    for seed in range(N_SEEDS):
        print(f"\n=== Seed {seed + 1}/{N_SEEDS} ===")
        for iters in all_iters:
            print(f"  Training {iters:,} steps...")
            primitive_env, SP = _train(iters, seed)
            for task_name, ltl in TASKS.items():
                r_sm    = _eval(primitive_env, SP, ltl, SkillMachine,        seed)
                r_gsbta = _eval(primitive_env, SP, ltl, GoalSetSkillMachine, seed)
                sm_data[task_name][iters].append(r_sm)
                gsbta_data[task_name][iters].append(r_gsbta)
                print(f"    {task_name}: SM={r_sm:.3f}  BTA={r_gsbta:.3f}")

    # Save results
    with open(os.path.join(OUT_DIR, "sm_convergence.pkl"), "wb") as f:
        pickle.dump({"sm": sm_data, "gsbta": gsbta_data}, f)
    print("Results saved to", OUT_DIR)

    # ─── Plotting ─────────────────────────────────────────────────────────────
    COLORS = {"SM": "steelblue", "Goal-Set BTA": "tomato"}

    def _plot_curve(ax, task, data, method, color):
        vals  = [data[task][it] for it in MAX_ITERS]
        means = [np.mean(v) for v in vals]
        stds  = [np.std(v)  for v in vals]
        opt   = data[task][OPTIMAL_ITERS]
        ax.plot(MAX_ITERS, means, color=color, label=method)
        ax.fill_between(MAX_ITERS, np.subtract(means, stds), np.add(means, stds),
                        color=color, alpha=0.2)
        for fn, ls in [(np.mean, "--"), (np.min, ":"), (np.max, ":")]:
            ax.axhline(fn(opt), color=color, linestyle=ls, linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Training steps"); ax.set_ylabel("Success rate")
        ax.set_title(task)

    three_tasks = ["Coffee", "Patrol", "CoffeeMail"]

    # Two combined 3-subplot figures (one per method)
    for method, data, color in [("SM", sm_data, COLORS["SM"]),
                                 ("Goal-Set BTA", gsbta_data, COLORS["Goal-Set BTA"])]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, task in zip(axes, three_tasks):
            _plot_curve(ax, task, data, method, color)
        fig.suptitle(f"Convergence — {method}")
        plt.tight_layout()
        fname = "fig_sm_3tasks.pdf" if method == "SM" else "fig_gsbta_3tasks.pdf"
        fig.savefig(os.path.join(OUT_DIR, fname)); plt.close(fig)

    # Individual task figures with both methods overlaid
    for task in TASKS:
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_curve(ax, task, sm_data,    "SM",            COLORS["SM"])
        _plot_curve(ax, task, gsbta_data, "Goal-Set BTA", COLORS["Goal-Set BTA"])
        ax.legend(); plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"fig_{task}.pdf")); plt.close(fig)

    print("Figures saved to", OUT_DIR)
