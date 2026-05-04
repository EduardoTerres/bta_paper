import random
import numpy as np
import deepdish as dd
from four_rooms.GridWorld import GridWorld
from tqdm import tqdm
from four_rooms.library import (
    Goal_Oriented_Q_learning,
)
from four_rooms.config import (
    Config_4,
    Config_8,
    Config_16,
)
from four_rooms.extension.utils import (
    proportional_sample,
    get_composed_tasks,
    evaluate,
)

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------
np.object = object  # Hack to avoid error in save

random.seed(42)

NUM_ROOMS = 4
configs = {
    4: Config_4,
    8: Config_8,
    16: Config_16,
}
config = configs[NUM_ROOMS]

terminal_states = config["T_states"]
goals = config["Goals"]
tasks = config["Tasks"]
base_tasks = config["Bases"]
composition_rules = config["Composition_rules"]

tasks = proportional_sample(tasks, 5)

# Remove universal and empty tasks
other_tasks = [task for task in tasks if not (len(task) == len(goals) or len(task) == 0)]

# (Sparse rewards, Same terminal states)
# types = [(True, True), (True, False), (False, True), (False, False)]
types = [(True, True)]

num_runs = 1_000

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
returns_per_maxiter = {}
for maxiter in [10, 25, 50, 75, 100, 250, 500, 1_000, 2_000, 5_000]:
    pbar = tqdm(base_tasks, desc="Training tasks", total=len(base_tasks) + 2)
    pbar.set_description(f"Training tasks (maxiter: {maxiter})")

    # Universal task
    env = GridWorld(
        MAP="MAP_" + str(NUM_ROOMS),
        goals=terminal_states,
        dense_rewards=not types[0][0],
    )
    learned_universal_EQ, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)
    # render_EQ(learned_universal_EQ, env, f"four_rooms/extension/figures/learned_universal_task_rooms_{NUM_ROOMS}.png")
    pbar.update(1)

    # Empty task
    env = GridWorld(
        MAP="MAP_" + str(NUM_ROOMS),
        goals=terminal_states,
        goal_reward=-0.1,
        dense_rewards=not types[0][0],
    )
    learned_empty_EQ, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)
    # render_EQ(learned_empty_EQ, env, f"four_rooms/extension/figures/learned_empty_task_rooms_{NUM_ROOMS}.png")
    pbar.update(1)

    # Base tasks
    learned_base_tasks_EQs = []
    for i, task in enumerate(base_tasks):
        task_goals = [[pos, pos] for pos in task]
        env = GridWorld(
            MAP="MAP_" + str(NUM_ROOMS),
            goals=task_goals,
            dense_rewards=not types[0][0],
            T_states=terminal_states if types[0][1] else task_goals,
        )
        learned_EQ, _ = Goal_Oriented_Q_learning(
            env, maxiter=maxiter, T_states=None if types[0][1] else terminal_states
        )
        learned_base_tasks_EQs.append(learned_EQ)
        # render_EQ(learned_EQ, env, f"four_rooms/extension/figures/learned_base_task_{i}_rooms_{NUM_ROOMS}.png")
        pbar.update(1)


    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    # With universal and empty tasks
    EQs_composed, time_taken = get_composed_tasks(
        tasks=other_tasks,
        goals=goals,
        EQ_on=learned_universal_EQ,
        EQ_off=learned_empty_EQ,
        EQ_basis=learned_base_tasks_EQs,
        composition_rules=composition_rules,
    )

    returns = {}
    for task, EQs in tqdm(EQs_composed.items(), desc="Evaluating tasks"):
        task_goals = [[pos, pos] for pos in task]
        returns[task] = {
            "onoff": [],
            "boolean": [],
        }
        for _ in range(num_runs):
            returns[task]["onoff"].append(
                evaluate(task_goals, EQs["onoff"], terminal_states, NUM_ROOMS)
            )
            returns[task]["boolean"].append(
                evaluate(task_goals, EQs["boolean"], terminal_states, NUM_ROOMS)
            )

    # Save returns
    returns = {task: {method: returns_list for method, returns_list in returns[task].items()} for task in returns}
    returns_per_maxiter[maxiter] = returns

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
dd.io.save(f"exps_data_extension/convergence_returns_{NUM_ROOMS}.h5", returns_per_maxiter)
