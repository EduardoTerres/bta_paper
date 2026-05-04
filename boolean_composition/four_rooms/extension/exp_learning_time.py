import random
import time
from four_rooms.GridWorld import GridWorld
from tqdm import tqdm
from four_rooms.extension.plot_utils import plot_learning_time
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
)


# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------
random.seed(42)
configs = {
    4: Config_4,
    8: Config_8,
    16: Config_16,
}

maxiter = 5_000
num_runs = 1
types = [(True, True)]

learning_time = {}

# ------------------------------------------------------------
# Main loop over number of rooms
# ------------------------------------------------------------
for num_rooms in tqdm([4, 8, 16], desc="Processing rooms"):
    config = configs[num_rooms]

    terminal_states = config["T_states"]
    goals = config["Goals"]
    tasks = config["Tasks"]
    base_tasks = config["Bases"]
    composition_rules = config["Composition_rules"]

    tasks = proportional_sample(tasks, 1)

    learning_time_on_off = 0
    learning_time_base_tasks = 0

    # ------------------------------------------------------------
    # Training loop for number of goals (rooms)
    # ------------------------------------------------------------
    for run in tqdm(range(num_runs), desc=f"Runs for {num_rooms} rooms", leave=False):
        # Universal task
        env = GridWorld(
            MAP="MAP_" + str(num_rooms),
            goals=terminal_states,
            dense_rewards=not types[0][0],
        )
        aux_learning_time = time.time()
        learned_universal_EQ, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)
        learning_time_on_off += time.time() - aux_learning_time

        # Empty task
        env = GridWorld(
            MAP="MAP_" + str(num_rooms),
            goals=terminal_states,
            goal_reward=-0.1,
            dense_rewards=not types[0][0],
        )
        aux_learning_time = time.time()
        learned_empty_EQ, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)
        learning_time_on_off += time.time() - aux_learning_time

        # Base tasks
        learned_base_tasks_EQs = []
        for task in base_tasks:
            task_goals = [[pos, pos] for pos in task]
            env = GridWorld(
                MAP="MAP_" + str(num_rooms),
                goals=task_goals,
                dense_rewards=not types[0][0],
                T_states=terminal_states if types[0][1] else task_goals,
            )
            aux_learning_time = time.time()
            learned_EQ, _ = Goal_Oriented_Q_learning(
                env, maxiter=maxiter, T_states=None if types[0][1] else terminal_states
            )
            learning_time_base_tasks += time.time() - aux_learning_time
            learned_base_tasks_EQs.append(learned_EQ)

    learning_time_on_off /= num_runs
    learning_time_base_tasks /= num_runs

    learning_time_total = learning_time_on_off + learning_time_base_tasks

    learning_time[num_rooms] = {
        "onoff": learning_time_on_off,
        "boolean": learning_time_total,
    }

# Plot learning time
plot_learning_time(
    learning_time=learning_time,
    save_name=f"four_rooms/extension/figures/learning_time_{maxiter}_{num_runs}.png",
)