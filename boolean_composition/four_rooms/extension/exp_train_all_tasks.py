import random
import numpy as np
import deepdish as dd
from pathlib import Path
from collections import defaultdict

# Import path
import sys
sys.path.append(Path(__file__).parent.parent)

from four_rooms.GridWorld import GridWorld
from tqdm import tqdm
from four_rooms.library import Goal_Oriented_Q_learning
from four_rooms.config import (
    Config_4,
    Config_8,
    Config_16,
)

np.object = object  # Hack to avoid error in save

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def convert_defaultdict_to_dict(obj):
    """Convert defaultdict objects to regular dictionaries to avoid pickling issues."""
    if isinstance(obj, defaultdict):
        return {key: convert_defaultdict_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, dict):
        return {key: convert_defaultdict_to_dict(value) for key, value in obj.items()}
    else:
        return obj


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
NUM_ROOMS = 16
if NUM_ROOMS == 4:
    Config = Config_4
elif NUM_ROOMS == 8:
    Config = Config_8
elif NUM_ROOMS == 16:
    Config = Config_16
else:
    raise ValueError("Invalid number of rooms")

Tasks, T_states, Goals = Config["Tasks"], Config["T_states"], Config["Goals"]

# Randomly select 3 goals
num_goals = 3
num_goals = min(num_goals, len(Goals))
random_goals = random.sample(Goals, num_goals)

# Randomly select tasks that contain any of the random goals
num_tasks = 50
random_tasks = [task for task in Tasks if any(goal in task for goal in random_goals)]
num_qs = min(num_tasks, len(random_tasks))
random_tasks = random.sample(random_tasks, num_qs)
print("Selected random tasks: ", random_tasks)

Tasks = random_tasks

# (Sparse rewards, Same terminal states)
types = [(True, True), (True, False), (False, True), (False, False)]

maxiter = 20000

print("type: (Sparse rewards, Same terminal states)")
t = 0

EQs_learned = []
# Learning base tasks and doing composed tasks
for task in tqdm(Tasks, desc="Training tasks"):
    goals = [[pos, pos] for pos in task]
    env = GridWorld(
        MAP="MAP_" + str(NUM_ROOMS),
        goals=goals,
        dense_rewards=not types[t][0],
        T_states=T_states if types[t][1] else goals,
    )
    Q, _ = Goal_Oriented_Q_learning(
        env, maxiter=maxiter, T_states=None if types[t][1] else T_states
    )
    EQs_learned.append(Q)

# Convert all Q objects to regular dictionaries
EQs_learned_converted = [convert_defaultdict_to_dict(eq) for eq in EQs_learned]

task_to_EQs = {tuple(task): eq for task, eq in zip(Tasks, EQs_learned_converted)}
# Name convention:
# exp1_<number_of_rooms>_<number_of_goals>_<number_of_tasks_learned>_<type_of_environment>_<maxiter>.h5
save_name = (
    f"exps_data_extension/exp1_{str(NUM_ROOMS)}_{str(num_goals)}_{str(num_tasks)}_{str(t)}_{str(maxiter)}.h5"
)
dd.io.save(save_name, task_to_EQs)
