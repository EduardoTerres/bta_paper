from collections import defaultdict
import random
import numpy as np
import deepdish as dd
from four_rooms.GridWorld import GridWorld
from tqdm import tqdm
from four_rooms.library import (
    EQ_P,
    EQ_V,
    Goal_Oriented_Q_learning,
)
from four_rooms.config import (
    Config_4,
    Config_8,
    Config_16,
)
import matplotlib.pyplot as plt

from plots import plot

np.object = object  # Hack to avoid error in save

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def get_random_partition(Goals):
    """Returns the set partitioned in two but with all the elements."""
    random.shuffle(Goals)
    partition_point = random.randint(1, len(Goals) - 1)
    return Goals[:partition_point], Goals[partition_point:]


def evaluate(goals, EQ, T_states):
    env = GridWorld(MAP="MAP_" + str(NUM_ROOMS), goals=goals, T_states=T_states)
    # Render
    # render_EQ(EQ, env, f"exp2_composed_type_rooms_{NUM_ROOMS}.png")
    policy = EQ_P(EQ)
    state = env.reset()
    done = False
    t = 0
    G = 0
    while not done and t < 100:
        action = policy[state]
        state_, reward, done, _ = env.step(action)
        state = state_
        G += reward
        t += 1
    return G


def deep_copy(EQ):
    if isinstance(EQ, list):
        return [deep_copy(eq) for eq in EQ]

    elif isinstance(EQ, dict):
        # Create a completely new defaultdict with no reference to the original
        new_dict = defaultdict(dict)
        for state, qvals in EQ.items():
            new_dict[state] = {}
            for goal, qval in qvals.items():
                new_dict[state][goal] = qval
        return new_dict
    else:
        raise ValueError(f"Invalid EQ type, accepted list or dict but got {type(EQ)}.")


def order_EQs(EQs, Partition):
    """
    Reorders EQs so that one contains Q-values for all undesired goals (Partition[0]) and the other for all desired goals (Partition[1]), both returned at once.

    Args:
        EQs (list): List of two EQ dictionaries. Each dictionary maps states to goal-specific Q-values.
        Partition (tuple): Tuple of two lists of goals. Partition[0] contains one subset of goals, Partition[1] the other.

    Returns:
        Both EQ functions at once: the first for undesired goals, the second for desired goals.
    """
    EQ_desired = defaultdict(dict)
    EQ_undesired = defaultdict(dict)
    
    for goal in Goals:
        goal_partition = int(goal in Partition[1])  # 0 if goal in partition 0, 1 if goal in partition 1
        other_partition = not goal_partition
        for state in EQs[0].keys():
            EQ_undesired[state][str([goal, goal])] = EQs[other_partition][state][str([goal, goal])]
            EQ_desired[state][str([goal, goal])] = EQs[goal_partition][state][str([goal, goal])]

    return EQ_undesired, EQ_desired


# Convert defaultdict objects to regular dictionaries to avoid pickling issues
def convert_defaultdict_to_dict(obj):
    if isinstance(obj, defaultdict):
        return {key: convert_defaultdict_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, dict):
        return {key: convert_defaultdict_to_dict(value) for key, value in obj.items()}
    else:
        return obj


def build_EQ(task, Goals, EQ_on, EQ_off):
    """Compose an EQ for a specific task by combining Q-values from EQ_on (for task goals) and EQ_off (for non-task goals)."""
    EQ = defaultdict(dict)
    undesired_goals = set(Goals) - set(task)
    desired_goals = set(task)
    for state in EQ_on.keys():
        for goal in desired_goals:
            EQ[state][str([goal, goal])] = EQ_on[state][str([goal, goal])]
        for goal in undesired_goals:
            EQ[state][str([goal, goal])] = EQ_off[state][str([goal, goal])]
    return EQ


def get_composed_tasks(Tasks, Goals, EQ_on, EQ_off):
    """Generate composed EQs for all tasks by combining Q-values from EQ_on (desired goals) and EQ_off (undesired goals).
    
    Returns the list of tasks with a one to one correspondence to the composed list.
    """
    return [build_EQ(task, Goals, EQ_on, EQ_off) for task in Tasks]


def render_EQ(EQ, env, filename=None):
    fig = env.render(P=EQ_P(EQ), V=EQ_V(EQ))
    save_directory = "four_rooms/extension/figures"
    if filename:
        fig.savefig(f"{save_directory}/{filename}", bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_directory}/{filename}")
    plt.close(fig)

def proportional_sample(tasks, total_samples):
    """Sample the same number of tasks from each length."""
    lenghts = defaultdict(list)
    samples_per_length = total_samples // len(lenghts)
    if total_samples % len(lenghts) != 0:
        print(
            f"Warning: Number of total samples {total_samples} is not"
            f"divisible by the number of lengths {len(lenghts)}.",
        )
    for task in tasks:
        lenghts[len(task)].append(task)
    sampled_tasks = []
    for length in lenghts:
        sampled_tasks.extend(random.sample(lenghts[length], samples_per_length))
    return sampled_tasks

# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------
random.seed(42)

NUM_ROOMS = 4
if NUM_ROOMS == 4:
    Config = Config_4
elif NUM_ROOMS == 8:
    Config = Config_8
elif NUM_ROOMS == 16:
    Config = Config_16
else:
    raise ValueError("Invalid number of rooms")

T_states, Goals, Tasks = Config["T_states"], Config["Goals"], Config["Tasks"]

# (Sparse rewards, Same terminal states)
# types = [(True, True), (True, False), (False, True), (False, False)]
types = [(True, True)]

maxiter = 5000
num_runs = 1000

EQs_all = {}
Returns_all = {}

for t in range(len(types)):
    print("type: ", t)

    # Learning universal bounds (min and max tasks)
    env = GridWorld(
        MAP="MAP_" + str(NUM_ROOMS),
        goals=T_states,
        dense_rewards=not types[t][0],
    )
    EQ_universal_task, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)

    env = GridWorld(
        MAP="MAP_" + str(NUM_ROOMS),
        goals=T_states,
        goal_reward=-0.1,
        dense_rewards=not types[t][0],
    )
    EQ_empty_task, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)
    
    # Learning base tasks and doing composed tasks
    # EQs = []  # [EQ_desired, EQ_undesired]
    # for goals_slice in Partition:
    #     goals = [[pos, pos] for pos in goals_slice]
    #     env = GridWorld(
    #         MAP="MAP_" + str(NUM_ROOMS),
    #         goals=goals,
    #         dense_rewards=not types[t][0],
    #         T_states=T_states if types[t][1] else goals,
    #     )
    #     EQ, _ = Goal_Oriented_Q_learning(
    #         env, maxiter=maxiter, T_states=None if types[t][1] else T_states
    #     )
    #     EQs.append(EQ)


    # EQ_off, EQ_on = order_EQs(EQs, Partition)

    # render_EQ(EQ_off, env, f"exp2_undesired_type_{t}_rooms_{NUM_ROOMS}.png")
    # render_EQ(EQ_on, env, f"exp2_desired_type_{t}_rooms_{NUM_ROOMS}.png")

    EQs_composed = get_composed_tasks(Tasks, Goals, EQ_universal_task, EQ_empty_task)

    # Save base tasks on and off
    EQs_all[t] = {
        "all_on": EQ_universal_task,
        "all_off": EQ_empty_task,
    }

    data = np.zeros((num_runs, len(Tasks)))
    for i in tqdm(range(num_runs), desc="Runs"):
        for j in range(len(Tasks)):
            goals = [[pos, pos] for pos in Tasks[j]]
            data[i, j] = evaluate(goals, EQs_composed[j], T_states)

    Returns_all[t] = data

# Convert all Q objects to regular dictionaries
EQs_all_converted = [convert_defaultdict_to_dict(eq) for eq in EQs_all]

dd.io.save(f"exps_data_extension/exp2_all_EQs_{NUM_ROOMS}.h5", EQs_all_converted)
dd.io.save(f"exps_data_extension/exp2_all_returns_{NUM_ROOMS}.h5", Returns_all)

plot(
    num_rooms=NUM_ROOMS,
    tasks=Tasks,
    save_name=f"four_rooms/extension/figures/exp2_output_{NUM_ROOMS}.png",
)