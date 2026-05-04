"""Util files."""

from collections import defaultdict
import random
import time
import matplotlib
from four_rooms.GridWorld import GridWorld
from four_rooms.library import (
    EQ_P,
    EQ_V,
    AND,
    OR,
    NOT,
)
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # non-interactive plots

def get_random_partition(goals):
    """Returns the set partitioned in two but with all the elements."""
    random.shuffle(goals)
    partition_point = random.randint(1, len(goals) - 1)
    return goals[:partition_point], goals[partition_point:]


def evaluate(goals, EQ, terminal_states, num_rooms, save_name=None):
    """Evaluates the EQ on the environment."""
    env = GridWorld(MAP="MAP_" + str(num_rooms), goals=goals, T_states=terminal_states)
    # TODO: Add t[0][0]

    # Render
    if save_name:
        render_EQ(EQ, env, save_name)
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


def order_EQs(EQs, partition, goals):
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
    
    for goal in goals:
        goal_partition = int(goal in partition[1])  # 0 if goal in partition 0, 1 if goal in partition 1
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


def build_EQ_from_on_off(task, goals, EQ_on, EQ_off):
    """Compose an EQ for a specific task by combining Q-values from EQ_on (for task goals) and EQ_off (for non-task goals)."""
    EQ = defaultdict(dict)
    undesired_goals = set(goals) - set(task)
    desired_goals = set(task)
    for state in EQ_on.keys():
        for goal in desired_goals:
            EQ[state][str([goal, goal])] = EQ_on[state][str([goal, goal])]
        for goal in undesired_goals:
            EQ[state][str([goal, goal])] = EQ_off[state][str([goal, goal])]
    return EQ.copy()


def build_single_goal_EQs(composition_rule, EQ_basis, EQ_on, EQ_off):
    """
    Compose an EQ for a specific task by combining Q-values
    from EQ_on (for task goals) and EQ_off (for non-task goals).
    """
    # Composition rule is a list of 0s and 1s corresponding to the EQ base tasks
    # Use NOT(EQ) when 0, EQ when 1. And all the EQs.
    EQ_not_applied = [
        EQ_base_task if rule == 1 else NOT(EQ_base_task, EQ_max=EQ_on, EQ_min=EQ_off)
        for EQ_base_task, rule in zip(EQ_basis, composition_rule)
    ]
    composed_EQs = AND(EQ_not_applied[0], EQ_not_applied[1])
    for EQ_base_task in EQ_not_applied[2:]:
        composed_EQs = AND(composed_EQs, EQ_base_task)
    return composed_EQs


def build_EQ_from_boolean_ops(task, composition_rules, EQ_basis, EQ_on, EQ_off):
    """
    Compose an EQ for a specific task by evaluating a boolean expression.

    It first obtains the EQs for the goals independently from the base tasks.
    Then, it composes the single goals EQs into the required task.
    """
    if len(task) == 0:
        return EQ_off

    if len(task) == len(composition_rules):
        return EQ_on

    if len(task) == 1:
        return build_single_goal_EQs(composition_rules[task[0]], EQ_basis, EQ_on, EQ_off)

    goal_EQ_1 = build_single_goal_EQs(composition_rules[task[0]], EQ_basis, EQ_on, EQ_off)
    goal_EQ_2 = build_single_goal_EQs(composition_rules[task[1]], EQ_basis, EQ_on, EQ_off)

    goal_EQ = OR(goal_EQ_1, goal_EQ_2)

    for goal in task[2:]:
        goal_EQ = OR(goal_EQ, build_single_goal_EQs(composition_rules[goal], EQ_basis, EQ_on, EQ_off))

    return goal_EQ


def get_composed_tasks(
    tasks: list[list[tuple[int, int]]],
    goals: list[tuple[int, int]],
    EQ_on: dict,
    EQ_off: dict,
    EQ_basis: list[dict],
    composition_rules: dict[tuple[int, int], list[int]],
) -> tuple[dict[tuple[int, int], dict[str, dict]], dict[str, list[float]]]:
    """Zero-shot composition of tasks using two methods.
        1. From the universal and empty tasks
        2. From boolean operations.
    
    Args:
        tasks: list of tasks
        goals: list of goals
        EQ_on: EQ for the universal task
        EQ_off: EQ for the empty task
        EQ_basis: list of EQs for the base tasks
        composition_rules: list of composition rules for each task

    Returns:
        composed_EQs: dictionary of composed EQs for each task
    """
    composed_EQs = {}
    time_taken = {"onoff": [], "boolean": []}

    for task in tasks:
        start_time = time.time()
        on_off_EQ = build_EQ_from_on_off(task, goals, EQ_on, EQ_off)
        time_taken["onoff"].append(time.time() - start_time)

        start_time = time.time()
        boolean_EQ = build_EQ_from_boolean_ops(task, composition_rules, EQ_basis, EQ_on, EQ_off)
        time_taken["boolean"].append(time.time() - start_time)

        composed_EQs[tuple(task)] = {
            "onoff": on_off_EQ,
            "boolean": boolean_EQ,
        }

    return composed_EQs, time_taken


def render_EQ(EQ, env, filename=None):
    fig = env.render(P=EQ_P(EQ), V=EQ_V(EQ))
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Figure saved to {filename}.", end="\n")


def proportional_sample(tasks: list[list[tuple[int, int]]], total_samples: int):
    """Sample tasks in a stratified way by length of task (number of goals).
    If for a given length, there are less tasks than total_samples, then all
    tasks of that length are sampled, which will be less than total_samples.
    
    Args:
        tasks: list of tasks

    Returns:
        sampled_tasks: list of sampled tasks
    """
    # Separate tasks by length
    tasks_by_length = defaultdict(list)
    for task in tasks:
        tasks_by_length[len(task)].append(task)

    sampled_tasks = []
    for length in tasks_by_length.keys():
        sampled_tasks.extend(
            random.sample(
                tasks_by_length[length],
                min(total_samples, len(tasks_by_length[length])),
            )
        )

    return sampled_tasks