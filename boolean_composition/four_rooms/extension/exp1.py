import deepdish as dd
import numpy as np
from tqdm import tqdm

from four_rooms.config import Bases_4, Goals_4

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def equal_on_shared_goals(
    task_to_EQs,
    tol=1e-9
):
    """Check that the state-action slices of all tasks that contain a shared goal are the same.

    This function compares the state-action slices of all tasks that contain a shared goal among each other.
    Each match signifies a match of between two tasks on one of their shared goals.

    Args:
        EQs: List of EQs learned for the base tasks.
        base_tasks: List of tasks where each task is represented as a list of goals. E.g., [(3, 3), (3, 9)] means
                    the task is to reach the goals (3, 3) and (3, 9).
        tol: Tolerance for checking equality.
    """
    stats = {
        "mismatch_count": 0,
        "match_count": 0,
    }        
    # Compare all tasks containing goal among each other
    for idx_i, (task_i, EQ_i) in enumerate(task_to_EQs.items()):
        for idx_j, (task_j, EQ_j) in enumerate(task_to_EQs.items()):
            if idx_i <= idx_j:
                continue

            # Obtain intersection goals
            intersection_goals = set(task_i).intersection(set(task_j))
            if len(intersection_goals) == 0:
                continue

            states = list(EQ_i.keys())

            for goal in intersection_goals:
                matches = [
                    np.max(np.abs(EQ_i[state][str([goal, goal])] - EQ_j[state][str([goal, goal])])) < tol
                    for state in states
                ]
                stats["mismatch_count" if not all(matches) else "match_count"] += 1
    
    # Print the counts after all states have been checked
    if stats["mismatch_count"] > 0:
        print(f"\n ❌ {stats['mismatch_count']}/{stats['mismatch_count'] + stats['match_count']} goal slices mismatch.")
    if stats["match_count"] > 0:
        print(f"✅ {stats['match_count']}/{stats['mismatch_count'] + stats['match_count']} goal slices match.")


def equal_on_non_shared_goals(
    task_to_EQs,
    all_goals,
    tol=1e-9
):
    """Check that the state-action slices of all tasks that contain a non-shared goal are the same.

    This function compares the state-action slices of all tasks that contain a non-shared goal among each other.
    Each match signifies a match of between two tasks on one of their non-shared goals.

    Args:
        EQs: List of EQs learned for the base tasks.
        base_tasks: List of tasks where each task is represented as a list of goals. E.g., [(3, 3), (3, 9)] means
                    the task is to reach the goals (3, 3) and (3, 9).
        tol: Tolerance for checking equality.
    """
    stats = {
        "mismatch_count": 0,
        "match_count": 0,
    }        
    # Compare all tasks containing goal among each other
    for idx_i, (task_i, EQ_i) in enumerate(task_to_EQs.items()):
        for idx_j, (task_j, EQ_j) in enumerate(task_to_EQs.items()):
            if idx_i <= idx_j:
                continue

            # Obtain intersection goals
            intersection_goals = set(task_i).union(set(task_j))
            non_intersection_goals = all_goals - intersection_goals
            if len(non_intersection_goals) == 0:
                continue

            states = list(EQ_i.keys())

            for goal in non_intersection_goals:
                matches = [
                    np.max(np.abs(EQ_i[state][str([goal, goal])] - EQ_j[state][str([goal, goal])])) < tol
                    for state in states
                ]
                stats["mismatch_count" if not all(matches) else "match_count"] += 1
    
    # Print the counts after all states have been checked
    if stats["mismatch_count"] > 0:
        print(f"\n❌ {stats['mismatch_count']}/{stats['mismatch_count'] + stats['match_count']} goal slices mismatch.")
    if stats["match_count"] > 0:
        print(f"✅ {stats['match_count']}/{stats['mismatch_count'] + stats['match_count']} goal slices match.")


def parse_filename(fname, print_params=False):
    """Parse filename to extract experiment parameters.
    
    Format: exp1_<number_of_rooms>_<number_of_goals>_<number_of_tasks_learned>_<type_of_environment>_<maxiter>.h5
    """
    parts = fname.split('/')[-1].replace('.h5', '').split('_')
    if print_params:
        print(f"Loading: rooms={parts[1]}, goals={parts[2]}, tasks={parts[3]}, env={parts[4]}, maxiter={parts[5]}", end=" ")
    return {
        'number_of_rooms': parts[1],
        'number_of_goals': parts[2],
        'number_of_tasks_learned': parts[3],
        'type_of_environment': parts[4],
        'maxiter': parts[5]
    }

def pretty_print(func, number):
    print("-" * 46 + "[Test " + str(number) + "]" + "-" * 46)
    func()
    print("-" * 100)

# ------------------------------------------------------------
# Test 1 - Base tasks from original experiments of 4 rooms
# ------------------------------------------------------------
def test_1():
    EQs_A = dd.io.load("exps_data/exp3_base_tasks_A_0.h5")
    EQs_B = dd.io.load("exps_data/exp3_base_tasks_B_0.h5")

    print("Test base tasks from original experiments of 4 rooms", end=" ")
    equal_on_shared_goals(task_to_EQs={tuple(Bases_4[0]): EQs_A, tuple(Bases_4[1]): EQs_B})
    equal_on_non_shared_goals(task_to_EQs={tuple(Bases_4[0]): EQs_A, tuple(Bases_4[1]): EQs_B}, all_goals=set(Goals_4))

pretty_print(func=test_1, number=1)

# ------------------------------------------------------------
# Test 2 - Equal on shared goals for randomly sampled tasks trained from scratch
# ------------------------------------------------------------
filenames = {
    4: "exps_data_extension/exp1_4_3_50_0_2000.h5",
    8: "exps_data_extension/exp1_8_3_50_0_2000.h5",
    16: "exps_data_extension/exp1_16_3_50_0_20000.h5",
}
def test_2():
    for num_rooms in [4, 9, 16]:
        # fname = f"exps_data_extension/exp1_{num_rooms}_3_50_0_2000.h5"
        fname = filenames[num_rooms]
        parse_filename(fname, print_params=True)
        task_to_EQs = dd.io.load(fname)
        equal_on_shared_goals(task_to_EQs=task_to_EQs)

pretty_print(func=test_2, number=2)

# Results of execution:
# Test base tasks from original experiments of 4 rooms ✅ 1/1 goal slices match.
# Loading: rooms=4, goals=3, tasks=50, env=0, maxiter=2000 ✅ 105/105 goal slices match.
# Loading: rooms=9, goals=3, tasks=50, env=0, maxiter=2000 ✅ 3303/3303 goal slices match.
# Loading: rooms=16, goals=3, tasks=50, env=0, maxiter=2000 ✅ 4828/4828 goal slices match.

# ------------------------------------------------------------
# Test 3 - Equal on non-shared goals for randomly sampled tasks trained from scratch
# ------------------------------------------------------------
filenames = {
    4: "exps_data_extension/exp1_4_3_50_0_2000.h5",
    8: "exps_data_extension/exp1_8_3_50_0_2000.h5",
    16: "exps_data_extension/exp1_16_3_50_0_20000.h5",
}
def test_3():
    for num_rooms in [4, 9, 16]:
        fname = filenames[num_rooms]
        parse_filename(fname, print_params=True)
        task_to_EQs = dd.io.load(fname)
        equal_on_non_shared_goals(task_to_EQs=task_to_EQs, all_goals=set(Goals_4))

pretty_print(func=test_3, number=3)