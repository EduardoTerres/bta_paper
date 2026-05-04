from itertools import chain, combinations
import numpy as np

# ------------------------------------
# Types
# ------------------------------------
GoalType = tuple[int, int]
TerminalStatesType = list[list[GoalType]]
BasisType = list[list[GoalType]]
TasksType = list[list[GoalType]]

# ------------------------------------
# Utils
# ------------------------------------
# Generate all non-empty combinations of T_states as tasks
def all_combinations(iterable):
    "all non-empty subsets of iterable"
    s = list(iterable)
    s = list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))
    s = [list(task) for task in s]
    s.insert(0, [])

    # Order by length and then lexicographically
    s.sort(key=lambda x: (len(x), x))
    return s

def get_base_tasks(goals: list[GoalType]) -> tuple[BasisType, dict[GoalType, list[list[int]]]]:
    """
    Given a list of goals, this function generates a set of 'base tasks' such that
    each base task is a subset of goals corresponding to a column in the binary encoding
    of the goal indices. This is useful for constructing a basis for goal composition,
    where each goal can be represented as a combination of base tasks.

    Arguments:
        goals: list of goals

    Returns:
        base_tasks: list of base tasks
        composition_rules: dictionary of composition rules for each goal
    """
    num_rows, num_cols = len(goals), int(np.ceil(np.log2(len(goals))))
    matrix = [
        [int(b) for b in format(i, f'0{int(num_cols)}b')] for i in range(num_rows)
    ]
    matrix = np.array(matrix)

    base_tasks = []
    for k in range(num_cols):
        base_tasks.append([])
        for i, goal in enumerate(goals):
            if matrix[i, k] == 1:
                base_tasks[k].append(goal)

    composition_rules = dict(zip(goals, list(matrix)))

    return base_tasks, composition_rules

# ------------------------------------
# 4 rooms configuration
# ------------------------------------
Goals_4 = [
    (3, 3), (9, 3),
    (3, 9), (9, 9),
]
T_states_4 = [[pos, pos] for pos in Goals_4]

Tasks_4 = all_combinations(Goals_4)

Bases_4, Composition_rules_4 = get_base_tasks(Goals_4)

Config_4 = {
    "Goals": Goals_4,
    "T_states": T_states_4,
    "Tasks": Tasks_4,
    "Bases": Bases_4,
    "Composition_rules": Composition_rules_4,
}

# ------------------------------------
# 8 rooms configuration
# ------------------------------------
Goals_8 = [
    (3, 3),     (9, 3),     (15, 3),
    (3, 9),                 (15, 9),
    (3, 15),    (9, 15),    (15, 15),
]
T_states_8 = [[pos, pos] for pos in Goals_8]

Tasks_8 = all_combinations(Goals_8)

Bases_8, Composition_rules_8 = get_base_tasks(Goals_8)

Config_8 = {
    "Goals": Goals_8,
    "T_states": T_states_8,
    "Tasks": Tasks_8,
    "Bases": Bases_8,
    "Composition_rules": Composition_rules_8,
}

# ------------------------------------
# 16 rooms configuration
# ------------------------------------
Goals_16 = [
    (3, 3),     (9, 3),     (15, 3),    (21, 3),
    (3, 9),     (9, 9),     (15, 9),    (21, 9),
    (3, 15),    (9, 15),    (15, 15),   (21, 15),
    (3, 21),    (9, 21),    (15, 21),   (21, 21),
]
T_states_16 = [[pos, pos] for pos in Goals_16]

Tasks_16 = all_combinations(Goals_16)

Bases_16, Composition_rules_16 = get_base_tasks(Goals_16)

Config_16 = {
    "Goals": Goals_16,
    "T_states": T_states_16,
    "Tasks": Tasks_16,
    "Bases": Bases_16,
    "Composition_rules": Composition_rules_16,
}

# ------------------------------------
# MAPS
# ------------------------------------
MAP_4 = (
    "1 1 1 1 1 1 1 1 1 1 1 1 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 1 0 1 1 1 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 1 1 1 0 1 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 1 1 1 1 1 1 1 1 1 1 1 1"
)

MAP_8 = (
    "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 1 0 1 1 1 1 0 0 0 0 0 1 1 1 1 0 1 1\n"
    "1 0 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 1 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
)

MAP_16 = (
    "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 1 0 1 1 1 1 0 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 0 1 1 0 1 1 1 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 0 1 1 0 1 1 1 1\n"
    "1 1 0 1 1 1 1 0 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n"
    "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
)