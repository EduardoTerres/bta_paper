import numpy as np
from GridWorld import GridWorld
from library import Q_learning, Goal_Oriented_Q_learning
import deepdish as dd


def load_value_function(file_path: str, extended=False):
    """Load deepdish file into numpy array"""
    import h5py

    def extract_index(key):
        """Extract integer index from key string like 'i0' -> 0"""
        return int(key[1:])

    with h5py.File(file_path, "r") as f:
        data_group = f['data']

        Qs = [
            {
                extract_index(state): np.array(data_group[task][state]["i1"])  # (num_actions,)
                for state in sorted(data_group[task].keys(), key=extract_index)
            }
            for task in sorted(data_group.keys(), key=extract_index)
        ]
        if extended:
            Qs = [
                {
                    extract_index(state): {
                        extract_index(goal): np.array(data_group[task][state]["i1"][goal]["i1"][:])  # (num_actions,)
                        for goal in sorted(data_group[task][state].keys(), key=extract_index)
                    }
                    for state in sorted(data_group[task].keys(), key=extract_index)
                }
                for task in sorted(data_group.keys(), key=extract_index)
            ]

        return Qs

def count_states(Qs):
    """Count the number of states in the Qs.
    
    States are represented as strings like '[None, (3,3)]' or '[(3,3), (3,3)]' or '[None, None]'.
    Intermediate states have one None and one non-None (e.g. (3,3)).
    None state is '[None, None]'.
    Goal states dont have a None.
    """
    count = len(Qs[0].keys())
    intermediate = len([key for key in Qs[0].keys() if key[1:5] == 'None'])
    terminal = len([key for key in Qs[0].keys() if key[1:5] != 'None'])
    none_state = [key for key in Qs[0].keys() if key == '[None, None]']
    print(
        f"There are {count} states"
        f" = {intermediate} (intermediate)"
        f" + {terminal} (terminal)"
        f" + {len(none_state)} (none state)"
    )

env = GridWorld()
maxiter = 300
T_states = [(3, 3), (3, 9), (9, 3), (9, 9)]
T_states = [[pos, pos] for pos in T_states]
Tasks = [
    [],
    [(3, 3), (3, 9), (9, 3), (9, 9)],
    [(3, 3)],
    [(3, 9)],
    [(9, 3)],
    [(9, 9)],
    [(3, 3), (3, 9)],
    [(3, 9), (9, 3)],
    [(9, 3), (9, 9)],
    [(3, 3), (9, 3)],
    [(3, 3), (3, 9), (9, 3)],
    [(3, 3), (3, 9), (9, 9)],
    [(3, 3), (9, 3), (9, 9)],
    [(3, 9), (9, 3), (9, 9)],
    [(3, 3), (9, 9)],
    [(3, 9), (9, 3)],
]

Qs = dd.io.load("exps_data/4Goals_Optimal_Qs.h5")
Qs = [{s: v for (s, v) in Q} for Q in Qs]
count_states(Qs)

EQs = dd.io.load("exps_data/4Goals_Optimal_EQs.h5")
EQs = [{s: {s__: v__ for (s__, v__) in v} for (s, v) in EQ} for EQ in EQs]
count_states(EQs)

num_runs = 1
dataQ = np.zeros((num_runs, len(Tasks)))
dataEQ = np.zeros((num_runs, len(Tasks)))
idxs = np.arange(len(Tasks))
for i in range(num_runs):
    print("run: ", i)
    np.random.shuffle(idxs)
    for j in idxs:
        print("Task: ", j)
        goals = [[pos, pos] for pos in Tasks[j]]
        env = GridWorld(
            goals=goals, goal_reward=1, step_reward=-0.01, T_states=T_states
        )
        _, stats = Q_learning(env, Q_optimal=Qs[j])
        dataQ[i, j] = stats["T"]
        _, stats = Goal_Oriented_Q_learning(env, Q_optimal=EQs[j])
        dataEQ[i, j] = stats["T"]

print(EQs)

np.object = object # Hack to avoid error in save
data1 = dd.io.save("exps_data/exp1_samples_Qs.h5", dataQ)
data2 = dd.io.save("exps_data/exp1_samples_EQs.h5", dataEQ)
