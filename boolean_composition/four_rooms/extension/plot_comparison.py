import deepdish as dd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import pandas as pd

from four_rooms.config import Config_4, Config_8, Config_16

def plot_comparison(
    data_original_composition,
    data_new_composition,
    tasks,
    num_rooms,
    save_name=None,
):
    tasks = ["\n".join(str(goal) for goal in task) for task in tasks]
    plt.ylim(-0.5, 2)
    rc_ = {
        "figure.figsize": (30, 10),
        "axes.labelsize": 30,
        "font.size": 30,
        "legend.fontsize": 20,
        "axes.titlesize": 30,
    }
    sns.set(rc=rc_, style="darkgrid", font_scale=1.8)
    rc("text", usetex=False)

    types = [
        "Original Composition",
        "New Composition",
    ]

    # Create DataFrame with all runs for each composition type
    # Each row represents one run with values for all tasks, plus the composition type
    data = pd.DataFrame(
        [[data_original_composition[i, n] for n in range(len(tasks))] + [types[0]] 
         for i in range(len(data_original_composition))] +
        [[data_new_composition[i, n] for n in range(len(tasks))] + [types[1]] 
         for i in range(len(data_new_composition))],
        columns=tasks + ["Domain"],
    )
    data = pd.melt(data, "Domain", var_name="Tasks", value_name="Average Returns")

    fig, ax = plt.subplots()
    ax = sns.boxplot(
        x="Tasks",
        y="Average Returns",
        hue="Domain",
        data=data,
        linewidth=3,
        showfliers=False,
    )
    if save_name is None:
        save_name = f"four_rooms/extension/figures/exp_comparison_output_{num_rooms}.png"
    fig.savefig(save_name, bbox_inches="tight")

if __name__ == "__main__":
    NUM_ROOMS = 4
    if NUM_ROOMS == 4:
        Config = Config_4
    elif NUM_ROOMS == 8:
        Config = Config_8
    elif NUM_ROOMS == 16:
        Config = Config_16
    else:
        raise ValueError("Invalid number of rooms")
    Tasks, T_states, Goals = Config["Tasks"], Config["T_states"], Config["Goals"]

    data_original_composition = dd.io.load("exps_data/exp3_returns_0.h5")
    data_new_composition = dd.io.load("exps_data_extension/exp2_all_returns_4.h5")
    data_new_composition = data_new_composition[0]

    save_name = f"four_rooms/extension/figures/exp_comparison_output_{NUM_ROOMS}.png"
    plot_comparison(data_original_composition, data_new_composition, Tasks, NUM_ROOMS, save_name)