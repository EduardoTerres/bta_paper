Code for:
"A Goal-Set Characterization of Task Composition in the Boolean Task Algebra"

# Rooms environment
The packages in this environment are managed through `uv`.

Training tasks:
```
uv run four_rooms/extension/exp_train.py
```
specifying the number of rooms you want to run the experiment for
(e.g. `NUM_ROOMS = 8`).

Training tasks by increasing level of optimality:
```
uv run four_rooms/extension/exp_convergence.py
```

Plots (all):
```
uv run four_rooms/extension/plots.py
```

# Boxman environment
Train the primitives:
```
uv run boxman_sts/extension/scripts/run_exp_convergence.sh
```

Evaluate the composed policies:
```
uv run boxman_sts/extension/scripts/eval_exp_conv.sh
```

Plot the results:
```
uv run boxman_sts/extension/scripts/make_plots.sh
```


# Office Gridworld for Skill Machines
Comment that here you need to install the environment
...

Train the Office tasks:
```
bash skill_machines/extension/office/scripts/train.sh
```

Evaluate the trained policies:
```
bash skill_machines/extension/office/scripts/eval.sh
```

Plot the Office results:
```
bash skill_machines/extension/office/scripts/plots.sh
```



# Safety Gym for Skill Machines
Comment very shortly on the mujoco setup.

Train the Safety-Gym tasks:
```
bash skill_machines/extension/safety_gym/scripts/train.sh
```

Evaluate the trained policies:
```
bash skill_machines/extension/safety_gym/scripts/eval.sh
```

Plot the Safety-Gym results:
```
bash skill_machines/extension/safety_gym/scripts/plots.sh
```
