"""Render decoder inputs and reward predictions for goal-slice GCB checkpoints."""
import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from environments.four_rooms import FourRoomsGoalEnv
from goal_slice_gcb_four_rooms import (
    GoalSliceGCB,
    extended_reward,
    goal_set_split,
    sample_set,
    transition,
)


LOG = logging.getLogger("goal_slice_decoder_samples")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, action="append",
                        help="Checkpoint to compare; supply once per run")
    parser.add_argument("--label", action="append",
                        help="Optional display label, in the same order as --checkpoint")
    parser.add_argument("--output", type=Path,
                        default=Path("figures/goal_slice_gcb_decoder_samples.png"))
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of transition/goal-set combinations (figure rows)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-set-split", choices=("auto", "train", "held-out", "random"), default="auto",
                        help="auto uses held-out sets when the checkpoint was trained with them")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def choose_goal_set(goals, query, split, trained_held_out, rng, contains_query):
    if split == "auto":
        split = "held-out" if trained_held_out else "random"
    if split == "random":
        return sample_set(goals, query, contains_query, rng), split
    train_sets, held_out_sets = goal_set_split(goals)
    source = train_sets if split == "train" else held_out_sets
    compatible = [goal_set for goal_set in source if (query in goal_set) == contains_query]
    return compatible[rng.randint(len(compatible))], split


def load_checkpoint(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: %s" % checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
        raise ValueError("Expected a goal-slice GCB checkpoint with an args dictionary")
    if not isinstance(payload.get("model"), dict):
        raise ValueError("Checkpoint is missing the model state dictionary: %s" % checkpoint_path)
    model_args = SimpleNamespace(**payload["args"])
    if not hasattr(model_args, "num_rooms"):
        raise ValueError("Checkpoint is missing num_rooms: %s" % checkpoint_path)
    env = FourRoomsGoalEnv(num_rooms=model_args.num_rooms,
                           obs_img_dim=getattr(model_args, "obs_img_dim", 64), slip_prob=0.0)
    model = GoalSliceGCB(env.observation_space.shape, model_args, device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model_args, payload.get("step"), env, model


def previous_state(target, env, rng):
    """Choose a legal predecessor of target, preferring a visibly different state."""
    positions = list(env.env.possiblePositions)
    walls, valid_positions = set(env.env.walls), set(positions)
    candidates = [(state, action) for state in positions for action in range(env.action_space.shape[0])
                  if transition(state, action, walls, valid_positions) == target]
    moving_candidates = [(state, action) for state, action in candidates if state != target]
    return (moving_candidates or candidates)[rng.randint(len(moving_candidates or candidates))]


def make_samples(env, model_args, count, split, rng):
    """Build varied, legal transitions with known rewards for the comparison rows."""
    goals = [tuple(goal) for goal in env.goal_positions]
    valid_positions = list(env.env.possiblePositions)
    samples = []
    selected_split = None
    for index in range(count):
        case = index % 4
        query = goals[rng.randint(len(goals))]
        contains_query = case in (0, 2)
        goal_set, selected_split = choose_goal_set(
            goals, query, split, bool(getattr(model_args, "evaluate_on_held_out", False)), rng, contains_query)
        goal_set = tuple(goal_set)
        if case in (0, 1):
            next_position = query  # query hit: universal or empty reward
            case_name = "query hit, query %s" % ("included" if contains_query else "excluded")
        elif case == 2:
            other_members = [position for position in goal_set if position != query]
            if not other_members:
                next_position = query
                case_name = "query hit, only query member"
            else:
                next_position = other_members[rng.randint(len(other_members))]
                case_name = "non-query goal member"
        else:
            non_members = [position for position in valid_positions if position != query and position not in goal_set]
            next_position = non_members[rng.randint(len(non_members))]
            case_name = "non-goal transition"
        state_position, action = previous_state(next_position, env, rng)
        reward = extended_reward(next_position, query, goal_set,
                                 model_args.r_universal, model_args.r_empty, model_args.r_min)
        samples.append(dict(state=state_position, action=action, next_state=next_position,
                            query=query, goal_set=goal_set, reward=reward, description=case_name))
    return samples, selected_split


def image_tensor(env, positions, device):
    image = env.render_goal_set(positions)
    return torch.as_tensor(image, device=device).float().div(255).unsqueeze(0)


@torch.no_grad()
def predicted_reward(model, env, sample, device):
    state = image_tensor(env, [sample["state"]], device)
    next_state = image_tensor(env, [sample["next_state"]], device)
    query = image_tensor(env, [sample["query"]], device)
    goal_set = image_tensor(env, sample["goal_set"], device)
    z = model.encode(state, query, goal_set)
    z_next = model.encode(next_state, query, goal_set)
    return model.decoder(torch.cat((z, z_next), dim=1)).item()


def image_for_plot(env, positions):
    return np.moveaxis(env.render_goal_set(positions), 0, -1)


def run_label(path, model_args, step):
    return "fdim %s\nstep %s" % (getattr(model_args, "feature_dim", "?"),
                                   step if step is not None else "?")


def plot(models, samples, output, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_columns = ("State $I_s$", "Query $I_g$", "Goal set $I_{G+}$", "Next state $I_{s'}$")
    columns = len(input_columns) + len(models)
    figure, axes = plt.subplots(len(samples), columns,
                                figsize=(2.45 * len(input_columns) + 1.65 * len(models) + 1, 2.55 * len(samples) + 1),
                                squeeze=False, constrained_layout=True)
    for row, sample in enumerate(samples):
        inputs = ([sample["state"]], [sample["query"]], sample["goal_set"], [sample["next_state"]])
        for column, positions in enumerate(inputs):
            axis = axes[row, column]
            axis.imshow(image_for_plot(models[0][2], positions), interpolation="nearest", aspect="equal")
            axis.set_axis_off()
            if row == 0:
                axis.set_title(input_columns[column], fontsize=11)
        for offset, (label, model_args, env, model, device) in enumerate(models):
            axis = axes[row, len(input_columns) + offset]
            prediction = predicted_reward(model, env, sample, device)
            axis.set_axis_off()
            axis.set_facecolor("#f7f7f7")
            axis.text(0.5, 0.60, "decoded reward", ha="center", va="center", fontsize=9,
                      transform=axis.transAxes)
            axis.text(0.5, 0.43, "%.3f" % prediction, ha="center", va="center", fontsize=18,
                      fontweight="bold", transform=axis.transAxes)
            axis.text(0.5, 0.22, "target %.2f" % sample["reward"], ha="center", va="center", fontsize=9,
                      transform=axis.transAxes)
            if row == 0:
                axis.set_title(label, fontsize=10)
        axes[row, 0].set_ylabel("%s\naction=%d" % (sample["description"], sample["action"]), fontsize=9)
    figure.suptitle("Goal-slice GCB reward-decoder comparison", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.label and len(args.label) != len(args.checkpoint):
        raise ValueError("Supply exactly one --label for every --checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    loaded = [load_checkpoint(path, device) for path in args.checkpoint]
    reference_args, _, reference_env, _ = loaded[0]
    for checkpoint, (model_args, _, env, _) in zip(args.checkpoint[1:], loaded[1:]):
        if (model_args.num_rooms, getattr(model_args, "obs_img_dim", 64)) != (
                reference_args.num_rooms, getattr(reference_args, "obs_img_dim", 64)):
            raise ValueError("All checkpoints must use the same environment shape: %s" % checkpoint)
    rng = np.random.RandomState(args.seed)
    samples, selected_split = make_samples(reference_env, reference_args, args.num_samples,
                                           args.goal_set_split, rng)
    labels = args.label or [run_label(path, model_args, step)
                             for path, (model_args, step, _, _) in zip(args.checkpoint, loaded)]
    models = [(label, model_args, env, model, device)
              for label, (model_args, _, env, model) in zip(labels, loaded)]
    print("goal_set_split=%s" % selected_split)
    for index, sample in enumerate(samples, start=1):
        print("row %d: %s; state=%s action=%d next_state=%s query=%s goal_set=%s target_reward=%.3f" %
              (index, sample["description"], sample["state"], sample["action"], sample["next_state"],
               sample["query"], list(sample["goal_set"]), sample["reward"]))
    plot(models, samples, args.output, args.dpi)
    LOG.info("Saved %d input/output combinations for %d checkpoint(s) to %s",
             len(samples), len(models), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
