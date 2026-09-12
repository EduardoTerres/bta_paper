#!/usr/bin/env python3
"""Offline image IQL conditioned on the learned goal-slice representation.

The trainer reuses GoalSliceReplayBuffer datasets from goal_slice_gcb_four_rooms:
the stored query-hit flag supplies the terminal mask, so no replay recollection is
needed for a compatible .pt file. IQL uses the core GoalPixelIQLAgent update flow
with a psi state encoder and Gaussian actor/twin-Q/value heads conditioned on (psi, phi).
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from environments.four_rooms import FourRoomsGoalEnv
from goalbisim.agents.goalpixeliql import GoalPixelIQLAgent
from goalbisim.representation.goalbisim import GoalBisim
from rlkit.core import logger as rlkit_logger
from goal_slice_gcb_four_rooms import (
    GoalSliceReplayBuffer,
    collect,
    evaluate_collapse,
    goal_set_split,
    log_training_umap,
    model_checkpoint_steps,
    seed_everywhere,
    start_wandb,
    transition,
    umap_snapshot_steps,
)

log = logging.getLogger("goal_slice_gcb_iql")


class GoalSliceFrameworkReplay:
    """Expose goal-slice transitions through the core GCB replay-buffer contract."""

    def __init__(self, replay, batch_size, device, query_hit_oversample, pool_indices):
        self.replay = replay
        self.batch_size = batch_size
        self.device = device
        self.query_hit_oversample = query_hit_oversample
        self.pool_indices = pool_indices

    def sample(self):
        batch = self.replay.sample(
            self.batch_size, self.device, self.query_hit_oversample, self.pool_indices,
        )
        terminal = (batch["next_state"] == batch["query"]).flatten(1).all(dim=1, keepdim=True)
        kwargs = {"goal_set": batch["goal_set"], "rtg": None, "td": None}
        return (batch["state"], batch["action"], batch["reward"], batch["next_state"],
                (~terminal).float(), batch["query"], kwargs)


class MetricCapture:
    """Small logger shim for existing framework metric calls."""

    def __init__(self):
        self.metrics = {}

    def log(self, metrics):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().mean().item()
            if isinstance(value, np.generic):
                value = value.item()
            self.metrics[key] = value

    def consume(self):
        metrics, self.metrics = self.metrics, {}
        return metrics


def select_goal_set_action(agent, state, query, goal_set, device):
    """Use the framework deterministic Gaussian actor; Four Rooms takes argmax(action)."""
    image = lambda x: torch.as_tensor(x, device=device).float().div(255).unsqueeze(0)
    with torch.no_grad():
        action_mean, _, _ = agent.actor(
            image(state), image(query), compute_pi=False, compute_log_pi=False,
            goal_set=image(goal_set),
        )
    return int(action_mean.argmax(dim=1).item())


def held_out_policy_evaluation(agent, args, device):
    """Roll out deterministic policies on unseen goal sets using image-only inputs."""
    env = FourRoomsGoalEnv(num_rooms=args.num_rooms, obs_img_dim=args.obs_img_dim,
                           max_episode_steps=args.max_episode_steps, slip_prob=0.0)
    rng = np.random.RandomState(args.seed + 8011)
    positions = list(env.env.possiblePositions)
    goals = [tuple(goal) for goal in env.goal_positions]
    if args.evaluate_on_held_out:
        _, goal_sets = goal_set_split(goals)
    else:
        goal_sets = [tuple([goal]) for goal in goals]
    walls, valid = set(env.env.walls), set(positions)

    returns, lengths, included_returns, excluded_returns = [], [], [], []
    included_successes, excluded_avoids = [], []
    for member in (False, True):
        for _ in range(args.rl_eval_episodes_per_status):
            query = goals[rng.randint(len(goals))]
            candidates = [goal_set for goal_set in goal_sets if (query in goal_set) == member]
            if not candidates:
                continue
            goal_set = candidates[rng.randint(len(candidates))]
            position = positions[rng.randint(len(positions))]
            query_image = env.render_goal_set([query])
            set_image = env.render_goal_set(goal_set)
            episode_return, hit_query = 0.0, False
            for step in range(args.max_episode_steps):
                state_image = env.render_goal_set([position])
                action = select_goal_set_action(agent, state_image, query_image, set_image, device)
                position = transition(position, action, walls, valid)
                reward = args.r_universal if position == query and member else (
                    args.r_empty if position == query else (
                        args.r_min if position in goal_set else 0.0
                    )
                )
                episode_return += (args.discount ** step) * reward
                if position == query:
                    hit_query = True
                    break
            returns.append(episode_return)
            lengths.append(step + 1)
            if member:
                included_returns.append(episode_return)
                included_successes.append(float(hit_query))
            else:
                excluded_returns.append(episode_return)
                excluded_avoids.append(float(not hit_query))

    return {
        "eval/rl/return_mean": float(np.mean(returns)),
        "eval/rl/return_std": float(np.std(returns)),
        "eval/rl/episode_length_mean": float(np.mean(lengths)),
        "eval/rl/query_included_success_rate": float(np.mean(included_successes)),
        "eval/rl/query_included_return_mean": float(np.mean(included_returns)),
        "eval/rl/query_excluded_avoid_rate": float(np.mean(excluded_avoids)),
        "eval/rl/query_excluded_return_mean": float(np.mean(excluded_returns)),
        "eval/rl/episodes": len(returns),
    }


def write_metrics(metrics_file, wandb_run, metrics):
    metrics_file.write(json.dumps(metrics) + "\n")
    metrics_file.flush()
    if wandb_run:
        wandb_run.log(metrics, step=metrics["step"])


def train(args, dataset_path):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    replay = GoalSliceReplayBuffer.load(dataset_path, args.number_training_points)
    train_indices = np.arange(replay.idx)

    representation_kwargs = dict(
        transition_model_type="next_observation",
        psi_loss_form="direct",
        metric_loss=args.metric_loss,
        metric_distance="reward",
        decoder_type="reward",
        dynamics_loss="delta",
        action_weight=args.action_weight,
        decode_both=True,
        feature_dim=args.feature_dim,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        num_layers_paired=args.num_layers,
        num_filters_paired=args.num_filters,
        lr=args.lr,
        lr_paired=args.lr,
        weight_decay=args.weight_decay,
        weight_decay_paired=args.weight_decay,
        action_shape=replay.action_shape,
        discount=args.discount,
        ground_space=True,
        phi_updates_before_psi=0,
        goal_set_conditioned=True,
        lambda_comp=0.0,
    )
    critic_representation = GoalBisim(replay.image_shape, device, **representation_kwargs)
    actor_representation = GoalBisim(replay.image_shape, device, **representation_kwargs)
    target_critic_representation = GoalBisim(replay.image_shape, device, **representation_kwargs)
    agent = GoalPixelIQLAgent(
        replay.image_shape, replay.action_shape, device, lambda obs, _: obs,
        actor_representation, critic_representation, target_critic_representation, None,
        policy_hidden_dim=args.iql_hidden_dim, discount=args.discount,
        actor_lr=args.actor_lr, actor_beta=0.9, critic_lr=args.critic_lr, critic_beta=0.9,
        critic_tau=args.iql_target_tau, encoder_tau=args.iql_target_tau,
        quantile=args.iql_expectile, beta=args.iql_beta, clip_score=args.iql_advantage_clip,
        target_update_period=1, q_update_period=1, policy_update_period=1,
        detach_encoder=True, detach_conv=False, use_adamw=True, phi_config="psi_phi",
    )
    train_replay = GoalSliceFrameworkReplay(
        replay, args.batch_size, device, args.query_hit_oversample, train_indices,
    )
    capture = MetricCapture()
    rlkit_logger.logging_tool = capture
    env = FourRoomsGoalEnv(num_rooms=args.num_rooms, obs_img_dim=args.obs_img_dim, slip_prob=0.0)

    name = (
        "goal-slice-gcb-iql-rooms%d-fdim%d-rmin%s-iters%d-bs%d-qhit%.3g-seed%d"
        % (args.num_rooms, args.feature_dim, args.r_min, args.training_iterations,
           args.batch_size, args.query_hit_oversample, args.seed)
    )
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = start_wandb(args, name)
    umap_steps = umap_snapshot_steps(args.training_iterations, args.num_umap_images) if wandb_run and args.log_umap else set()
    checkpoint_steps = model_checkpoint_steps(args.training_iterations, args.num_checkpoints)
    started = time.time()

    with (output_dir / "metrics.jsonl").open("w") as metrics_file:
        for step in range(args.training_iterations):
            agent.update(train_replay, step)
            metrics = capture.consume()
            if step % args.log_freq == 0:
                metrics.update(
                    step=step,
                    **{"train/steps_per_second": (step + 1) / max(time.time() - started, 1e-8)},
                )
                write_metrics(metrics_file, wandb_run, metrics)
                log.info("step=%d phi=%.5f psi=%.5f q=%.5f v=%.5f actor=%.5f",
                         step, metrics.get("train/phi/loss", float("nan")),
                         metrics.get("train/psi/loss", float("nan")),
                         metrics.get("train/critic/Q_loss", float("nan")),
                         metrics.get("train/critic/V_loss", float("nan")),
                         metrics.get("train/actor/pi_loss", float("nan")))

            if step % args.eval_freq == 0 or step + 1 == args.training_iterations:
                metrics = evaluate_collapse(critic_representation.phi, env, args, device)
                metrics.update({"stats_eval/" + key: value for key, value in metrics.items() if key.startswith("collapse/")})
                metrics.update(held_out_policy_evaluation(agent, args, device))
                metrics["step"] = step
                write_metrics(metrics_file, wandb_run, metrics)
                log.info("step=%d separation=%.3f eval_return=%.3f included_success=%.3f",
                         step, metrics["collapse/separation_ratio"], metrics["eval/rl/return_mean"],
                         metrics["eval/rl/query_included_success_rate"])

            if wandb_run and args.log_umap and step in umap_steps:
                log_training_umap(critic_representation.phi, env, args, device, wandb_run, output_dir, step, split="train")
                if args.evaluate_on_held_out:
                    log_training_umap(critic_representation.phi, env, args, device, wandb_run, output_dir, step, split="eval")

            if step in checkpoint_steps:
                checkpoint = {
                    "agent": agent.state_dict(),
                    "args": vars(args),
                    "step": step + 1,
                }
                checkpoint_path = output_dir / (
                    "goal_slice_gcb_iql.pt" if step + 1 == args.training_iterations
                    else "goal_slice_gcb_iql_step_%07d.pt" % (step + 1)
                )
                torch.save(checkpoint, checkpoint_path)

    if args.training_iterations - 1 not in checkpoint_steps:
        torch.save({"agent": agent.state_dict(), "args": vars(args), "step": args.training_iterations},
                   output_dir / "goal_slice_gcb_iql.pt")
    if wandb_run:
        wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rooms", type=int, default=8, choices=(4, 8, 16))
    parser.add_argument("--obs-img-dim", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=75)
    parser.add_argument("--dataset-loc", default="")
    parser.add_argument("--dataset-steps", type=int, default=200000)
    parser.add_argument("--number-training-points", type=int, default=180000)
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--training-iterations", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--query-hit-oversample", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--validation-freq", type=int, default=500)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=.5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--num-checkpoints", type=int, default=10)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-filters", type=int, default=32)
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--metric-loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--discount", type=float, default=.99)
    parser.add_argument("--action-weight", type=float, default=25.)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--r-universal", type=float, default=1.)
    parser.add_argument("--r-empty", type=float, default=-1.)
    parser.add_argument("--r-min", type=float, default=-.25)
    parser.add_argument("--evaluate-on-held-out", action="store_true")
    parser.add_argument("--collapse-eval-states", type=int, default=32)
    parser.add_argument("--collapse-sets-per-status", type=int, default=16)
    parser.add_argument("--log-umap", action="store_true")
    parser.add_argument("--num-umap-images", type=int, default=50)
    parser.add_argument("--global-umap-states", type=int, default=32)
    parser.add_argument("--individual-umap-contexts", type=int, default=4)
    parser.add_argument("--umap-sets-per-class", type=int, default=64)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=.1)
    parser.add_argument("--umap-usetex", action="store_true")
    parser.add_argument("--iql-hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--iql-expectile", type=float, default=.7)
    parser.add_argument("--iql-beta", type=float, default=3.0)
    parser.add_argument("--iql-advantage-clip", type=float, default=100.)
    parser.add_argument("--iql-target-tau", type=float, default=.005)
    parser.add_argument("--rl-eval-episodes-per-status", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="goal_slice_gcb_iql_runs")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--project-name", default="gcb-four-rooms")
    parser.add_argument("--group", default="")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    if args.r_universal == args.r_empty:
        raise ValueError("r_universal and r_empty must differ for goal-set control")
    seed_everywhere(args.seed)
    suffix = "_heldout80_20" if args.evaluate_on_held_out else ""
    dataset = args.dataset_loc or "replay_goal_slice_four_rooms_%d%s.pt" % (args.num_rooms, suffix)
    if not os.path.exists(dataset):
        if args.no_collect:
            raise FileNotFoundError("Dataset missing: %s" % dataset)
        collect(args, dataset)
    elif args.evaluate_on_held_out:
        manifest = Path(dataset).with_suffix(".goal_set_split.json")
        if not manifest.exists():
            raise FileNotFoundError("Held-out evaluation requires %s" % manifest)
    log.info("Using compatible goal-slice replay: %s", dataset)
    train(args, dataset)


if __name__ == "__main__":
    main()
