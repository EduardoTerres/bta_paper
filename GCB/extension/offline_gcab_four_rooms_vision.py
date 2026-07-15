import argparse
import logging
import os
import random
import time
from collections import deque
from itertools import cycle

import numpy as np
import torch

import goalbisim.utils.hyperparameter as hyp
from environments.environment_init import load_env
from goalbisim.data.management.goalreplaybuffer import GoalReplayBuffer
from goalbisim.trainers.offline_trainers import _sample_joint_goal_sets, train_representation_offline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("four_rooms")


def one_hot(action, n=5):
    out = np.zeros(n, dtype=np.float32)
    out[action] = 1.0
    return out


def oracle_action(env):
    start = tuple(env.env.state[1])
    goals = (
        set(env._joint_goal_positions)
        if env._joint_goal_positions is not None
        else {tuple(env._goal_position)}
    )
    if start in goals:
        return one_hot(4)

    moves = [(0, (-1, 0)), (1, (0, 1)), (2, (1, 0)), (3, (0, -1))]
    walls = set(env.env.walls)
    queue = deque([(start, None)])
    seen = {start}

    while queue:
        pos, first_action = queue.popleft()
        for action, delta in moves:
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if nxt in seen or nxt in walls:
                continue
            if nxt not in env.env.possiblePositions:
                continue
            next_first = action if first_action is None else first_action
            if nxt in goals:
                return one_hot(next_first)
            seen.add(nxt)
            queue.append((nxt, next_first))

    log.warning("oracle_action: no path found from %s to any of %s, taking random action", start, goals)
    return one_hot(env.env.action_space.sample())


def collect_four_rooms_replay(details, save_path, num_steps):
    log.info("Collecting %d replay steps -> %s", num_steps, save_path)
    log.info("env_kwargs: %s", details["env_kwargs"])

    random.seed(details["seed"])
    np.random.seed(details["seed"])
    torch.manual_seed(details["seed"])

    env = load_env(details)
    env._rng.seed(details["seed"])
    replay = GoalReplayBuffer(
        env.observation_space.shape,
        env.state_space.shape,
        env.action_space.shape,
        capacity=num_steps + env._max_episode_steps,
        batch_size=details["replay_buffer_kwargs"]["batch_size"],
        device="cpu",
        discount=details["discount"],
    )

    start_time = time.time()
    log_every = max(num_steps // 20, 1)
    last_logged = 0
    num_episodes = 0
    num_successes = 0

    train_multi_goal = details.get("train_multi_goal", False)
    goal_set_cycle = None
    if train_multi_goal:
        # Reuse the exact fixed goal sets that eval's evaluate_joint_goal_agent will be
        # scored on (same seeded sampler), rather than resampling new sets every episode.
        joint_goal_sets = _sample_joint_goal_sets(env.goal_positions)
        train_goal_sets = [goals for sets in joint_goal_sets.values() for goals in sets]
        log.info(
            "train_multi_goal: cycling through %d fixed eval goal sets (sizes %s)",
            len(train_goal_sets), sorted(joint_goal_sets.keys()),
        )
        goal_set_cycle = cycle(train_goal_sets)

    while replay.idx < num_steps:
        if train_multi_goal:
            obs, goal, extra = env.reset_joint(next(goal_set_cycle))
        else:
            obs, goal, extra = env.reset()
        done = False
        episode_len = 0
        while not done:
            action = oracle_action(env)
            next_obs, reward, done, next_extra = env.step(action)
            replay.add(
                obs,
                extra["state"],
                action,
                reward,
                reward,
                next_obs,
                next_extra["state"],
                done,
                done,
                goal,
            )
            obs, extra = next_obs, next_extra
            episode_len += 1

        num_episodes += 1
        num_successes += int(reward > 0.01)

        if replay.idx - last_logged >= log_every:
            last_logged = replay.idx
            elapsed = time.time() - start_time
            rate = replay.idx / elapsed if elapsed > 0 else 0.0
            log.info(
                "steps=%d/%d episodes=%d success_rate=%.2f last_episode_len=%d elapsed=%.1fs (%.1f steps/s)",
                replay.idx, num_steps, num_episodes, num_successes / num_episodes,
                episode_len, elapsed, rate,
            )

    log.info(
        "Finished collecting %d steps across %d episodes (success_rate=%.2f) in %.1fs",
        replay.idx, num_episodes, num_successes / num_episodes, time.time() - start_time,
    )

    save_dir = os.path.dirname(save_path) or "."
    name = os.path.splitext(os.path.basename(save_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    replay.save(save_dir, name)
    log.info("Saved replay buffer to %s", os.path.join(save_dir, name + ".pt"))


def build_variant(args):
    goal_mode = "multigoal" if args.train_multi_goal else "singlegoal"
    dataset_suffix = "_multigoal" if args.train_multi_goal else ""
    dataset_loc = args.dataset_loc or f"replay_four_rooms_{args.num_rooms}{dataset_suffix}.pt"
    return dict(
        training_form="dataset",
        train_multi_goal=args.train_multi_goal,
        discount=0.99,
        env_kwargs=dict(
            package="four_rooms",
            domain_name="four_rooms",
            domain_kwargs=dict(
                num_rooms=args.num_rooms,
                max_episode_steps=args.max_episode_steps,
                obs_img_dim=64,
                dense_rewards=False,
                slip_prob=0.0,
            ),
            frame_stack_count=1,
            action_repeat=1,
        ),
        dataset_loc=dataset_loc,
        replay_buffer_type="Goal",
        replay_buffer_kwargs=dict(
            capacity=args.dataset_steps + args.max_episode_steps,
            batch_size=256,
        ),
        representation_algorithm="GoalBiSim",
        goalbisim_kwargs=dict(
            transition_model_type="next_observation",
            psi_loss_form="direct",
            metric_distance="reward",
            decoder_type="reward",
            on_policy_dynamics="",
            use_contrastive=False,
            contrastive_weight=0.25,
            action_weight=25,
            train_iters_per_update_psi=1,
            train_iters_per_update_phi=1,
            discount=0.99,
            feature_dim=128,
            num_layers=6,
            num_filters=32,
            metric_loss="l1",
            lr=5e-5,
            weight_decay=5e-4,
            num_layers_paired=6,
            num_filters_paired=32,
            lr_paired=5e-5,
            weight_decay_paired=5e-4,
            dynamics_loss="delta",
            ground_space=True,
            decode_both=True,
            dual_optimization=False,
            steps_till_on_policy=0,
            phi_updates_before_psi=0,
            compositionality_weight=args.compositionality_weight,
        ),
        training_transforms=[],
        eval_transforms=[],
        iql_kwargs=dict(
            discount=0.99,
            actor_lr=5e-5,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            critic_lr=5e-5,
            critic_beta=0.9,
            critic_tau=0.005,
            encoder_tau=0.01,
            quantile=0.7,
            policy_update_period=1,
            q_update_period=1,
            target_update_period=1,
            clip_score=100,
            soft_target_tau=0.005,
            beta=1.0 / 3,
            detach_encoder=True,
            detach_conv=False,
            use_adamw=True,
            phi_config="psi",
        ),
        rl_algorithm="IQL",
        training_iterations=args.training_iterations,
        online_training_trajectories=0,
        num_eval_episodes=args.num_rooms,
        pre_eval_freq=0,
        eval_video_save_dir="",
        eval_analogy_save_dir="",
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        eval_traj_freq=0,
        tests=[],
        use_wandb=args.use_wandb,
        reload_best_agent=False,
        offline_wandb=False,
        save_model=False,
        analogy_goal=False,
        save_wandb_video=False,
        device=args.device,
        project_name="gcb-four-rooms",
        group=f"GCRB_four_rooms_{args.num_rooms}_{goal_mode}_iters{args.training_iterations}_compw{args.compositionality_weight}",
        name=f"gcb-rooms-{args.num_rooms}_{goal_mode}_iters{args.training_iterations}_compw{args.compositionality_weight}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rooms", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--dataset-loc", type=str, default="")
    parser.add_argument("--dataset-steps", type=int, default=60000)
    parser.add_argument("--number-training-points", type=int, default=50000)
    parser.add_argument("--training-iterations", type=int, default=20005)
    parser.add_argument("--eval-freq", type=int, default=4000)
    parser.add_argument("--log-freq", type=int, default=100,
                         help="How often (in steps) to push training losses to W&B")
    parser.add_argument("--max-episode-steps", type=int, default=75)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--train-multi-goal", action="store_true",
                         help="Collect episodes on the same fixed joint goal sets used by "
                              "eval's evaluate_joint_goal_agent (env.reset_joint), instead of "
                              "only single-goal episodes")
    parser.add_argument("--compositionality-weight", type=float, default=0.0,
                         help="Weight on the auxiliary loss enforcing "
                              "psi(g1 & g2) ~= min(psi(g1), psi(g2)) and "
                              "psi(g1 | g2) ~= max(psi(g1), psi(g2)) over randomly sampled "
                              "goal-set pairs. 0 disables it (default)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for reproducibility; random if unset")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(np.random.randint(99999))

    log.info("Args: %s", vars(args))

    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.warning("--device=cuda requested but torch.cuda.is_available() is False; "
                        "check the torch/CUDA build in this environment")
        else:
            idx = torch.cuda.current_device()
            log.info(
                "Using GPU %d: %s (compute capability %s), torch=%s cuda=%s",
                idx, torch.cuda.get_device_name(idx), torch.cuda.get_device_capability(idx),
                torch.__version__, torch.version.cuda,
            )

    variant = build_variant(args)
    search_space = {
        "seed": [args.seed],
        "number_training_points": [args.number_training_points],
        "add_her_relabels": [False],
        "representation_pre_training_iterations": [0],
        "policy_pre_training_iterations": [0],
        "allow_pre_critic_gradients": [False],
        "step_lr": [False],
        "use_distractor": [False],
        "disconnect_psi": [False],
        "batch_size": [256],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = sweeper.iterate_hyperparameters()
    log.info("Running %d variant(s)", len(variants))

    for i, variant in enumerate(variants):
        variant["replay_buffer_kwargs"]["batch_size"] = variant["batch_size"]

        log.info(
            "=== Variant %d/%d: seed=%s dataset_loc=%s batch_size=%s ===",
            i + 1, len(variants), variant["seed"], variant["dataset_loc"], variant["batch_size"],
        )

        if not args.no_collect and not os.path.exists(variant["dataset_loc"]):
            log.info("Dataset %s not found, collecting a new one", variant["dataset_loc"])
            collect_four_rooms_replay(
                variant,
                variant["dataset_loc"],
                args.dataset_steps,
            )
        else:
            log.info("Using existing dataset %s", variant["dataset_loc"])

        log.info("Starting train_representation_offline for variant %d/%d", i + 1, len(variants))
        train_start = time.time()
        train_representation_offline(variant)
        log.info(
            "Finished variant %d/%d in %.1fs", i + 1, len(variants), time.time() - train_start,
        )
