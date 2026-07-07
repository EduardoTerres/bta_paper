import argparse
import logging
import os
import time

import numpy as np
import torch

from offline_gcab_four_rooms_vision import oracle_action
from state_gcab.environment import VectorFourRoomsGoalEnv
from state_gcab.replaybuffer import VectorGoalReplayBuffer
from state_gcab.trainer import train_state_representation_offline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("four_rooms_state")


def collect_four_rooms_replay(details, save_path, num_steps):
    log.info("Collecting %d replay steps (one-hot state obs) -> %s", num_steps, save_path)
    log.info("domain_kwargs: %s", details["env_kwargs"]["domain_kwargs"])

    env = VectorFourRoomsGoalEnv(**details["env_kwargs"]["domain_kwargs"])
    replay = VectorGoalReplayBuffer(
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

    while replay.idx < num_steps:
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
    dataset_loc = args.dataset_loc or f"replay_four_rooms_onehot_{args.num_rooms}.pt"
    return dict(
        discount=0.99,
        env_kwargs=dict(
            domain_kwargs=dict(
                num_rooms=args.num_rooms,
                max_episode_steps=args.max_episode_steps,
                dense_rewards=False,
                slip_prob=0.0,
            ),
        ),
        dataset_loc=dataset_loc,
        replay_buffer_kwargs=dict(
            capacity=args.dataset_steps + args.max_episode_steps,
            batch_size=256,
        ),
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
            num_layers=2,
            num_filters=128,
            metric_loss="l1",
            lr=5e-5,
            weight_decay=5e-4,
            num_layers_paired=2,
            num_filters_paired=128,
            lr_paired=5e-5,
            weight_decay_paired=5e-4,
            dynamics_loss="delta",
            ground_space=True,
            decode_both=True,
            dual_optimization=False,
            steps_till_on_policy=0,
            phi_updates_before_psi=0,
        ),
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
        number_training_points=args.number_training_points,
        num_eval_episodes=args.num_rooms,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        eval_video_save_dir="" if args.no_video else (
            args.eval_video_save_dir or f"eval_videos_state_{args.num_rooms}"),
        eval_video_record_number=args.eval_video_record_number,
        composition_eval=not args.no_composition_eval,
        composition_min_goals=args.composition_min_goals,
        composition_max_goals=args.composition_max_goals or None,
        composition_num_sets=args.composition_num_sets,
        composition_episodes_per_set=args.composition_episodes_per_set,
        use_wandb=args.use_wandb,
        offline_wandb=False,
        device=args.device,
        project_name="gcb-four-rooms-state",
        group=f"GCRB_four_rooms_state_{args.num_rooms}",
        name=f"gcb-rooms-state-{args.num_rooms}",
        seed=np.random.randint(99999),
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
    parser.add_argument("--eval-video-save-dir", type=str, default="",
                         help="Where to save rendered rollout/goal GIFs during eval "
                              "(default: eval_videos_state_<num-rooms>)")
    parser.add_argument("--eval-video-record-number", type=int, default=10,
                         help="Max number of episodes to render per eval pass")
    parser.add_argument("--no-video", action="store_true",
                         help="Disable rendering/saving rollout images during eval")
    parser.add_argument("--no-composition-eval", action="store_true",
                         help="Disable W&B composition rollouts during eval")
    parser.add_argument("--composition-min-goals", type=int, default=2,
                         help="Smallest sampled composed goal-set size")
    parser.add_argument("--composition-max-goals", type=int, default=0,
                         help="Largest sampled composed goal-set size (0 means all goals)")
    parser.add_argument("--composition-num-sets", type=int, default=1,
                         help="Number of random goal sets to sample per set size")
    parser.add_argument("--composition-episodes-per-set", type=int, default=1,
                         help="Rollouts to run for each sampled composed goal set")
    args = parser.parse_args()

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

    if not args.no_collect and not os.path.exists(variant["dataset_loc"]):
        log.info("Dataset %s not found, collecting a new one", variant["dataset_loc"])
        collect_four_rooms_replay(variant, variant["dataset_loc"], args.dataset_steps)
    else:
        log.info("Using existing dataset %s", variant["dataset_loc"])

    log.info("Starting train_state_representation_offline")
    train_start = time.time()
    train_state_representation_offline(variant)
    log.info("Finished training in %.1fs", time.time() - train_start)
