import logging
import os

import imageio
import numpy as np
import torch

from goalbisim.utils.misc_utils import eval_mode
from goalbisim.utils.video import MultiVideoRecorder
from rlkit.core import logger

log = logging.getLogger("state_gcab.eval")


def evaluate_state_goal_agent(env, agent, discount, num_episodes, step,
                               video_save_dir="", record_number=10, fps=5):
    """Success-rate evaluation for vector-obs envs. If `video_save_dir` is set, also
    renders rollouts/goals as grid images via env.render_obs()/render_goal()
    (visualization only -- the agent itself only ever sees vectors) and saves them as
    local GIFs, plus pushes them to W&B via the unmodified MultiVideoRecorder.save().
    """
    recording = bool(video_save_dir)
    records = 0

    if recording:
        video = MultiVideoRecorder(dir_name=video_save_dir, width=5, fps=fps)
        goalvideo = MultiVideoRecorder(dir_name=video_save_dir, width=5, fps=1)
        record_number = min(record_number, num_episodes)
        goalvideo.init(num_trajectories=record_number, max_trajectory_length=4)
        video.init(num_trajectories=record_number, max_trajectory_length=env._max_episode_steps + 1)

    successes = []
    rewards = []

    for _ in range(num_episodes):
        obs, goal, extra = env.reset()
        init_obs = obs
        done = False
        episode_reward = 0.0
        episode_step = 0
        success = 0.0

        if recording and records < record_number:
            for _ in range(4):
                goalvideo.record(env.render_goal())
            goalvideo.step()
            video.record(env.render_obs())

        while not done:
            with eval_mode(agent):
                action = agent.sample_action(obs, goal, init_obs=init_obs)

            obs, reward, done, extra = env.step(action)

            if recording and records < record_number:
                video.record(env.render_obs())

            success = float(reward > 0.01)
            episode_reward += reward * (discount ** episode_step)
            episode_step += 1

        if recording and records < record_number:
            video.step()
            records += 1

        successes.append(success)
        rewards.append(episode_reward)

    stats = {
        'train_step': step,
        'eval/avg_success': float(np.mean(successes)),
        'eval/std_success': float(np.std(successes)),
        'eval/avg_episode_reward': float(np.mean(rewards)),
        'eval/std_episode_reward': float(np.std(rewards)),
    }
    logger.logging_tool.log(stats)

    if recording:
        # Save local GIFs first: they don't depend on the network/W&B backend, so a
        # dead W&B connection (e.g. lost network on a compute node) can't lose them.
        _save_local_gif(video, os.path.join(video_save_dir, f"rollouts_step{step}.gif"), fps)
        _save_local_gif(goalvideo, os.path.join(video_save_dir, f"goals_step{step}.gif"), 1)

        try:
            video.save("eval/video_rollouts", step)
            goalvideo.save("eval/goals", step)
        except Exception:
            log.warning(
                "Failed to push rollout/goal videos to W&B (backend may have died); "
                "local GIFs under %s were still saved, continuing training.",
                video_save_dir, exc_info=True,
            )

    return float(np.mean(successes)), stats


def evaluate_composition_agent(
        env, agent, discount, step, min_goals=2, max_goals=None,
        num_sets_per_size=1, episodes_per_set=1):
    """Evaluate OR-composed tasks: each sampled set succeeds at any goal in the set.

    The composed policy is selected in abstract space by scoring each primitive goal
    with the learned critic value V(s, g), then executing the policy for the best goal.
    """
    all_goals = list(env.goal_positions)
    if max_goals is None:
        max_goals = len(all_goals)
    max_goals = min(max_goals, len(all_goals))
    min_goals = max(min_goals, 2)

    stats = {'train_step': step}
    all_rewards = []
    all_successes = []

    for goal_count in range(min_goals, max_goals + 1):
        rewards = []
        successes = []

        for _ in range(num_sets_per_size):
            goal_idxs = np.random.choice(len(all_goals), size=goal_count, replace=False)
            goal_set = [all_goals[idx] for idx in goal_idxs]

            for _ in range(episodes_per_set):
                reward, success = _composition_rollout(env, agent, goal_set, discount)
                rewards.append(reward)
                successes.append(success)

        stats[f'composition/reward_{goal_count}_goals'] = float(np.mean(rewards))
        stats[f'composition/success_{goal_count}_goals'] = float(np.mean(successes))
        stats[f'composition/std_reward_{goal_count}_goals'] = float(np.std(rewards))
        all_rewards.extend(rewards)
        all_successes.extend(successes)

    if all_rewards:
        stats['composition/avg_reward'] = float(np.mean(all_rewards))
        stats['composition/avg_success'] = float(np.mean(all_successes))

    logger.logging_tool.log(stats)
    return stats


def _composition_rollout(env, agent, goal_set, discount):
    obs, _, _ = env.reset()
    env.set_goal_positions(goal_set)
    init_obs = obs
    done = False
    episode_reward = 0.0
    episode_step = 0
    success = 0.0

    while not done:
        with eval_mode(agent):
            goal = _select_composed_goal(env, agent, obs, goal_set)
            action = agent.sample_action(obs, goal, init_obs=init_obs)

        obs, reward, done, _ = env.step(action)
        success = float(reward > 0.01)
        episode_reward += reward * (discount ** episode_step)
        episode_step += 1

    return episode_reward, success


def _select_composed_goal(env, agent, obs, goal_set):
    obs_batch = np.repeat(obs[None], len(goal_set), axis=0)
    goal_batch = np.stack([env._one_hot_position(goal) for goal in goal_set], axis=0)

    obs_tensor = agent.eval_transforms(obs_batch, agent.device)
    goal_tensor = agent.eval_transforms(goal_batch, agent.device)
    with torch.no_grad():
        values = agent.critic.forward_v(obs_tensor, goal_tensor).squeeze(-1)
        best_idx = int(torch.argmax(values).item())
    return goal_batch[best_idx]


def _save_local_gif(recorder, path, fps):
    """MultiVideoRecorder.save() only pushes to W&B when use_wandb=True (otherwise it's
    a no-op), so also write the frames to a local GIF directly, independent of W&B.
    """
    if recorder.frames is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = np.transpose(recorder.frames, (0, 2, 3, 1))  # (T, C, H, W) -> (T, H, W, C)
    imageio.mimsave(path, frames, fps=max(int(fps), 1))
