import logging
import random

import numpy as np
import torch
from tqdm import tqdm

from goalbisim.agents.agent_init import agent_initalization
from loggers.wandb.wandb_init import setup_logger
from rlkit.core import logger

from .environment import VectorFourRoomsGoalEnv
from .eval import evaluate_composition_agent, evaluate_state_goal_agent
from .goalbisim import StateGoalBisim
from .replaybuffer import VectorGoalReplayBuffer
from .transforms import state_eval_transform

log = logging.getLogger("state_gcab.trainer")


def load_state_four_rooms_env(details):
    return VectorFourRoomsGoalEnv(**details['env_kwargs']['domain_kwargs'])


def init_state_replay(obs_shape, state_shape, action_shape, device, details):
    return VectorGoalReplayBuffer(
        obs_shape, state_shape, action_shape, device=device, **details['replay_buffer_kwargs'],
    )


def init_state_representation(obs_shape, action_shape, device, details):
    return StateGoalBisim(obs_shape, device, action_shape=action_shape, **details['goalbisim_kwargs'])


def train_state_representation_offline(details):
    """Offline GCAB/GoalBiSim training loop for low-dim state-action-goal transitions.

    Mirrors goalbisim.trainers.offline_trainers.train_representation_offline: same
    replay-buffer -> 3x representation -> IQL agent -> (eval, agent.update) loop, so the
    GCAB losses and training mechanism are unchanged. What's dropped is the pixel/video
    machinery (MultiVideoRecorder, image eval_transforms, VideoDistractor, online
    rollout phase) that doesn't apply to vector observations.
    """
    random.seed(details['seed'])
    np.random.seed(details['seed'])
    torch.manual_seed(details['seed'])

    setup_logger(details)
    logger.logging_tool.define_metric('train_step')
    logger.logging_tool.define_metric('composition/*', step_metric='train_step')

    env = load_state_four_rooms_env(details)
    eval_env = load_state_four_rooms_env(details)
    device = torch.device(details['device'])

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    state_shape = env.state_space.shape

    replay_buffer = init_state_replay(obs_shape, state_shape, action_shape, device, details)
    eval_replay_buffer = init_state_replay(obs_shape, state_shape, action_shape, device, details)

    replay_buffer.load(details['dataset_loc'], start=0, end=details['number_training_points'])
    eval_replay_buffer.load(details['dataset_loc'], start=details['number_training_points'])

    critic_representation = init_state_representation(obs_shape, action_shape, device, details)
    actor_representation = init_state_representation(obs_shape, action_shape, device, details)
    target_critic_representation = init_state_representation(obs_shape, action_shape, device, details)

    agent = agent_initalization(
        details, env, device, state_eval_transform,
        actor_representation, critic_representation, target_critic_representation,
    )

    best_success_rate = 0.0
    log_freq = details.get('log_freq', details['eval_freq'])
    save_model = details.get('save_model', True)
    snapshot_dir = logger.get_snapshot_dir()

    pbar = tqdm(range(details['training_iterations']), desc='offline training (state)')
    for step in pbar:
        if step % details['eval_freq'] == 0:
            success_rate, _ = evaluate_state_goal_agent(
                eval_env, agent, details['discount'], details['num_eval_episodes'], step,
                video_save_dir=details.get('eval_video_save_dir', ''),
                record_number=details.get('eval_video_record_number', 10),
            )

            if save_model and success_rate > best_success_rate:
                agent.save(snapshot_dir, 'best_agent')
                log.info("step=%d: new best success_rate=%.3f, saved checkpoint to %s/agents/",
                          step, success_rate, snapshot_dir)

            best_success_rate = max(best_success_rate, success_rate)
            pbar.set_postfix(success_rate=success_rate, best_success_rate=best_success_rate)

            logger.logging_tool.log({'train_step': step, 'eval/best_success': best_success_rate})
            agent.test_representation(eval_replay_buffer, step)
            if details.get('composition_eval', True):
                evaluate_composition_agent(
                    eval_env, agent, details['discount'], step,
                    min_goals=details.get('composition_min_goals', 2),
                    max_goals=details.get('composition_max_goals'),
                    num_sets_per_size=details.get('composition_num_sets', 1),
                    episodes_per_set=details.get('composition_episodes_per_set', 1),
                )
            logger.logging_tool.record(step=step)

        agent.update(replay_buffer, step)

        if log_freq and step % log_freq == 0 and step % details['eval_freq'] != 0:
            logger.logging_tool.record(step=step)

    if save_model:
        agent.save(snapshot_dir, 'final_agent')
        log.info("Saved final agent checkpoint (params being trained) to %s/agents/", snapshot_dir)

    return agent, best_success_rate
