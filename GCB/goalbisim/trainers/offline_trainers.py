import torch
import numpy as np
import dmc2gym
import time
from tqdm import tqdm
from goalbisim.utils.misc_utils import set_seed_everywhere, FrameStack, GoalFrameStack
from goalbisim.data.management.replaybuffer import ReplayBuffer
from goalbisim.agents.pixelsac import PixelSACAgent
from goalbisim.agents.goalpixelsac import GoalPixelSACAgent
from goalbisim.representation.base_representation import initialize_representation
from rlkit.core import logger
from goalbisim.testing.eval_rl import evaluate_agent
from goalbisim.testing.eval_rl import evaluate_goal_agent
from goalbisim.testing.eval_rl import evaluate_joint_goal_agent
from goalbisim.utils.misc_utils import eval_mode
from loggers.wandb.wandb_init import setup_logger
from environments.environment_init import load_env
from goalbisim.data.management.init_replay import init_replay
from goalbisim.data.manipulation.transform import initialize_transform
from goalbisim.testing.init_testing import init_testing
from goalbisim.agents.agent_init import agent_initalization
import wandb
import random
from itertools import combinations


JOINT_EVAL_SEED = 42
JOINT_EVAL_MAX_SETS = 5
JOINT_EVAL_EPISODES = 5

COMPOSITIONALITY_SEED = 43
COMPOSITIONALITY_SAMPLES_PER_N = 16


def _sample_joint_goal_sets(goal_positions):
    """For N=2..len(goal_positions), fix (once, via a constant seed independent of the
    training seed) up to JOINT_EVAL_MAX_SETS goal subsets of size N. These are reused for
    every evaluation call throughout training so the joint-MDP metrics are comparable over time."""
    rng = random.Random(JOINT_EVAL_SEED)
    goal_sets = {}
    for n in range(2, len(goal_positions) + 1):
        combos = list(combinations(goal_positions, n))
        rng.shuffle(combos)
        goal_sets[n] = combos[:JOINT_EVAL_MAX_SETS]
    return goal_sets


def _sample_compositionality_tuples(env, goal_positions, samples_per_n=COMPOSITIONALITY_SAMPLES_PER_N,
                                     min_n=2, max_n=None, seed=COMPOSITIONALITY_SEED):
    """Fix (once, via a constant seed) a pool of N-goal tuples for N=min_n..max_n. Each
    `member` is a region (position set) built from a shared, randomly sized `core` plus
    distinct extra cells, so intersection(members) == core is guaranteed non-empty and
    varies per sample -- unlike intersecting N distinct singleton goals, which is always
    empty. Paired at train time with a live state s (see PairedStateGoal.compositionality_loss)
    to train phi(s, g_1 ∪ ... ∪ g_N) ≈ max_i phi(s, g_i) and
    phi(s, g_1 ∩ ... ∩ g_N) ≈ min_i phi(s, g_i)."""
    rng = random.Random(seed)
    goal_positions = [tuple(p) for p in goal_positions]
    max_n = max_n or len(goal_positions)

    pool = {}
    for n in range(min_n, max_n + 1):
        member_imgs, union_imgs, inter_imgs = [], [], []
        for _ in range(samples_per_n):
            core_size = rng.randint(1, max(1, len(goal_positions) - n))
            core = set(rng.sample(goal_positions, core_size))
            remaining = [p for p in goal_positions if p not in core]
            members = []
            for _ in range(n):
                extra = set(rng.sample(remaining, rng.randint(0, len(remaining))))
                members.append(core | extra)
            member_imgs.append([env.render_goal_set(m) for m in members])
            union_imgs.append(env.render_goal_set(set.union(*members)))
            inter_imgs.append(env.render_goal_set(set.intersection(*members)))

        # Encoders assert inputs lie in [0, 1] (replay buffer samples are normalized the
        # same way via `.float() / 255`, see goalreplaybuffer.py), so normalize here too.
        members_t = torch.as_tensor(np.array(member_imgs), dtype=torch.float32) / 255  # (S, N, 3, H, W)
        union_t = torch.as_tensor(np.stack(union_imgs), dtype=torch.float32) / 255
        inter_t = torch.as_tensor(np.stack(inter_imgs), dtype=torch.float32) / 255
        pool[n] = (members_t, union_t, inter_t)

    return pool


def _sample_negation_pairs(env, goal_positions, samples_per_size=COMPOSITIONALITY_SAMPLES_PER_N,
                            seed=COMPOSITIONALITY_SEED + 1):
    """Fix (once, via a constant seed) a pool of (goal_set, NOT goal_set) pairs across
    every subset size k=1..len(goal_positions)-1 -- varying sizes, rather than only ever
    negating singletons -- where NOT goal_set := goal_positions \\ goal_set, guaranteed
    non-empty on both sides since 1 <= k <= len(goal_positions) - 1. Paired at train time
    with a live state s to train phi(s, NOT g) ≈ 1 - phi(s, g)."""
    rng = random.Random(seed)
    goal_positions = [tuple(p) for p in goal_positions]
    total = len(goal_positions)

    pool = {}
    for k in range(1, total):
        goal_imgs, not_goal_imgs = [], []
        for _ in range(samples_per_size):
            subset = set(rng.sample(goal_positions, k))
            complement = set(goal_positions) - subset
            goal_imgs.append(env.render_goal_set(subset))
            not_goal_imgs.append(env.render_goal_set(complement))

        goal_t = torch.as_tensor(np.stack(goal_imgs), dtype=torch.float32) / 255
        not_goal_t = torch.as_tensor(np.stack(not_goal_imgs), dtype=torch.float32) / 255
        pool[k] = (goal_t, not_goal_t)

    return pool


def train_representation_offline(details):

    random.seed(details['seed'])
    np.random.seed(details['seed'])
    torch.manual_seed(details['seed'])

    logging_tool = setup_logger(details)
    if details['use_wandb']:
        wandb.log(details) #Logs Details

    env = load_env(details)
    eval_env = load_env(details)

    joint_goal_sets = _sample_joint_goal_sets(eval_env.goal_positions) if hasattr(eval_env, 'goal_positions') else {}

    compositionality_weight = details.get('goalbisim_kwargs', {}).get('compositionality_weight', 0.0)
    compositionality_minmax_pool = None
    compositionality_neg_pool = None
    if compositionality_weight > 0 and hasattr(eval_env, 'goal_positions'):
        compositionality_minmax_pool = _sample_compositionality_tuples(eval_env, eval_env.goal_positions)
        compositionality_neg_pool = _sample_negation_pairs(eval_env, eval_env.goal_positions)

    device = torch.device(details['device'])

    test_functions = init_testing(details)
    env = load_env(details)
    obs_shape = env.observation_space.shape
    eval_transforms = initialize_transform(details['eval_transforms'])

    replay_buffer = init_replay(obs_shape, env.action_space.shape, device, details, state_shape = env.state_space.shape)
    eval_replay_buffer = init_replay(obs_shape, env.action_space.shape, device, details, state_shape = env.state_space.shape)

    #Technically offline replay buffer, with consistent policy, but shouldn't matter...

    if details['training_form'] == 'dataset':
        replay_buffer.load(details['dataset_loc'], start = 0, end = details['number_training_points'])
        eval_replay_buffer.load(details['dataset_loc'], start = details['number_training_points'])
    elif details['training_form'] == 'policy':
        demo_critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details['demo_policy_representation_kwargs'], main = True)
        demo_actor_representation = initialize_representation(obs_shape, env.action_space.shape, device, details['demo_policy_representation_kwargs'])
        demo_target_critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details['demo_policy_representation_kwargs'])
        
        demo_agent = GoalPixelSACAgent(obs_shape, env.action_space.shape, device, eval_transforms, demo_actor_representation, demo_critic_representation, demo_target_critic_representation,
        None, **details['sac_kwargs'])
        load_policy = demo_agent.load(details['policy_loc'], 'best_agent')
        replay_buffer = collect_offline_replay(demo_agent, env, details['discount'], replay_buffer, sample_points = details['sample_points'])
        eval_replay_buffer = collect_offline_replay(demo_agent, env, details['discount'], eval_replay_buffer, sample_points = details['eval_sample_points'])

        del demo_agent

    elif details['training_form'] == 'gail':
        raise NotImplementedError

    else:
        raise NotImplementedError

    critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details, main = True)
    actor_representation = initialize_representation(obs_shape, env.action_space.shape, device, details)
    target_critic_representation = initialize_representation(obs_shape, env.action_space.shape, device, details)

    best_critic_representation = initialize_representation(env.observation_space.shape, env.action_space.shape, device, details, main = True)
    best_actor_representation = initialize_representation(env.observation_space.shape, env.action_space.shape, device, details)
    best_target_critic_representation = initialize_representation(env.observation_space.shape, env.action_space.shape, device, details)

    if compositionality_minmax_pool is not None:
        critic_representation.set_compositionality_pool(compositionality_minmax_pool, compositionality_neg_pool)

    if details['add_her_relabels']:
        replay_buffer.relabel()

    agent = agent_initalization(details, env, device, eval_transforms, actor_representation, critic_representation, target_critic_representation)
    best_agent = agent_initalization(details, env, device, eval_transforms, best_actor_representation, best_critic_representation, best_target_critic_representation)

    if details['use_distractor']:
        from goalbisim.utils.video import VideoDistractor
        train_distractor = VideoDistractor(env, details['distractor_kwargs'], replay_buffer = replay_buffer, ratio_end = .2)
        eval_distractor = VideoDistractor(eval_env, details['distractor_kwargs'], replay_buffer = eval_replay_buffer, ratio_start = .2, ratio_end = .4)

        env.set_distractor(train_distractor)
        eval_env.set_distractor(eval_distractor)

        replay_buffer.overlay(train_distractor)
        eval_replay_buffer.overlay(eval_distractor)


    best_success_rate = 0
    add_to = 0
    log_freq = details.get('log_freq', details['eval_freq'])

    pbar = tqdm(range(details['training_iterations']), desc='offline training')
    for step in pbar:
        if step % details['eval_freq'] == 0:
            conditional = True if details['representation_algorithm'] == 'Ccvae' else False

            success_rate, video, goalvideo, success_stats, seed_list, samples = evaluate_goal_agent(eval_env, agent, details['discount'], details['num_eval_episodes'], step + add_to, details, conditional = conditional, analogy_goal = details.get('analogy_goal', False))
            rel_success_rate, rel_video, rel_goalvideo, rel_success_stats, rel_seed_list, rel_samples = evaluate_goal_agent(eval_env, best_agent, details['discount'], details['num_eval_episodes'], step + add_to, details, conditional = conditional, analogy_goal = details.get('analogy_goal', False))

            pbar.set_postfix(success_rate=success_rate, best_success_rate=best_success_rate)

            if success_rate > rel_success_rate:
                agent.save(logger.get_snapshot_dir(), 'best_agent')
                best_agent.load(logger.get_snapshot_dir(), 'best_agent')

                if best_success_rate < success_rate:
                    best_success_rate = success_rate

                goalvideo.save("eval/goals", step + add_to)
                video.save("eval/video_rollouts", step + add_to)
                logger.logging_tool.log(success_stats)

            else:
                if details['reload_best_agent']:
                    try:
                        agent.load(logger.get_snapshot_dir(), 'best_agent')
                    except:
                        pass
                if details['step_lr']:
                    agent.step_all()
                    
                rel_goalvideo.save("eval/goals", step + add_to)
                rel_video.save("eval/video_rollouts", step + add_to)
                logger.logging_tool.log(rel_success_stats)

                if rel_success_rate > best_success_rate:
                    best_success_rate = rel_success_rate

            stats = {'train_step' : step,
                    'eval/best_sucess': best_success_rate,
                    }
            logger.logging_tool.log(stats)

            for n, goal_sets in joint_goal_sets.items():
                set_success_rates, set_episode_rewards = zip(*[
                    evaluate_joint_goal_agent(eval_env, agent, details['discount'], goals, JOINT_EVAL_EPISODES, conditional=conditional)
                    for goals in goal_sets
                ])
                logger.logging_tool.log({
                    'train_step': step + add_to,
                    f'eval_joint/avg_success/N_{n}': float(np.mean(set_success_rates)),
                    f'eval_joint/avg_episode_reward/N_{n}': float(np.mean(set_episode_rewards)),
                })



                

            for test in test_functions:
                try:
                    test(eval_env, eval_transforms, device, replay_buffer, critic_representation, step + add_to, details, train_set = True, eval_replay_buffer = replay_buffer)
                    test(eval_env, eval_transforms, device, replay_buffer, critic_representation, step + add_to, details, eval_replay_buffer = eval_replay_buffer)
                    test(eval_env, eval_transforms, device, replay_buffer, critic_representation, step + add_to, details, eval_replay_buffer = eval_replay_buffer, forced_samples = samples)
                except:
                    pass

            agent.test_representation(eval_replay_buffer, step + add_to)
            logger.logging_tool.record()

        agent.update(replay_buffer, step + add_to)

        if log_freq and step % log_freq == 0 and step % details['eval_freq'] != 0:
            logger.logging_tool.record()

    add_to = add_to + details['training_iterations']
    ignore_first = True
    total_steps = 0
    total_episode_reward = 0
    total_successes = 0
    #assert replay_buffer.capacity > details['online_training_iterations'] + replay_buffer.idx
    discount = 0.99
    for traj in range(details['online_training_trajectories']):
        if traj % details['eval_traj_freq'] == 0:
            success_rate, video, goalvideo, success_stats, seed_list, samples = evaluate_goal_agent(eval_env, agent, details['discount'], details['num_eval_episodes'], total_steps + add_to, details)
            goalvideo.save("online/eval/goals", total_steps + add_to)
            video.save("online/eval/video_rollouts", total_steps + add_to)
            logger.logging_tool.log(success_stats)
            if success_rate > best_success_rate:
                agent.save(logger.get_snapshot_dir(), 'best_agent')
                #best_agent.load(logger.get_snapshot_dir(), 'best_agent')

            else:
                #try:
                if details['reload_best_agent']:
                    try:
                        agent.load(logger.get_snapshot_dir(), 'best_agent')
                    except:
                        pass
                if details['step_lr']:
                    agent.step_all()
                

            for test in test_functions:
                test(eval_env, eval_transforms, device, replay_buffer, critic_representation, total_steps + add_to, details, train_set = True, eval_replay_buffer = replay_buffer)
                test(eval_env, eval_transforms, device, replay_buffer, critic_representation, total_steps + add_to, details, eval_replay_buffer = eval_replay_buffer)
                test(eval_env, eval_transforms, device, replay_buffer, critic_representation, total_steps + add_to, details, eval_replay_buffer = eval_replay_buffer, forced_samples = samples)

            agent.test_representation(eval_replay_buffer, total_steps + add_to)
            logger.logging_tool.record()

        if not ignore_first:

            total_successes += float(is_success)
            total_episode_reward += float(episode_reward)


            stats = {'train_step' : total_steps,
                'online/train/success': done_bool,
                'online/train/average_episode_reward' : total_episode_reward/traj,
                'online/train/average_success' : total_successes/traj}
            logger.logging_tool.log(stats)

            episode_reward = 0
            episode_step = 0
            reward = 0
            #total
                #logger.dump_tabular(with_prefix=True, with_timestamp=False)

        else:
            ignore_first = False
            #state = extra['state']
            episode_reward = 0
            episode_step = 0
            reward = 0
            #logger.logging_tool.record()


        obs, goal, extra = env.reset()

        state = extra['state']

        done = False

        #ignore_first = True

        while not done:
            
            #episode += 1
            

            with eval_mode(agent): #Gather Data but detach
                action = agent.sample_action(obs, goal)

            curr_reward = reward
            next_obs, reward, is_success, extra = env.step(action)

            next_state = extra['state']

            # allow infinit bootstrap

            if episode_step + 1 == env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
                done_bool = float(is_success)
                done_trajectory = 1
                done = True
            else: 
                done_bool = float(is_success)
                done_trajectory = float(is_success)

            if is_success:
                total_successes += 1
                #total
                done = True 

            episode_reward += reward * (discount ** episode_step)

            replay_buffer.add(obs, state, action, episode_reward, reward, next_obs, next_state, done_bool, done_trajectory, goal)

            for _ in range(details['grad_steps_per_online_step']):
                agent.update(replay_buffer, total_steps + add_to)

            obs = next_obs
            state = next_state
            episode_step += 1
            total_steps += 1

def collect_offline_replay(agent, env, discount, replay_buffer, sample_points = 100000):

    episode, episode_reward, done = 0, 0, True
    episode_step = 0
    ignore_first = True
    start_time = time.time()
    success = 0
    average_reward = 0
    total_steps = 0
    for step in range(sample_points):
        if done:
            print(success)
            total_steps += episode_step
            print(total_steps)
            if agent == 'demo':
                obs, goal, extra = env.demo_reset()
            else:
                obs, goal, extra = env.reset()

            state = extra['state']
            done = False
            average_reward += episode_reward
            episode += 1
            episode_reward = 0

            episode_step = 0
            reward = 0

        if agent == 'demo':
            action = env.get_demo_action()
        else:
            with eval_mode(agent):
                action = agent.select_action(obs, goal, init_obs=init_obs)

        curr_reward = reward
        next_obs, reward, is_success, extra = env.step(action)

        next_state = extra['state']

        if episode_step + 1 == env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
            done_bool = 1 #Not Sure what this should be?
            done_trajectory = 1
            done = True
        else: 
            done_bool = float(is_success)
            done_trajectory = float(is_success)

        if is_success:
            success += 1
            done = True  

        episode_reward += (discount ** episode_step) * reward

        replay_buffer.add(obs, state, action, episode_reward, reward, next_obs, next_state, done_bool, done_trajectory, goal)

        obs = next_obs
        if episode_step == 0:
            init_obs = obs
        state = next_state
        episode_step += 1

    print("Success_Rate of Offline Set: " , success/episode)
    print("Average Reward of Offline Set: ", average_reward/episode)

    return replay_buffer

def collect_analogy_offline_replay(agent, env, discount, analogy_replay_buffer, trajectories = 5000):

    episode, episode_reward, done = 0, 0, True
    episode_step = 0
    ignore_first = True
    start_time = time.time()
    success = 0
    average_reward = 0
    total_steps = 0
    for traj in range(trajectories):
        if agent == 'demo':
            obs, goal, extra = env.demo_reset()
        else:
            obs, goal, extra = env.reset()

        print(traj)
        total_steps += episode_step
        #print(total_steps)
        

        state = extra['state']
        done = False
        average_reward += episode_reward
        episode += 1
        episode_reward = 0

        episode_step = 0
        reward = 0

        while not done:
            

            if agent == 'demo':
                action = env.get_demo_action()
            else:
                with eval_mode(agent):
                    action = agent.select_action(obs, goal)

            curr_reward = reward
            next_obs, reward, is_success, extra = env.step(action)

            next_state = extra['state']

            if episode_step + 1 == env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
                done_bool = 1
                done_trajectory = 1
                done = True
            else: 
                done_bool = float(is_success)
                done_trajectory = float(is_success)

            if is_success:
                success += 1
                done = True  

            episode_reward += (discount ** episode_step) * reward

            analogy_replay_buffer.add(obs, action, reward, next_obs, done_bool, done_trajectory, goal)

            obs = next_obs
            state = next_state
            episode_step += 1

        obs, goal, extra = env.demo_jitter_reset()

        done = False
        episode_step = 0
        reward = 0

        while not done:

            if agent == 'demo':
                action = env.get_demo_action()
            else:
                with eval_mode(agent):
                    action = agent.select_action(obs, goal)

            curr_reward = reward
            next_obs, reward, is_success, extra = env.step(action)

            next_state = extra['state']

            if episode_step + 1 >= env._max_episode_steps: #Doesn't deal with case where goal is met on final step sadly
                done_bool = 1
                done_trajectory = 1
                done = True
            else: 
                done_bool = float(is_success)
                done_trajectory = float(is_success)

            if is_success:
                done = True  

            #episode_reward += (discount ** episode_step) * reward

            analogy_replay_buffer.add_analogy(obs, action, reward, next_obs, done_bool, done_trajectory, goal)

            obs = next_obs
            state = next_state
            episode_step += 1

    print("Success_Rate of Offline Set: " , success/episode)
    #print("Average Reward of Offline Set: ", average_reward/episode)

    return analogy_replay_buffer
