import goalbisim.utils.hyperparameter as hyp
from goalbisim.trainers.offline_trainers import train_representation_offline
import numpy as np
import os
os.environ["MUJOCO_GL"] = 'osmesa'

img_size = 64
if __name__ == "__main__":
    variant = dict(
        training_form = 'dataset',
        #dataset_loc = '/home/philippe/RAIL/data/replay_sawyer_v4_easy.pt',
        #dataset_loc = '/global/scratch/users/hansenpmeche/RAIL/data/replay_sawyer_v1_color_96464.pt',
        #dataset_loc = '/global/scratch/users/hansenpmeche/RAIL/data/replay_sawyer_v5_easy_yaw.pt',
        discount=0.99,
        env_kwargs = dict(
            package = 'roboverse',
            domain_name = 'sawyer_rig_v5',
            domain_kwargs = dict(
                expl = False,
                random_color_p=1,
                max_episode_steps = 75,
                obs_img_dim = 64,
                claw_spawn_mode='fixed',
                drawer_yaw_setting = (0, 360),
                color_range = (0, 255),
                drawer_bounding_x = [.46, .84],
                drawer_bounding_y = [-.19, .19],
                yaw_jitter_var = .1,
                translation_jitter_var = [0.08, 0.08],
                view_distance = 0.55,
                max_distractors = 4,
                test_env = True,
                ),
            frame_stack_count = 1,
            action_repeat = 1,
            ),

        replay_buffer_type = 'GoalAnalogy',
        replay_buffer_kwargs = dict(
            capacity = 200000,
            batch_size = 256,
            #num_goals=1
            ),

        representation_algorithm = 'GoalBiSim',
        goalbisim_kwargs = dict(
            transition_model_type = 'ensemble',
            psi_loss_form = 'delta',
            metric_distance = 'reward',
            decoder_type = 'reward',
            on_policy_dynamics = False,
            use_contrastive = False,
            contrastive_weight = 0.25,
            action_weight = 4,
            train_iters_per_update_psi = 1,
            train_iters_per_update_phi = 1,
            discount = 0.99,
            feature_dim = 256,
            num_layers = 6,
            num_filters = 32,
            metric_loss = 'l1',
            lr=1e-3,
            weight_decay = 0,
            num_layers_paired = 6,
            num_filters_paired = 32,
            lr_paired=1e-3,
            weight_decay_paired = 0
        ),


        relabel_strategy_kwargs = dict(
            dense = False,
            use_state = False,
            eps = 1.6,
            reward_scaling = 1
            ),

        training_transforms = [],
        eval_transforms = [],

        iql_kwargs = dict(
        discount=0.99,
        actor_lr=1e-4,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        critic_lr=1e-4,
        critic_beta=0.9,
        critic_tau=0.005,
        encoder_tau=0.01,
        quantile=0.7,
        policy_update_period=1,
        q_update_period=1,
        target_update_period=1,
        clip_score=100,
        soft_target_tau=0.005,
        beta=1.0/3,
        detach_encoder = False,
        detach_conv = False,
        use_adamw = True,
        analogy_goal = True
        ),

        use_distractor = True,

        distractor_kwargs = dict(
            video_format = 'stitch_mp4',
            pixels_to_cut = [[255, 255, 255]], #White Pixel Background
            augmentation = 'identity',
            dataset_loc = '/videos_compressed',
            replay_kwargs = dict(
                replay_buffer_type = 'Goal',
                replay_buffer_kwargs = dict(
                capacity = 200000,
                batch_size = 256
            ),
            ),
        ),

        rl_algorithm = 'IQL',

        training_iterations = 20005,

        online_training_trajectories = 0,

        num_eval_episodes = 5,

        pre_eval_freq = 0,

        eval_video_save_dir = "",

        eval_analogy_save_dir = "",

        eval_freq = 4000, #5 Epochs Each

        eval_traj_freq = 0,

        tests = [],

        use_wandb = True,

        reload_best_agent = False,

        offline_wandb = False,

        save_model = False,

        analogy_goal = True, 

        save_wandb_video = False,

        #launcher_config=dict(slurm_config=dict(slurm_config_name="gpulowprio")),

        #slurm_config_name = 'gpu_v100',

        device = 'cuda',

        project_name = "gcab",

        entity = "gcab",



    )

    search_space = {
        "seed": [1] * 1,
        "psi_loss_form" : ['direct'],
        "number_training_points" : [50000],
        "phi_config" : ['psi'],
        "claw_spawn_mode" : ['fixed'],
        "add_her_relabels" : [False],
        "train_iters_per_update_psi" : [1],
        "train_iters_per_update_phi" : [1],
        "representation_pre_training_iterations" : [0],
        "policy_pre_training_iterations" : [0],
        "dual_optimization" : [False],
        "steps_till_on_policy" : [0],
        "allow_pre_critic_gradients" : [False],
        "reload_best_agent" : [False],
        "step_lr" : [False],
        "dynamics_loss" : ['delta'],
        "metric_loss" : ['l1'],
        "contrastive_weight" : [0.25],
        "use_distractor" : [False],
        "dataset_loc" : ['replay_sawyer_analogy.pt'],
        "drawer_yaw_setting" : [(0, 360)],
        "use_contrastive" : [False],
        "ground_space" : [True],
        "weight_decay" : [5e-4],
        "decode_both" : [True],
        "action_weight" : [25],
        "phi_updates_before_psi" : [0],
        "lr" : [1e-4],
        "lr_paired" : [1e-4],
        "disconnect_psi" : [False],
        "batch_size" : [256],
        "metric_distance" : ['reward'], #Global Structure Not Needed
        "decoder_type" : ['reward'],
        "transition_model_type" : ['next_observation'],
        "on_policy_dynamics" : [''], #Unclear
        "feature_dim" : [128],
        "detach_encoder" : [True], #False -> PixelIQL 
        "detach_conv" : [False]
    }



    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    count = 0
    for variant in sweeper.iterate_hyperparameters():

        variant['seed'] = np.random.randint(99999)

        variant['env_kwargs']['domain_kwargs']['claw_spawn_mode'] = variant['claw_spawn_mode']

        variant['iql_kwargs']['analogy_goal'] = variant['analogy_goal']

        if variant['drawer_yaw_setting'] == (0, 360) and 'yaw' in variant['dataset_loc']:
            continue

        if variant['drawer_yaw_setting'] == (0, 90, 270, 360) and 'base' in variant['dataset_loc']:
            continue

        variant['env_kwargs']['domain_kwargs']['drawer_yaw_setting'] = variant['drawer_yaw_setting']
        #variant['experiment_name'] = "offline_gcab_distractor" 
        
        if variant['decoder_type'] == 'action' and variant['metric_distance'] == 'reward':
            continue
        
        if variant['metric_distance'] == 'action':
            variant['group'] = 'GCAB_'

        if variant['metric_distance'] == 'reward':
            variant['group'] = 'GCRB_'

        if variant['decoder_type'] == 'action':
            variant['group'] += '_AD'

        if variant['decoder_type'] == 'reward':
            variant['group'] += '_RD'

        if variant['transition_model_type'] == 'next_observation':
            variant['group'] += '_NO'

        if variant['transition_model_type'] == 'ensemble':
            variant['group'] += '_E'

        if variant['on_policy_dynamics'] == 'deterministic':
            variant['group'] += '_OP'

        if not variant['detach_encoder']:
            variant['group'] += '_CG'

        if 'base' in variant['dataset_loc']:
            variant['group'] += '_' + 'base'

        if 'rot' in variant['dataset_loc']:
            variant['group'] += '_' + 'rot'

        if variant['use_distractor']:
            variant['group'] += '_' + 'distractor'

        variant['group'] += variant['phi_config']

        variant['group'] += '_analogy_goal_v3'

        if variant['on_policy_dynamics'] == 'deterministic' and 'next_observation' in variant['transition_model_type']:
            continue

        if variant['metric_loss'] == 'l2' and 'next_observation_l1' in variant['transition_model_type']:
            continue

        if variant['metric_distance'] == 'action' and variant['action_weight'] == 0.25:
            continue

        if variant['metric_distance'] == 'reward' and variant['action_weight'] == 0.25:
            continue

        if variant['metric_distance'] == 'temporal' and variant['action_weight'] == 12:
            continue

        if variant['detach_encoder'] and variant['detach_conv']:
            continue

        if variant['metric_distance'] == 'reward' and variant['decoder_type'] == 'action':
            continue

        if variant['metric_distance'] == 'reward' and variant['on_policy_dynamics'] == 'deterministic':
            continue

        if variant['replay_buffer_type'] == 'Goal' and variant['add_her_relabels']:
            continue

        if variant['replay_buffer_type'] == 'HER':
            variant['replay_buffer_kwargs']['num_goals'] = 1

        variant['replay_buffer_kwargs']['batch_size'] = variant['batch_size']

        variant['iql_kwargs']['detach_encoder'] = variant['detach_encoder']
        variant['iql_kwargs']['detach_conv'] = variant['detach_conv']
        variant['goalbisim_kwargs']['psi_loss_form'] = variant['psi_loss_form']
        variant['goalbisim_kwargs']['train_iters_per_update_psi'] = variant['train_iters_per_update_psi']
        variant['goalbisim_kwargs']['train_iters_per_update_phi'] = variant['train_iters_per_update_phi']
        variant['goalbisim_kwargs']['lr'] = variant['lr']
        variant['goalbisim_kwargs']['lr_paired'] = variant['lr_paired']
        variant['goalbisim_kwargs']['dual_optimization'] = variant['dual_optimization']
        variant['goalbisim_kwargs']['weight_decay'] = variant['weight_decay']
        variant['goalbisim_kwargs']['weight_decay_paired'] = variant['weight_decay']
        variant['goalbisim_kwargs']['steps_till_on_policy'] = variant['steps_till_on_policy']
        variant['goalbisim_kwargs']['dynamics_loss'] = variant['dynamics_loss']
        variant['goalbisim_kwargs']['ground_space'] = variant['ground_space']
        variant['goalbisim_kwargs']['metric_distance'] = variant['metric_distance']
        variant['goalbisim_kwargs']['decoder_type'] = variant['decoder_type']
        variant['goalbisim_kwargs']['transition_model_type'] = variant['transition_model_type']
        variant['goalbisim_kwargs']['on_policy_dynamics'] = variant['on_policy_dynamics']
        variant['goalbisim_kwargs']['action_weight'] = variant['action_weight']
        variant['goalbisim_kwargs']['feature_dim'] = variant['feature_dim']
        variant['goalbisim_kwargs']['metric_loss'] = variant['metric_loss']
        variant['goalbisim_kwargs']['decode_both'] = variant['decode_both']
        variant['goalbisim_kwargs']['phi_updates_before_psi'] = variant['phi_updates_before_psi']
        variant['goalbisim_kwargs']['use_contrastive'] = variant['use_contrastive']
        variant['goalbisim_kwargs']['contrastive_weight'] = variant['contrastive_weight']
        variant['goalbisim_kwargs']['disconnect_psi'] = variant['disconnect_psi']


        variants.append(variant)

    #run_variants(train_representation_offline, variants, run_id='offline_phi_psi_training')
