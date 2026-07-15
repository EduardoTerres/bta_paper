import random
import torch
import torch.nn as nn
from goalbisim.representation.encoders.RADencoder import PixelEncoder
from goalbisim.representation.pairedstategoal import PairedStateGoal
from rlkit.core import logger
import torch.nn.functional as F
import wandb
import math


class GoalBisim(nn.Module):
    def __init__(
            self,
            obs_shape,
            device,
            transition_model_type = 'ensemble',
            psi_loss_form = 'delta',
            disconnect_psi = False,
            using_phi = True,
            metric_loss = 'l1',
            metric_distance = 'reward',
            decoder_type = 'reward',
            dynamics_loss = 'direct',
            action_weight = 10,
            use_contrastive = False,
            contrastive_weight = 0.25,
            phi_updates_before_psi = 0,
            on_policy_dynamics = False,
            dual_optimization = False,
            disconnect_implict_policy = True,
            ground_space = True,
            decode_both = True,
            train_iters_per_update_psi = 1,
            train_iters_per_update_phi = 1,
            steps_till_on_policy = 4000,
            encoder_weight = 1,
            transition_weight = 1,
            action_shape = (1, 2),
            discount = 0.99,
            action_scale = 0.5,
            feature_dim = 256,
            num_layers = 4,
            num_filters = 32,
            lr=1e-3,
            weight_decay = 0,
            output_logits= True,
            output_logits_paired = True,
            num_layers_paired = 4,
            num_filters_paired = 32,
            lr_paired=1e-3,
            weight_decay_paired = 0,
            compositionality_weight = 0.0,
            compositionality_batch_size = 32):
        super().__init__()

        self.using_phi = using_phi
        self.device = device
        self.psi = PixelEncoder(obs_shape, feature_dim, num_layers, num_filters, output_logits = output_logits).to(self.device)
        self.encoder = self.psi
        self.feature_dim = feature_dim
        #self.device = device

        # Min/max/negation compositionality is enforced on psi directly (state-only, no
        # grounding) *and* on phi (state-conditioned, grounded -- see PairedStateGoal).
        # Re-enabled alongside phi's version; see compositionality_loss below for why
        # these are consistent rather than competing targets.
        self.compositionality_weight = compositionality_weight
        self.compositionality_batch_size = compositionality_batch_size
        self._comp_pool = None
        self._comp_pool_neg = None

        self.disconnect_psi = disconnect_psi

        self.phi_updates_before_psi = phi_updates_before_psi
        self.ground_space = ground_space

        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_weight = contrastive_weight
            self.cross_entropy = nn.CrossEntropyLoss()

        self.train_iters_per_update = train_iters_per_update_psi

        if self.using_phi:
            self.phi = PairedStateGoal(obs_shape, device, transition_model_type = transition_model_type, discount = discount, \
                metric_distance = metric_distance, decode_both = decode_both, decoder_type = decoder_type, dual_optimization = dual_optimization, \
                feature_dim = feature_dim, disconnect_implict_policy = disconnect_implict_policy, num_layers = num_layers_paired, dynamics_loss = dynamics_loss,\
                metric_loss = metric_loss, train_iters_per_update = train_iters_per_update_phi, num_filters = num_filters_paired, lr=lr_paired, \
                action_shape = action_shape, on_policy_dynamics = on_policy_dynamics, action_weight = action_weight, steps_till_on_policy = steps_till_on_policy, \
                action_scale = action_scale, output_logits = output_logits_paired, weight_decay = weight_decay_paired,  encoder_weight = encoder_weight, transition_weight = transition_weight, \
                ground_space = ground_space, compositionality_weight = compositionality_weight, compositionality_batch_size = compositionality_batch_size)
            try:
                self.psi_optimizer = torch.optim.AdamW(self.psi.parameters(), lr=lr, weight_decay=weight_decay)

            except:
                #If using a downgraded torch version
                raise NotImplementedError
                self.psi_optimizer = torch.optim.Adam(self.psi.parameters(), lr=lr, weight_decay=weight_decay)

            self.optimizer_step = torch.optim.lr_scheduler.StepLR(self.psi_optimizer, 1, gamma=0.95, last_epoch= -1, verbose=False)
            #self.psi_lr_fixer = torch.optim.lr_scheduler.ReduceLROnPlateau(self.psi_optimizer, mode = 'min', factor = 0.9, patience = 5, threshold = 0.0001)
            self.psi_loss_form = psi_loss_form #Make sure this doesnt take into account paired
            self.mse = nn.MSELoss()
            #self.train_iters_per_update = train_iters_per_update
            assert self.train_iters_per_update > 0
            self.discount = discount

    def forward(self, obs, detach=False):

        return self.encode(obs, detach = detach)

    def encode(self, obs, detach=False):

        z_out = self.psi(obs, detach = detach)

        return z_out

    def loss(self, obs, action, next_obs, goal, reward, policy, step, log = True, beginning = 'train', detach_paired = True):
        delta_z = self.phi(obs, goal) #Maybe there should be some cross talk???

        if self.ground_space:
            delta_z -= self.phi(goal, goal)

        if detach_paired:
            delta_z = delta_z.detach()

        state = self.encode(obs)
        goal = self.encode(goal)

        if 'detach_goal' in self.psi_loss_form:
            goal = goal.detach()

        norms = torch.norm(state.detach(), dim = 1)
        output_norm = torch.nn.functional.normalize(state.detach(), dim = 1)
        output_std = torch.std(output_norm, 0).mean().item()
        collapse_level = max(0., 1 - math.sqrt(self.feature_dim) * output_std)
        std_norm = torch.std(norms)

        stats = {'step' : step,
         beginning + '/psi/norm_std' : std_norm,
         beginning + '/psi/collapse_level' : collapse_level
        }

        logger.logging_tool.log(stats)

        if 'delta' in self.psi_loss_form:
            predicted_delta = goal - state
            delta_loss = self.mse(predicted_delta, delta_z)
        elif 'direct' in self.psi_loss_form:
            predicited_goal = state + delta_z
            delta_loss = self.mse(predicited_goal, goal)
        elif 'delta_contrast' in self.psi_loss_form:
            predicted_delta = goal - state
            delta_loss = self.temporal_contrastive_loss(predicted_delta, delta_z)
        elif 'direct_contrast' in self.psi_loss_form:
            predicited_goal = state + delta_z
            delta_loss = self.temporal_contrastive_loss(predicited_goal, goal)
        elif 'l2' in self.psi_loss_form:
            delta_norm = torch.norm(goal - state, dim = 1)
            delta_target = torch.norm(delta_z, dim = 1)
            delta_loss = ((delta_norm - delta_target) ** 2).mean()
        elif 'l1' == self.psi_loss_form:
            delta_norm = torch.norm(goal - state, dim = 1) #Should this be l2....
            delta_target = torch.norm(delta_z, ord = 1, dim = 1)
            delta_loss = ((delta_norm - delta_target) ** 2).mean()
        elif 'l1_pure' == self.psi_loss_form:
            delta_norm = torch.norm(goal - state, ord = 1, dim = 1) #Should this be l2....
            delta_target = torch.norm(delta_z, ord = 1, dim = 1)
            delta_loss = ((delta_norm - delta_target) ** 2).mean()        
        else:
            raise NotImplementedError        

        if self.use_contrastive:  
            contrastive_loss = self.contrastive_weight * self.temporal_contrastive_loss(obs, next_obs)    
            loss = delta_loss + contrastive_loss
            stats = {'step' : step,
            beginning + '/psi/delta_loss' : delta_loss.item(),
            beginning + '/psi/contrastive_loss' : contrastive_loss.item()
             } 
        else:
            loss = delta_loss
            stats = {'step' : step,
            beginning + '/psi/delta_loss' : delta_loss.item()
             } 

        logger.logging_tool.log(stats)

        return loss

    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.psi.W_contrast, z_pos.t())  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def temporal_contrastive_loss(self, obs_anchor, obs_positive):
        z_a = self.encode(obs_anchor)
        z_pos = self.encode(obs_positive)
        
        logits = self.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy(logits, labels)

        return loss

    def set_compositionality_pool(self, minmax_pool, neg_pool = None):
        """Forwards both pools to phi (state-conditioned, grounded -- see
        PairedStateGoal.set_compositionality_pool) *and* keeps copies for psi's own
        direct, unconditioned versions below (compositionality_loss / _negation)."""
        self._comp_pool = {
            n: (members.to(self.device), union.to(self.device), inter.to(self.device))
            for n, (members, union, inter) in minmax_pool.items()
        }
        self._comp_pool_neg = {
            k: (goal.to(self.device), not_goal.to(self.device))
            for k, (goal, not_goal) in neg_pool.items()
        } if neg_pool else None
        self.phi.set_compositionality_pool(minmax_pool, neg_pool)

    def compositionality_loss(self, members, union_img, inter_img):
        batch, n = members.shape[0], members.shape[1]
        flat_members = members.reshape(batch * n, *members.shape[2:])
        z_members = self.encode(flat_members).reshape(batch, n, -1)
        z_union_target = z_members.max(dim=1).values
        z_inter_target = z_members.min(dim=1).values

        z_union = self.encode(union_img)
        z_inter = self.encode(inter_img)

        union_loss = self.mse(z_union, z_union_target)
        inter_loss = self.mse(z_inter, z_inter_target)
        return inter_loss + union_loss

    def compositionality_loss_negation(self, goal_img, not_goal_img):
        z_goal = self.encode(goal_img)
        z_not_goal = self.encode(not_goal_img)
        return self.mse(z_not_goal, 1 - z_goal)

    def train_batch(self, obs, action, next_obs, goal, reward, policy, step, log = True, take_step = True, beginning = 'train'):

        #self.model.train()
        loss = self.loss(obs, action, next_obs, goal, reward, policy, step, log = log, beginning = beginning)

        if self.compositionality_weight > 0 and self._comp_pool:
            n = random.choice(list(self._comp_pool.keys()))
            members, union_img, inter_img = self._comp_pool[n]
            pool_size = members.shape[0]
            idx = torch.randint(0, pool_size, (min(self.compositionality_batch_size, pool_size),), device=self.device)
            comp_loss = self.compositionality_loss(members[idx], union_img[idx], inter_img[idx])
            loss = loss + self.compositionality_weight * comp_loss

            logger.logging_tool.log({
                'step': step,
                beginning + '/psi/compositionality_loss': comp_loss.item(),
                beginning + '/psi/compositionality_n': n,
            })

        if self.compositionality_weight > 0 and self._comp_pool_neg:
            k = random.choice(list(self._comp_pool_neg.keys()))
            goal_img, not_goal_img = self._comp_pool_neg[k]
            pool_size = goal_img.shape[0]
            idx = torch.randint(0, pool_size, (min(self.compositionality_batch_size, pool_size),), device=self.device)
            neg_loss = self.compositionality_loss_negation(goal_img[idx], not_goal_img[idx])
            loss = loss + self.compositionality_weight * neg_loss

            logger.logging_tool.log({
                'step': step,
                beginning + '/psi/compositionality_negation_loss': neg_loss.item(),
                beginning + '/psi/compositionality_k': k,
            })

        stats = {'step' : step,
        beginning + '/psi/loss' : loss.item()
        }

        logger.logging_tool.log(stats)

        if take_step:
            assert beginning == 'train'
            self.psi_optimizer.zero_grad()
            loss.backward()
            self.psi_optimizer.step()
        else:
            val_loss = loss.detach()
            #self.psi_lr_fixer.step(val_loss)

    def step_lr(self):
        self.optimizer_step.step()
        self.phi.step_lr()
        #self.decoder_optimizer_step.step()

    def eval_loss(self, replay_buffer, policy, kwargs, step, log = True):
        self.train_batch(kwargs['obs'], kwargs['action'], kwargs['next_obs'], kwargs['goal'], kwargs['reward'], \
            policy, step, log = log, take_step = False, beginning = 'eval')

        self.phi.eval_loss(replay_buffer, policy, kwargs, step, log = True)

    def update(self, replay_buffer, policy, kwargs, step, log = True):
        #Will run through dataset...
        #Should we do multiple times? Could possibly resample, but encoder is quite small to begin with

        

        assert self.using_phi, "Should not be updating this if you are just using the psi encoder!"
        self.phi.update(replay_buffer, policy, kwargs, step, log = log)

        self.phi_updates_before_psi -= 1

        if self.phi_updates_before_psi <= 0:
            if not self.disconnect_psi:
                self.train_batch(kwargs['obs'], kwargs['action'], kwargs['next_obs'], kwargs['goal'], kwargs['reward'], policy, step, log = log)
                for _ in range(self.train_iters_per_update - 1):
                    obs, action, _, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample()
                    self.train_batch(obs, action, next_obs, goals, reward, policy, step, log = log)









