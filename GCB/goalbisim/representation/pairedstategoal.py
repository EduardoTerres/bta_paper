import random
import torch
import torch.nn as nn
from goalbisim.representation.encoders.RADencoder import PixelEncoder
from rlkit.core import logger
import torch.nn.functional as F
from goalbisim.dynamics.dynamics_models import make_transition_model
import numpy as np
import wandb
import math


class PairedStateGoal(nn.Module):
    def __init__(
            self,
            obs_shape,
            device,
            transition_model_type = 'ensemble',
            metric_loss = 'l1',
            metric_distance = 'reward',
            decoder_type = 'reward',
            dynamics_loss = 'direct',
            dual_optimization = False,
            action_weight = 1,
            on_policy_dynamics = False, #Might be needed when performing offline RL
            decode_both = False,
            disconnect_implict_policy = True,
            train_iters_per_update = 1,
            action_shape = (5, 1), #We will need to fix at somepoint....
            action_scale = 1, #to clip actions properly...
            discount = 0.99,
            steps_till_on_policy = 3000,
            encoder_weight = 1,
            transition_weight = 1,
            feature_dim = 256,
            num_layers = 4,
            num_filters = 32,
            output_logits = True,
            lr=1e-3,
            weight_decay = 0,
            ground_space = True,
            lambda_comp = 1.0,
            beta_comp = 1.0,
            lambda_comp_warmup_steps = 10000):
        super().__init__()

        self.device = device
        self.encoder = PixelEncoder(obs_shape, feature_dim, num_layers, num_filters, output_logits = output_logits, goal_flag = True).to(self.device)
        self.phi = self

        self.ground_space = ground_space

        # Boolean-algebra compositionality loss on phi (Nangue Tasse et al. 2020), enforced
        # over Phi, a permutation-invariant set extension of phi -- see encode_set /
        # compositionality_loss below.
        self.lambda_comp = lambda_comp
        self.beta_comp = beta_comp
        self.lambda_comp_warmup_steps = lambda_comp_warmup_steps
        self.comp_pool_size_range = (2, 8)

        self.set_pool_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)).to(self.device)
        # Zero-init the last layer so Phi starts as a plain mean-pool (near-identity residual).
        nn.init.zeros_(self.set_pool_mlp[-1].weight)
        nn.init.zeros_(self.set_pool_mlp[-1].bias)

        # Learned empty-set token: an image-shaped parameter fed through the same phi trunk
        # as any other goal, so Phi(s, empty) and Gamma(empty) reuse the trunk exactly like a
        # singleton goal would (sigmoid keeps it in the [0,1] range the encoder asserts on).
        self.empty_token = nn.Parameter(torch.zeros(obs_shape, device=self.device))

        self.action_scale = action_scale
        self.decode_both = decode_both
        self.metric_loss = metric_loss
        self.psi = self
        self.action_weight = action_weight
        self.metric_distance = metric_distance
        self.decoder_type = decoder_type
        self.on_policy_dynamics = on_policy_dynamics
        self.feature_dim = feature_dim
        self.transition_model_type = transition_model_type
        self.train_iters_per_update = train_iters_per_update
        self.encoder_weight = encoder_weight
        self.transition_weight = transition_weight
        self.disconnect_implict_policy = disconnect_implict_policy
        self.steps_till_on_policy = steps_till_on_policy
        self.dynamics_loss = dynamics_loss
        self.dual_optimization = dual_optimization
        self.cross_entropy = nn.CrossEntropyLoss()

        if self.decode_both:
            scale = 2
        else:
            scale = 1

        if self.decoder_type == 'reward' or self.decoder_type == 'rtg' or self.decoder_type == 'temporal' or self.decoder_type == 'none':
            self.decoder = nn.Sequential( #Should be action decoder next
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(self.device)
        elif self.decoder_type == 'rtg_reward':
            self.decoder = nn.Sequential( #Should be action decoder next
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 2)).to(self.device)
        elif self.decoder_type == 'temporal_action':
            self.decoder = nn.Sequential( #Should be action decoder next
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1 + action_shape[0])).to(self.device)           
        elif self.decoder_type == 'action':
            self.decoder = nn.Sequential( #Should be action decoder next
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, action_shape[0])).to(self.device)
        else:
            raise NotImplementedError



        if self.transition_model_type != 'next_observation' and self.transition_model_type != 'next_observation_l1':
            if self.transition_model_type == 'ensemble_proper':
                input_text = 'ensemble'
            elif self.transition_model_type == 'next_observation_ensemble':
                input_text = 'ensemble'
            else:
                input_text = self.transition_model_type

            self.dynamics_model = make_transition_model(input_text, feature_dim, action_shape).to(self.device)
        else:
            self.dynamics_model = make_transition_model('', feature_dim, action_shape).to(self.device)

        
        try:
            self.encoder_optimizer = torch.optim.AdamW(
                list(self.encoder.parameters()) + list(self.set_pool_mlp.parameters()) + [self.empty_token],
                lr=lr, weight_decay=weight_decay)
            self.decoder_optimizer = torch.optim.AdamW(list(self.decoder.parameters()) + list(self.dynamics_model.parameters()), lr=lr, weight_decay=weight_decay)
        except:
            raise NotImplementedError
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
            self.decoder_optimizer = torch.optim.Adam(list(self.decoder.parameters()) + list(self.dynamics_model.parameters()), lr=lr, weight_decay=weight_decay)

        self.encoder_optimizer_step = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, 1, gamma=0.9, last_epoch= -1, verbose=False)
        self.decoder_optimizer_step = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, 1, gamma=0.9, last_epoch= -1, verbose=False)

        self.discount = discount
        self.mse = nn.MSELoss()



    def forward(self, obs, goal = None, detach=False):
        if goal is not None:
            obs = torch.cat([obs, goal], dim = 1) 
        return self.encode(obs, detach = detach)

    def encode(self, obs, goal = None, detach=False):
        if goal is not None:
            obs = torch.cat([obs, goal], dim = 1) 

        z_out = self.encoder(obs, detach = detach)

        return z_out

    def sample_compositionality_pool(self, replay_buffer):
        """Draws one pool P of n ~ Uniform{2,...,8} goal images from the replay buffer, to
        be held fixed across the whole training batch (Phi_U / Phi_empty are pool-relative,
        so mixing pools within a batch would make the bounds hinge incoherent)."""
        n = random.randint(*self.comp_pool_size_range)
        return replay_buffer.sample_goal_pool(n).to(self.device)

    def _pool_member_embeddings(self, state, pool_imgs):
        """Raw (ungrounded, gradient-flowing) phi(s, g) for every g in the shared pool.
        state: (B, C, H, W); pool_imgs: (n, C, H, W). Returns (B, n, d)."""
        batch, n = state.shape[0], pool_imgs.shape[0]
        state_rep = state.unsqueeze(1).expand(-1, n, *state.shape[1:]).reshape(batch * n, *state.shape[1:])
        goal_rep = pool_imgs.unsqueeze(0).expand(batch, -1, *pool_imgs.shape[1:]).reshape(batch * n, *pool_imgs.shape[1:])
        return self.encode(state_rep, goal_rep).reshape(batch, n, -1)

    def _pool_grounding(self, pool_imgs):
        """phi_bar(g, g) for every g in the pool -- always detached. (n, d)."""
        with torch.no_grad():
            return self.encode(pool_imgs, pool_imgs)

    def _aggregate(self, z_members, mask):
        """Phi(s, A) via mean-pool + residual MLP (DeepSets-style), given raw per-member
        embeddings z_members (B, n, d) and a boolean membership mask (B, n) with >=1 True
        per row. Exactly identity-on-singletons: singleton rows bypass the MLP entirely, so
        Phi(s, {g}) == phi(s, g) holds numerically regardless of how set_pool_mlp is trained."""
        mask_f = mask.float().unsqueeze(-1)
        counts = mask_f.sum(dim=1).clamp(min=1)
        mean = (z_members * mask_f).sum(dim=1) / counts
        pooled = mean + self.set_pool_mlp(mean)
        is_singleton = (mask.sum(dim=1) == 1).unsqueeze(-1)
        return torch.where(is_singleton, mean, pooled)

    def _aggregate_grounding(self, ground_members, mask):
        """Gamma(A) = mean_{g in A} phi_bar(g, g). ground_members: (n, d), mask: (B, n)."""
        mask_f = mask.float().unsqueeze(-1)
        counts = mask_f.sum(dim=1).clamp(min=1)
        ground_rep = ground_members.unsqueeze(0).expand(mask.shape[0], -1, -1)
        return (ground_rep * mask_f).sum(dim=1) / counts

    def _phi_tilde(self, z_members, ground_members, mask):
        """Phi~(s, A) = Phi(s, A) - Gamma(A), the grounded set embedding used everywhere
        below (raw Phi has no canonical origin -- see L_psi, which grounds the same way)."""
        return self._aggregate(z_members, mask) - self._aggregate_grounding(ground_members, mask)

    def _empty_embedding(self, state):
        """Phi(s, empty) -- the learned empty-set token, encoded through the same trunk as
        any singleton goal."""
        token_img = torch.sigmoid(self.empty_token).unsqueeze(0).expand(state.shape[0], *self.empty_token.shape)
        return self.encode(state, token_img)

    def _empty_grounding(self, batch_size):
        """Gamma(empty) = phi_bar(empty_token, empty_token) -- always detached."""
        token_img = torch.sigmoid(self.empty_token).unsqueeze(0)
        with torch.no_grad():
            z = self.encode(token_img, token_img)
        return z.expand(batch_size, -1)

    def _phi_tilde_empty(self, state):
        return self._empty_embedding(state) - self._empty_grounding(state.shape[0])

    def _sample_overlapping_subsets(self, batch_size, n):
        """Sample A, B subseteq {0,...,n-1} (as boolean masks) with A ∩ B != {} guaranteed
        by construction: a shared core (>=1 element) plus disjoint private extras. Sampling
        A and B independently would almost surely yield disjoint sets for small pools, which
        degenerates the intersection term into a no-op."""
        idx_all = np.arange(n)
        mask_a = np.zeros((batch_size, n), dtype=bool)
        mask_b = np.zeros((batch_size, n), dtype=bool)
        for i in range(batch_size):
            core_size = np.random.randint(1, n + 1)
            core = np.random.choice(idx_all, size=core_size, replace=False)
            remaining = np.setdiff1d(idx_all, core)
            assign = np.random.randint(0, 3, size=remaining.shape[0])  # 0=neither, 1=A-only, 2=B-only
            mask_a[i, core] = True
            mask_a[i, remaining[assign == 1]] = True
            mask_b[i, core] = True
            mask_b[i, remaining[assign == 2]] = True
        return (torch.as_tensor(mask_a, device=self.device),
                torch.as_tensor(mask_b, device=self.device))

    def _sq_norm_mean(self, x):
        return x.pow(2).sum(dim=-1).mean()

    def compositionality_loss(self, state, pool_imgs, step, log = True, beginning = 'train'):
        """Boolean-algebra loss on Phi~ (Nangue Tasse et al. 2020): union/intersection via
        coordinatewise max/min, negation via the sum-minus-target form, plus a bounds hinge
        enforcing Phi~_empty <= Phi~(s,A) <= Phi~_U. Every target is detached; gradient only
        flows into the composite (LHS) branch -- see module docstring notes at call site."""
        n, batch = pool_imgs.shape[0], state.shape[0]
        mask_a, mask_b = self._sample_overlapping_subsets(batch, n)
        mask_u = torch.ones_like(mask_a)

        # Phi~(s, X) for every goal set X we need, each aggregated over the *same* shared
        # pool via the membership masks above -- this is what makes the terms below directly
        # comparable (they all live in the same pool-relative, grounded embedding space).
        z_members = self._pool_member_embeddings(state, pool_imgs)
        ground_members = self._pool_grounding(pool_imgs)

        phi_AuB = self._phi_tilde(z_members, ground_members, mask_a | mask_b)   # Phi~(s, A ∪ B), composite (gradient-flowing) branch
        phi_AnB = self._phi_tilde(z_members, ground_members, mask_a & mask_b)   # Phi~(s, A ∩ B), composite branch
        phi_notA = self._phi_tilde(z_members, ground_members, ~mask_a)          # Phi~(s, ¬A) = Phi~(s, pool \ A), composite branch
        phi_U = self._phi_tilde(z_members, ground_members, mask_u)              # Phi~(s, U), the full pool as the "universe" set
        phi_empty = self._phi_tilde_empty(state)                                # Phi~(s, ∅) via the learned empty-set token

        # Targets (RHS of each Boolean identity) are always computed from detached
        # embeddings of the *primitive* sets A, B, U, ∅ -- only the composite set on the
        # LHS above gets gradient, so these losses shape how A ∪ B / A ∩ B / ¬A relate to
        # their operands without also dragging the operands themselves around.
        z_members_det = z_members.detach()
        phi_A_det = self._phi_tilde(z_members_det, ground_members, mask_a)      # Phi~(s, A), target-side
        phi_B_det = self._phi_tilde(z_members_det, ground_members, mask_b)      # Phi~(s, B), target-side
        phi_U_det = phi_U.detach()
        phi_empty_det = phi_empty.detach()

        # Coordinatewise max/min over sampled A, B enforce the lattice-join/meet identities
        # for union/intersection (Nangue Tasse et al. 2020): Phi~(A∪B) == max(Phi~A, Phi~B),
        # Phi~(A∩B) == min(Phi~A, Phi~B).
        union_term = self._sq_norm_mean(phi_AuB - torch.maximum(phi_A_det, phi_B_det))
        inter_term = self._sq_norm_mean(phi_AnB - torch.minimum(phi_A_det, phi_B_det))
        # Complement identity: Phi~(¬A) == Phi~(U) + Phi~(∅) - Phi~(A), sampled once per A.
        negation_term = self._sq_norm_mean(phi_notA - (phi_U_det + phi_empty_det - phi_A_det))

        # Bounds hinge: for every sampled A, Phi~(A) must sit between the empty-set and
        # universal-set embeddings (Phi~_empty <= Phi~(s,A) <= Phi~_U); only violations
        # (positive relu) are penalized.
        hinge_upper = F.relu(phi_A_det - phi_U)
        hinge_lower = F.relu(phi_empty - phi_A_det)
        bounds_hinge_term = self._sq_norm_mean(hinge_upper) + self._sq_norm_mean(hinge_lower)

        comp_loss = union_term + inter_term + negation_term + self.beta_comp * bounds_hinge_term
        lambda_current = self.lambda_comp * min(1.0, step / max(1, self.lambda_comp_warmup_steps))

        if log:
            with torch.no_grad():
                spread = (phi_A_det - phi_B_det).norm(dim=-1).mean()
                goal_point_var = ground_members.var(dim=0).mean()
                bound_violation_frac = ((phi_A_det < phi_empty) | (phi_A_det > phi_U)).float().mean()
            logger.logging_tool.log({
                'step': step,
                beginning + '/comp/union': union_term.item(),
                beginning + '/comp/intersect': inter_term.item(),
                beginning + '/comp/negation': negation_term.item(),
                beginning + '/comp/bounds_hinge': bounds_hinge_term.item(),
                beginning + '/comp/lambda_current': lambda_current,
                beginning + '/comp/spread': spread.item(),
                beginning + '/comp/goal_point_var': goal_point_var.item(),
                beginning + '/comp/bound_violation_frac': bound_violation_frac.item(),
                beginning + '/comp/norm_Phi_U': phi_U_det.norm(dim=-1).mean().item(),
                beginning + '/comp/norm_Phi_empty': phi_empty_det.norm(dim=-1).mean().item(),
                beginning + '/comp/norm_Phi_A': phi_A_det.norm(dim=-1).mean().item(),
            })

        return comp_loss, lambda_current

    def encoder_loss(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, log = True, beginning = 'train'):
        if self.on_policy_dynamics == 'probabilistic' and step > self.steps_till_on_policy:
            action = policy.sample_action(obs, goal, batched = True)
        elif self.on_policy_dynamics == 'deterministic' and step > self.steps_till_on_policy:
            action = policy.select_action(obs, goal, batched = True)

        z = self.encode(obs, goal)
        perm = np.random.permutation(obs.shape[0])
        z_pair = z[perm]
        reward_pair = reward[perm]

        norms = torch.norm(z.detach(), p=1, dim = 1)
        output_norm = torch.nn.functional.normalize(z.detach(), p = 1, dim = 1)
        output_std = torch.std(output_norm, 0).mean().item()
        collapse_level = max(0., 1 - math.sqrt(self.feature_dim) * output_std)
        std_norm = torch.std(norms).detach().item()


        if self.transition_model_type != 'next_observation' and self.transition_model_type != 'next_observation_l1':
            with torch.no_grad():
                pred_next_latent_mu1, pred_next_latent_sigma1 = self.dynamics_model(torch.cat([z, action], dim=1))
                if self.dynamics_loss == 'delta':
                    pred_next_latent_mu1 += z
            if pred_next_latent_sigma1 is None:
                pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
            if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
                pred_next_latent_mu2 = pred_next_latent_mu1[perm]
                pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
            elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
                pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
                pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
            else:
                raise NotImplementedError

        if self.metric_loss == 'l1':
            z_dist = torch.norm(z - z_pair, p = 1, dim = 1)
        elif self.metric_loss == 'l2':
            z_dist = torch.norm(z - z_pair, dim = 1)
        else:
            raise NotImplementedError


        if self.metric_distance == 'reward':
            metric = F.smooth_l1_loss(reward, reward_pair, reduction='none').squeeze() * self.action_weight
        elif self.metric_distance == 'action':
            metric = (torch.norm(action - action[perm], dim = 1) * self.action_weight).squeeze()
        elif self.metric_distance == 'temporal':
            td_pair = td[perm]
            metric = torch.norm(td - td_pair, dim = 1).squeeze() * self.action_weight
        elif self.metric_distance == 'advantage_target':
            q1, q2 = policy.critic_target(obs, goal, action)
            vs = policy.critic_target.forward_v(obs, goal).detach()
            adv = q1.detach() - vs
            adv_pair = adv[perm]
            metric = torch.norm(adv - adv_pair, dim = 1).squeeze() * self.action_weight
        elif self.metric_distance == 'advantage':
            q1, q2 = policy.critic(obs, goal, action)
            vs = policy.critic.forward_v(obs, goal).detach()
            adv = q1.detach() - vs
            adv_pair = adv[perm]
            metric = torch.norm(adv - adv_pair, dim = 1).squeeze() * self.action_weight
        else:
            raise NotImplementedError

        if self.transition_model_type == 'next_observation':
            z_next = self.encode(next_obs, goal)
            z_next_pair = z_next[perm]
            transition_dist = torch.norm(z_next - z_next_pair, dim = 1)
        elif self.transition_model_type == 'next_observation_l1':
            z_next = self.encode(next_obs, goal)
            z_next_pair = z_next[perm]
            transition_dist = torch.norm(z_next - z_next_pair, p = 1, dim = 1)
        elif self.transition_model_type == 'deterministic':
            transition_dist = torch.norm(pred_next_latent_mu1 - pred_next_latent_mu2, dim = 1)
        elif self.transition_model_type == 'probabilistic':
            transition_dist = torch.sqrt(torch.norm(pred_next_latent_mu1 - pred_next_latent_mu2, dim = 1).pow(2) + torch.norm(pred_next_latent_sigma1 - pred_next_latent_sigma2, dim = 1).pow(2))
        elif self.transition_model_type == 'ensemble':
            transition_dist = torch.sqrt(torch.norm(pred_next_latent_mu1 - pred_next_latent_mu2, dim = 2).pow(2) + torch.norm(pred_next_latent_sigma1 - pred_next_latent_sigma2, dim = 2).pow(2))
            #transition_dist = transition_dist.unsqueeze(2)
        else:
            raise NotImplementedError

        bisimilarity = metric + self.discount * transition_dist
        if not self.dual_optimization:
            bisimilarity = bisimilarity.detach()

        loss = (z_dist - bisimilarity).pow(2).mean()

        return loss, std_norm, collapse_level

    def policy_decoder(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = 'train'):
        if self.on_policy_dynamics == 'probabilistic' and step > self.steps_till_on_policy:
            action = policy.sample_action(obs, goal, batched = True)
        elif self.on_policy_dynamics == 'deterministic' and step > self.steps_till_on_policy:
            action = policy.select_action(obs, goal, batched = True)

        z = self.encode(obs, goal).detach()
        pred_action = self.policy_decoder(z) #Inverse Model
        policy_decoder_loss = F.mse_loss(pred_action.squeeze(), action.squeeze())

        return policy_decoder_loss

    def transition_loss(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = 'train'):
        if self.transition_model_type == 'next_observation' or self.transition_model_type == 'next_observation_l1':
            return torch.Tensor([0]).to(self.device)

        z = self.encode(obs, goal)
        dyn_input = z

        pred_next_latent_mu, pred_next_latent_sigma = self.dynamics_model(torch.cat([dyn_input, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        pred_next_latent_sigma_inv = torch.exp(-pred_next_latent_sigma)

        next_z = self.encode(next_obs, goal)
        if self.dynamics_loss == 'direct':
            diff = ((pred_next_latent_mu - next_z.detach()) ** 2 * pred_next_latent_sigma_inv) + pred_next_latent_sigma
        elif self.dynamics_loss == 'delta':
            diff = ((pred_next_latent_mu - (next_z.detach() - z.detach())) ** 2 * pred_next_latent_sigma_inv) + pred_next_latent_sigma
        else:
            raise NotImplementedError

        loss = torch.mean(diff)
 
        total_loss = loss
        if self.dynamics_loss == 'direct':
            stats = {'step' : step,
                beginning + '/phi/model_error' : torch.mean(torch.norm(pred_next_latent_mu - next_z.detach(), dim = 1)).item(),
                }
        elif self.dynamics_loss == 'delta':
            stats = {'step' : step,
                beginning + '/phi/model_error' : torch.mean(torch.norm(pred_next_latent_mu - (next_z.detach() - z.detach()), dim = 1)).item(),
                }
        else:
            raise NotImplementedError

        
        logger.logging_tool.log(stats)

        return total_loss 

    def decoder_loss(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = 'train'):
        if self.on_policy_dynamics == 'probabilistic' and step > self.steps_till_on_policy and not self.decode_both:
            action = policy.sample_action(obs, goal, batched = True)
        elif self.on_policy_dynamics == 'deterministic' and step > self.steps_till_on_policy and not self.decode_both:
            action = policy.select_action(obs, goal, batched = True)

        z = self.encode(obs, goal)
        next_z = self.encode(next_obs, goal)
        if self.transition_model_type != 'next_observation':
            decodee = self.dynamics_model.sample_prediction(torch.cat([z, action], dim=1))
        else:
            decodee = next_z #Try not to use...

        if self.decode_both:
            decodee = torch.cat([z, decodee], dim = 1)

        if self.decoder_type == 'reward':
            pred_next_reward = self.decoder(decodee)
            decoder_loss = F.mse_loss(pred_next_reward.squeeze(), reward.squeeze())
        elif self.decoder_type == 'rtg':
            pred_next_reward = self.decoder(decodee)
            decoder_loss = F.mse_loss(pred_next_reward.squeeze(), rtg.squeeze())
        elif self.decoder_type == 'rtg_reward':
            pred_next_reward = self.decoder(decodee)
            decoder_loss = F.mse_loss(pred_next_reward[:,0], rtg.squeeze()) + F.mse_loss(pred_next_reward[:,1], reward.squeeze())
        elif self.decoder_type == 'action':
            pred_next_action = self.decoder(decodee) #Inverse Model
            decoder_loss = F.mse_loss(pred_next_action.squeeze(), action.squeeze())
        elif self.decoder_type == 'temporal':
            pred_td = self.decoder(decodee) #Inverse Model
            decoder_loss = F.mse_loss(pred_td.squeeze(), td.squeeze())
        elif self.decoder_type == 'temporal_action':
            pred_action_td = self.decoder(decodee) #Inverse Model
            decoder_loss = F.mse_loss(pred_action_td[:,0], td.squeeze()) + F.mse_loss(pred_action_td[:,1:].squeeze(), action.squeeze())
        elif self.decoder_type =='none':
            decoder_loss = torch.Tensor([0]).to(self.device)
        else:
            raise NotImplementedError

        return decoder_loss

    def train_batch(self, obs, action, next_obs, goal, reward, rtg, td, policy, step, log = True, take_step = True, beginning = 'train', comp_pool_imgs = None):

        action = torch.clip(action, min = -1, max = 1) * self.action_scale

        #self.encoder.train()
        encoder_loss, std_norm, collapse_level = self.encoder_loss(obs, action, next_obs, goal, reward, rtg, td, policy, step, log = log, beginning = beginning)
        transition_loss = self.transition_loss(obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = beginning)
        decoder_loss = self.decoder_loss(obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = beginning)

        #policy_decoder_loss = self.policy_decoder_loss(obs, action, next_obs, goal, reward, rtg, td, policy, step, beginning = beginning)

        total_loss = self.encoder_weight * encoder_loss + self.transition_weight * (transition_loss + decoder_loss)

        # Didnt work
        # if self.lambda_comp > 0 and comp_pool_imgs is not None:
        #     comp_loss, lambda_current = self.compositionality_loss(obs, comp_pool_imgs, step, log = log, beginning = beginning)
        #     total_loss = total_loss + lambda_current * comp_loss

        if log:
            stats = {'step' : step,
            beginning + '/phi/loss' : total_loss.item(),
            beginning + '/phi/encoder_loss' : encoder_loss.item(),
            beginning + '/phi/transition_loss' : transition_loss.item(),
            beginning + '/phi/decoder_loss' : decoder_loss.item(),
            beginning + '/phi/std_norm' : std_norm,
            beginning + '/phi/collapse_level' : collapse_level
            }

            logger.logging_tool.log(stats)
        else:
            print("Loss_PHI: " ,total_loss.item())

        if take_step:
            assert beginning == 'train'

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            total_loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        else:
            val_loss = total_loss.detach()

    def train_batches(self, dataset, batches=100):

        for b in range(batches):
            idxs = np.random.permutation(dataset.shape[0], self.train_batch_size)
            self.train_batch(dataset[idxs])

    def train_epoch(self, dataset):
        order = np.random.permutation(dataset.shape[0])
        iterations = dataset.shape[0] // self.train_batch_size

        for itr in range(iterations):
            self.train_batch(dataset[itr * self.train_batch_size: (itr + 1) * self.train_batch_size])

    def encode_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

    def decode_np(self, inputs, cont=True):
        assert False, "No decoder avaliable"

    def step_lr(self):
        self.encoder_optimizer_step.step()
        self.decoder_optimizer_step.step()

    def eval_loss(self, replay_buffer, policy, kwargs, step, log = True):
        comp_pool_imgs = self.sample_compositionality_pool(replay_buffer) if self.lambda_comp > 0 else None
        self.train_batch(kwargs['obs'], kwargs['action'], kwargs['next_obs'], kwargs['goal'], kwargs['reward'], \
            kwargs['rtg'], kwargs['td'], policy, step, log = log, take_step = False, beginning = 'eval', comp_pool_imgs = comp_pool_imgs)

    def update(self, replay_buffer, policy, kwargs, step, log = True):
        #Will run through dataset...

         #Does it matter if not same batch....?

        comp_pool_imgs = self.sample_compositionality_pool(replay_buffer) if self.lambda_comp > 0 else None
        self.train_batch(kwargs['obs'], kwargs['action'], kwargs['next_obs'], kwargs['goal'], \
            kwargs['reward'], kwargs['rtg'], kwargs['td'], policy, step, log = log, comp_pool_imgs = comp_pool_imgs)

        for _ in range(self.train_iters_per_update - 1):
            obs, action, _, reward, next_obs, not_done, goals, kwargs = replay_buffer.sample()
            comp_pool_imgs = self.sample_compositionality_pool(replay_buffer) if self.lambda_comp > 0 else None
            self.train_batch(obs, action, next_obs, goals, reward, kwargs['rtg'], kwargs['td'], policy, step, log = log, comp_pool_imgs = comp_pool_imgs)



