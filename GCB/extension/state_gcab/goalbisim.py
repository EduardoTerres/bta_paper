import torch
import torch.nn as nn

from goalbisim.representation.goalbisim import GoalBisim

from .encoders import VectorEncoder
from .pairedstategoal import StatePairedStateGoal


class StateGoalBisim(GoalBisim):
    """GoalBisim for vector observations instead of images.

    Identical to GoalBisim except psi is a VectorEncoder (MLP) instead of a
    PixelEncoder, and phi is a StatePairedStateGoal instead of a PairedStateGoal.
    Every loss/update method (loss, compute_logits, temporal_contrastive_loss,
    train_batch, step_lr, eval_loss, update) is inherited unchanged, since they
    only operate on self.psi/self.phi's feature_dim-sized outputs.
    """

    def __init__(
            self,
            obs_shape,
            device,
            transition_model_type='ensemble',
            psi_loss_form='delta',
            disconnect_psi=False,
            using_phi=True,
            metric_loss='l1',
            metric_distance='reward',
            decoder_type='reward',
            dynamics_loss='direct',
            action_weight=10,
            use_contrastive=False,
            contrastive_weight=0.25,
            phi_updates_before_psi=0,
            on_policy_dynamics=False,
            dual_optimization=False,
            disconnect_implict_policy=True,
            ground_space=True,
            decode_both=True,
            train_iters_per_update_psi=1,
            train_iters_per_update_phi=1,
            steps_till_on_policy=4000,
            encoder_weight=1,
            transition_weight=1,
            action_shape=(1, 2),
            discount=0.99,
            action_scale=0.5,
            feature_dim=256,
            num_layers=4,
            num_filters=32,
            lr=1e-3,
            weight_decay=0,
            output_logits=True,
            output_logits_paired=True,
            num_layers_paired=4,
            num_filters_paired=32,
            lr_paired=1e-3,
            weight_decay_paired=0):
        nn.Module.__init__(self)

        self.using_phi = using_phi
        self.device = device
        self.psi = VectorEncoder(obs_shape, feature_dim, num_layers, num_filters, output_logits=output_logits).to(self.device)
        self.encoder = self.psi
        self.feature_dim = feature_dim

        self.disconnect_psi = disconnect_psi

        self.phi_updates_before_psi = phi_updates_before_psi
        self.ground_space = ground_space

        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_weight = contrastive_weight
            self.cross_entropy = nn.CrossEntropyLoss()

        self.train_iters_per_update = train_iters_per_update_psi

        if self.using_phi:
            self.phi = StatePairedStateGoal(
                obs_shape, device, transition_model_type=transition_model_type, discount=discount,
                metric_distance=metric_distance, decode_both=decode_both, decoder_type=decoder_type,
                dual_optimization=dual_optimization, feature_dim=feature_dim,
                disconnect_implict_policy=disconnect_implict_policy, num_layers=num_layers_paired,
                dynamics_loss=dynamics_loss, metric_loss=metric_loss,
                train_iters_per_update=train_iters_per_update_phi, num_filters=num_filters_paired,
                lr=lr_paired, action_shape=action_shape, on_policy_dynamics=on_policy_dynamics,
                action_weight=action_weight, steps_till_on_policy=steps_till_on_policy,
                action_scale=action_scale, output_logits=output_logits_paired,
                weight_decay=weight_decay_paired, encoder_weight=encoder_weight,
                transition_weight=transition_weight)

            self.psi_optimizer = torch.optim.AdamW(self.psi.parameters(), lr=lr, weight_decay=weight_decay)
            self.optimizer_step = torch.optim.lr_scheduler.StepLR(self.psi_optimizer, 1, gamma=0.95, last_epoch=-1, verbose=False)
            self.psi_loss_form = psi_loss_form
            self.mse = nn.MSELoss()
            assert self.train_iters_per_update > 0
            self.discount = discount
