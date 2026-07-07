import torch
import torch.nn as nn

from goalbisim.dynamics.dynamics_models import make_transition_model
from goalbisim.representation.pairedstategoal import PairedStateGoal

from .encoders import VectorEncoder


class StatePairedStateGoal(PairedStateGoal):
    """PairedStateGoal for vector (obs, goal) inputs instead of images.

    Identical to PairedStateGoal except the internal encoder is a VectorEncoder (MLP)
    instead of a PixelEncoder (CNN). All losses (encoder_loss, transition_loss,
    decoder_loss, train_batch, update, eval_loss) are inherited unchanged, since they
    only depend on self.encoder's feature_dim-sized output, not its architecture.
    """

    def __init__(
            self,
            obs_shape,
            device,
            transition_model_type='ensemble',
            metric_loss='l1',
            metric_distance='reward',
            decoder_type='reward',
            dynamics_loss='direct',
            dual_optimization=False,
            action_weight=1,
            on_policy_dynamics=False,
            decode_both=False,
            disconnect_implict_policy=True,
            train_iters_per_update=1,
            action_shape=(5, 1),
            action_scale=1,
            discount=0.99,
            steps_till_on_policy=3000,
            encoder_weight=1,
            transition_weight=1,
            feature_dim=256,
            num_layers=4,
            num_filters=32,
            output_logits=True,
            lr=1e-3,
            weight_decay=0):
        nn.Module.__init__(self)

        self.device = device
        self.encoder = VectorEncoder(
            obs_shape, feature_dim, num_layers, num_filters, output_logits=output_logits, goal_flag=True,
        ).to(self.device)
        self.phi = self
        self.psi = self

        self.action_scale = action_scale
        self.decode_both = decode_both
        self.metric_loss = metric_loss
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

        scale = 2 if self.decode_both else 1

        if self.decoder_type in ('reward', 'rtg', 'temporal', 'none'):
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(self.device)
        elif self.decoder_type == 'rtg_reward':
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 2)).to(self.device)
        elif self.decoder_type == 'temporal_action':
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1 + action_shape[0])).to(self.device)
        elif self.decoder_type == 'action':
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim * scale, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, action_shape[0])).to(self.device)
        else:
            raise NotImplementedError

        if self.transition_model_type not in ('next_observation', 'next_observation_l1'):
            if self.transition_model_type in ('ensemble_proper', 'next_observation_ensemble'):
                input_text = 'ensemble'
            else:
                input_text = self.transition_model_type

            self.dynamics_model = make_transition_model(input_text, feature_dim, action_shape).to(self.device)
        else:
            self.dynamics_model = make_transition_model('', feature_dim, action_shape).to(self.device)

        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.decoder_optimizer = torch.optim.AdamW(
            list(self.decoder.parameters()) + list(self.dynamics_model.parameters()), lr=lr, weight_decay=weight_decay,
        )

        self.encoder_optimizer_step = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, 1, gamma=0.9, last_epoch=-1, verbose=False)
        self.decoder_optimizer_step = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, 1, gamma=0.9, last_epoch=-1, verbose=False)

        self.discount = discount
        self.mse = nn.MSELoss()
