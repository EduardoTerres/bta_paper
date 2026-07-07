import torch
import torch.nn as nn


class VectorEncoder(nn.Module):
    """MLP encoder for low-dimensional/vector observations, such as one-hot states.

    Drop-in replacement for goalbisim.representation.encoders.RADencoder.PixelEncoder:
    same constructor signature/attributes (feature_dim, output_logits, goal_flag, ...) and
    same forward(obs, detach, detach_all) contract, so it can be substituted wherever a
    PixelEncoder is used without touching downstream losses/policies. No convolutions.
    """

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=128,
                 output_logits=True, tanh_scale=1, goal_flag=False):
        super().__init__()
        assert len(obs_shape) == 1, "VectorEncoder expects a 1D obs_shape, e.g. (2,)"

        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.tanh_scale = tanh_scale
        self.goal_flag = goal_flag
        self.output_logits = output_logits

        in_dim = obs_shape[0] * (2 if goal_flag else 1)
        hidden = num_filters

        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.net = nn.Sequential(*layers)

        self.fc = nn.Linear(hidden, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        # Kept for interface parity with PixelEncoder (used by GoalBisim.compute_logits
        # when use_contrastive=True).
        self.W_contrast = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.W_map = nn.Parameter(torch.rand(feature_dim, feature_dim))

        self.outputs = dict()

    def forward(self, obs, detach=False, detach_all=False):
        h = self.net(obs)

        if detach or detach_all:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm) * self.tanh_scale
            self.outputs['tanh'] = out

        if detach_all:
            out = out.detach()

        return out

    def copy_conv_weights_from(self, source):
        """No conv layers to tie; kept for interface parity with PixelEncoder."""
        pass

    def log(self, L, step, log_freq):
        pass
