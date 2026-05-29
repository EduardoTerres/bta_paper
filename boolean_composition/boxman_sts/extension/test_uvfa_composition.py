from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "boxman_sts"))

from boxman_sts.dqn import ComposedDQN_onoff


class ConstantDQN(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, obs_goal):
        if obs_goal.shape[0] == 1:
            return self.values.clone()
        return self.values.repeat(obs_goal.shape[0], 1)


def obs_goal_for(goal_value):
    obs = torch.zeros(1, 2, 2, 3)
    goal = torch.full((1, 2, 2, 3), goal_value)
    return torch.cat((obs, goal), dim=3)


def test_onoff_uses_on_dqn_for_desired_goal():
    dqn = ComposedDQN_onoff(
        ConstantDQN([1.0, 2.0]),
        ConstantDQN([-1.0, -2.0]),
        on_goals=[torch.full((2, 2, 3), 7.0)],
    )

    q = dqn(obs_goal_for(7.0))

    assert torch.equal(q, torch.tensor([1.0, 2.0]))


def test_onoff_uses_off_dqn_for_undesired_goal():
    dqn = ComposedDQN_onoff(
        ConstantDQN([1.0, 2.0]),
        ConstantDQN([-1.0, -2.0]),
        on_goals=[torch.full((2, 2, 3), 7.0)],
    )

    q = dqn(obs_goal_for(3.0))

    assert torch.equal(q, torch.tensor([-1.0, -2.0]))


def test_onoff_routes_mixed_goal_batch_independently():
    dqn = ComposedDQN_onoff(
        ConstantDQN([1.0, 2.0]),
        ConstantDQN([-1.0, -2.0]),
        on_goals=[torch.full((2, 2, 3), 7.0)],
    )
    batch = torch.cat([obs_goal_for(7.0), obs_goal_for(3.0)], dim=0)

    q = dqn(batch)

    assert torch.equal(q, torch.tensor([[1.0, 2.0], [-1.0, -2.0]]))
