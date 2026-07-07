import numpy as np
import torch

from goalbisim.data.management.goalreplaybuffer import GoalReplayBuffer


class VectorGoalReplayBuffer(GoalReplayBuffer):
    """GoalReplayBuffer for low-dim vector observations/goals.

    GoalReplayBuffer.sample() always divides obs/goals by 255 (image normalization).
    That's wrong for state vectors, so this override casts to float tensors
    without rescaling. Storage, add(), save()/load() are inherited unchanged.
    """

    def sample(self, batch_size=None, fetch_states=False, contrastive_fetch=False):
        used_batch_size = batch_size if batch_size else self.batch_size
        idxs = np.random.randint(
            self.sample_start, self.capacity if self.full else self.idx - self.current_trajectory_length,
            size=used_batch_size,
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        goals = self.goals[idxs]
        pos = obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float().contiguous()
        next_obses = torch.as_tensor(next_obses, device=self.device).float().contiguous()
        goals = torch.as_tensor(goals, device=self.device).float().contiguous()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rtg_rewards = torch.as_tensor(self.reward_to_go[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        tds = torch.as_tensor(self.temporal_distance[idxs], device=self.device)

        if fetch_states:
            kwargs = {'states': None, 'next_states': None, 'rtg': rtg_rewards, 'td': tds, 'idxs': idxs, 'pos': pos}
        else:
            kwargs = {'rtg': rtg_rewards, 'td': tds, 'idxs': idxs, 'pos': pos}

        return obses, actions, rewards, next_obses, not_dones, goals, kwargs
