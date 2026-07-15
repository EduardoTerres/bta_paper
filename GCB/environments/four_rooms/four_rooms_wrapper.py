import sys
from pathlib import Path

import gym
import numpy as np
from gym import spaces


REPO_ROOT = Path(__file__).resolve().parents[3]
BOOLEAN_COMPOSITION_ROOT = REPO_ROOT / "boolean_composition"
if str(BOOLEAN_COMPOSITION_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOLEAN_COMPOSITION_ROOT))

from four_rooms.GridWorld import GridWorld
from four_rooms.config import Config_4, Config_8, Config_16


CONFIGS = {
    4: Config_4,
    8: Config_8,
    16: Config_16,
}


class FourRoomsGoalEnv(gym.Env):
    def __init__(
        self,
        num_rooms=4,
        dense_rewards=False,
        goal_reward=1.0,
        step_reward=-0.01,
        max_episode_steps=None,
        obs_img_dim=64,
        slip_prob=0.0,
        start_position=None,
    ):
        if num_rooms not in CONFIGS:
            raise ValueError("num_rooms must be one of 4, 8, or 16")

        self.num_rooms = num_rooms
        self.config = CONFIGS[num_rooms]
        self.terminal_states = self.config["T_states"]
        self.goal_positions = self.config["Goals"]
        self._max_episode_steps = max_episode_steps or 4 * num_rooms + 32
        self.obs_img_dim = obs_img_dim
        self._rng = np.random.RandomState()
        self._episode_step = 0
        self._goal_position = None
        self._goal_cycle = []
        self._joint_goal_positions = None

        self.env = GridWorld(
            MAP="MAP_" + str(num_rooms),
            goals=self.terminal_states,
            T_states=self.terminal_states,
            dense_rewards=dense_rewards,
            goal_reward=goal_reward,
            step_reward=step_reward,
            slip_prob=slip_prob,
            start_position=start_position,
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            0, 255, shape=(3, obs_img_dim, obs_img_dim), dtype=np.uint8
        )
        self.state_space = spaces.Box(
            low=0,
            high=max(self.env.n, self.env.m),
            shape=(2,),
            dtype=np.float32,
        )

    def _action_to_discrete(self, action):
        action = np.asarray(action)
        if action.shape == ():
            return int(action)
        return int(np.argmax(action))

    def _state_position(self):
        return tuple(self.env.state[1])

    def _state_array(self):
        return np.asarray(self._state_position(), dtype=np.float32)

    def _goal_state(self):
        return [self._goal_position, self._goal_position]

    def _make_obs(self, position, extra_positions=None):
        grid = np.zeros((self.env.n, self.env.m, 3), dtype=np.uint8)
        grid[:, :] = np.array([255, 255, 255], dtype=np.uint8)

        for x, y in self.env.walls:
            grid[x, y] = np.array([0, 0, 0], dtype=np.uint8)

        gx, gy = self._goal_position
        grid[gx, gy] = np.array([0, 80, 255], dtype=np.uint8)

        for x, y in (extra_positions if extra_positions is not None else [position]):
            grid[x, y] = np.array([0, 180, 0], dtype=np.uint8)

        scale_x = int(np.ceil(self.obs_img_dim / self.env.n))
        scale_y = int(np.ceil(self.obs_img_dim / self.env.m))
        image = np.kron(grid, np.ones((scale_x, scale_y, 1), dtype=np.uint8))
        image = image[: self.obs_img_dim, : self.obs_img_dim]
        return image.transpose(2, 0, 1).copy()

    def _obs(self):
        return self._make_obs(self._state_position())

    def _goal_obs(self):
        return self._make_obs(self._goal_position)

    def _joint_goal_obs(self, goal_positions):
        return self._make_obs(self._goal_position, extra_positions=goal_positions)

    def render_goal_set(self, positions):
        """Pure rendering of an arbitrary goal cell set (green dots only, no blue
        marker). Unlike `_joint_goal_obs`, this does not depend on the episode's
        current `_goal_position`, so it can render e.g. an intersection/union of
        two sampled goal sets that may not include it (or may be empty)."""
        grid = np.zeros((self.env.n, self.env.m, 3), dtype=np.uint8)
        grid[:, :] = np.array([255, 255, 255], dtype=np.uint8)

        for x, y in self.env.walls:
            grid[x, y] = np.array([0, 0, 0], dtype=np.uint8)

        for x, y in positions:
            grid[x, y] = np.array([0, 180, 0], dtype=np.uint8)

        scale_x = int(np.ceil(self.obs_img_dim / self.env.n))
        scale_y = int(np.ceil(self.obs_img_dim / self.env.m))
        image = np.kron(grid, np.ones((scale_x, scale_y, 1), dtype=np.uint8))
        image = image[: self.obs_img_dim, : self.obs_img_dim]
        return image.transpose(2, 0, 1).copy()

    def _next_goal_index(self):
        if not self._goal_cycle:
            self._goal_cycle = list(range(len(self.goal_positions)))
            self._rng.shuffle(self._goal_cycle)
        return self._goal_cycle.pop()

    def reset(self, seed=None):
        if seed is not None:
            self._rng.seed(seed)
            np.random.seed(seed)
            self._goal_cycle = []

        self._joint_goal_positions = None
        self._goal_position = self.goal_positions[self._next_goal_index()]
        self.env.goals = [self._goal_state()]
        self.env.T_states = self.terminal_states
        self.env.reset()
        self._episode_step = 0

        extra = {
            "state": self._state_array(),
            "seed": seed if seed is not None else int(self._rng.randint(99999)),
            "goal_position": np.asarray(self._goal_position, dtype=np.float32),
        }
        return self._obs(), self._goal_obs(), extra

    def reset_joint(self, goal_positions):
        """Reset into an episode of the joint MDP whose terminal set is the union of
        `goal_positions` (see boolean_composition): success = reaching ANY of them."""
        self._joint_goal_positions = [tuple(p) for p in goal_positions]
        self._goal_position = self._joint_goal_positions[0]
        self.env.goals = [[list(p), list(p)] for p in self._joint_goal_positions]
        self.env.T_states = self.terminal_states
        self.env.reset()
        self._episode_step = 0

        extra = {
            "state": self._state_array(),
            "goal_positions": np.asarray(self._joint_goal_positions, dtype=np.float32),
        }
        return self._obs(), self._joint_goal_obs(self._joint_goal_positions), extra

    def demo_reset(self, seed=None):
        return self.reset(seed=seed)

    def jitter_reset(self, seed=None):
        obs, _, extra = self.reset(seed=seed)
        return obs, extra

    def demo_jitter_reset(self, seed=None):
        return self.jitter_reset(seed=seed)

    def step(self, action):
        discrete_action = self._action_to_discrete(action)
        self.env.step(discrete_action)
        self._episode_step += 1

        if self._joint_goal_positions is not None:
            success = self._state_position() in self._joint_goal_positions
        else:
            success = self.env.state == self._goal_state()
        timeout = self._episode_step >= self._max_episode_steps
        reward = 1.0 if success else -0.01

        extra = {
            "state": self._state_array(),
            "goal_position": np.asarray(self._goal_position, dtype=np.float32),
            "discount": 0.0 if success or timeout else 1.0,
        }
        return self._obs(), reward, bool(success or timeout), extra

    def set_distractor(self, distractor):
        self.distractor = distractor


if __name__ == "__main__":
    # Each goal is represented internally as a terminal state [pos, pos] (see
    # GridWorld.step / config.T_states), i.e. the goal position is collapsed
    # into both slots of the state pair. Print that mapping for every config
    # so it's clear which raw grid positions a given num_rooms setting uses.
    for num_rooms, config in CONFIGS.items():
        print(f"num_rooms={num_rooms}")
        for goal_position, terminal_state in zip(config["Goals"], config["T_states"]):
            print(f"  goal {goal_position} -> collapsed terminal state {terminal_state}")

