import numpy as np
from gym import spaces

from environments.four_rooms.four_rooms_wrapper import FourRoomsGoalEnv


class VectorFourRoomsGoalEnv(FourRoomsGoalEnv):
    """FourRoomsGoalEnv variant that exposes one-hot state vectors as observations/goals
    instead of rendered (3, 64, 64) grid images. GridWorld dynamics, rewards, and the
    oracle/BFS logic used to collect demonstrations are untouched (inherited); only the
    observation pipeline is swapped.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active_goal_positions = [tuple(self._goal_position)] if self._goal_position else []
        self._position_to_index = {
            position: idx for idx, position in enumerate(self.env.possiblePositions)
        }
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.env.possiblePositions),),
            dtype=np.float32,
        )

    def _one_hot_position(self, position):
        out = np.zeros(self.observation_space.shape, dtype=np.float32)
        out[self._position_to_index[tuple(position)]] = 1.0
        return out

    def _obs(self):
        return self._one_hot_position(self._state_position())

    def _goal_obs(self):
        return self._one_hot_position(self._goal_position)

    def reset(self, seed=None):
        obs, goal, extra = super().reset(seed=seed)
        self._active_goal_positions = [tuple(self._goal_position)]
        return obs, goal, extra

    def set_goal_positions(self, goal_positions):
        self._active_goal_positions = [tuple(goal) for goal in goal_positions]
        self._goal_position = self._active_goal_positions[0]
        goal_states = [[goal, goal] for goal in self._active_goal_positions]
        self.env.goals = goal_states
        self.env.T_states = goal_states
        return self._goal_obs()

    def step(self, action):
        discrete_action = self._action_to_discrete(action)
        self.env.step(discrete_action)
        self._episode_step += 1

        active_goal_states = [[goal, goal] for goal in self._active_goal_positions]
        success = self.env.state in active_goal_states
        timeout = self._episode_step >= self._max_episode_steps
        reward = 1.0 if success else -0.01

        extra = {
            "state": self._state_array(),
            "goal_position": np.asarray(self._goal_position, dtype=np.float32),
            "active_goal_positions": np.asarray(self._active_goal_positions, dtype=np.float32),
            "discount": 0.0 if success or timeout else 1.0,
        }
        return self._obs(), reward, bool(success or timeout), extra

    def render_obs(self):
        """Render the current state as a (3, H, W) uint8 grid image, for visualizing
        rollouts only -- the inherited `_make_obs` is never used as the training obs.
        """
        return self._make_obs(self._state_position())

    def render_goal(self):
        """Render the goal position as a (3, H, W) uint8 grid image (visualization only)."""
        return self._make_obs(self._goal_position)
