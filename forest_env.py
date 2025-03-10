"""
Copyright 2020 Sahand Rezaei-Shoshtari. All Rights Reserved.

Forest-fire gymnasium environment.

Modified by Logan Mondal Bhamidipaty.
"""

import numpy as np
import gymnasium as gym
import cv2
from gymnasium import spaces
from gymnasium.utils import seeding

from .forest import Forest

STATE_W = 64
STATE_H = 64
T_HORIZON = 100


class ForestFireEnv(gym.Env):

    def __init__(self, **env_kwargs):
        self.seed()
        self.reward = 0
        self.state = None
        self.t = 0

        self.forest = Forest(**env_kwargs)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10, shape=(STATE_H, STATE_W), dtype=np.uint8)
        self._max_episode_steps = T_HORIZON

    def step(self, action):
        aimed_fire, is_fire = self.forest.step(action)
        self.t += 1
        step_reward = 0

        # determine if we have reached the time limit
        truncated = self.t >= T_HORIZON
        terminated = False  # no other terminal condition

        # reward calculation
        if aimed_fire:
            step_reward += 1
        if not aimed_fire and is_fire:
            step_reward -= 1
        if truncated:
            if np.mean(self.forest.world) > 0.5 * self.forest.p_init_tree:
                step_reward += 100
            else:
                step_reward -= 100

        self.reward = step_reward

        state = self.forest.world
        if state.shape != (STATE_H, STATE_W):
            state = self._scale(state, STATE_H, STATE_W)
        self.state = np.array(state) / self.forest.FIRE_CELL

        return self.state, step_reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        # Optionally re-seed the environment.
        self.seed(seed)
        self.forest.reset()
        self.reward = 0
        self.t = 0
        # Do not take a step here so that time remains 0 at reset.
        state = self.forest.world
        if state.shape != (STATE_H, STATE_W):
            state = self._scale(state, STATE_H, STATE_W)
        self.state = np.array(state) / self.forest.FIRE_CELL
        return self.state, {}

    def render(self, mode='human'):
        self.forest.render()

    def close(self):
        cv2.destroyAllWindows()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _scale(self, im, height, width):
        original_height, original_width = im.shape
        return [[im[int(original_height * r / height)][int(original_width * c / width)]
                 for c in range(width)] for r in range(height)]
