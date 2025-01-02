import random

import gymnasium as gym
import numpy as np
# import pandas as pd

from gymnasium import spaces
from enum import Enum
from flowenv.src.const import Const

CONST = Const()

class Actions(Enum):
    right = 1
    wrong = -1

class FlowTestEnv(gym.Env):
    def __init__(self, render_mode=None, data=None):
        super(FlowTestEnv, self).__init__()

        self.render_mode = render_mode
        self.data = [] if data is None else data

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(CONST.features_labels),), dtype=np.float64
        )

        self.state = {}
        self.done = False
        self.reward = 0.0
        self.index = 0
        self.pull = []
        self.state_len = len(self.data)
        self.terminate = random.random() < 0.05

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self.data[CONST.features_labels].iloc[0].values
        self.state_len = len(self.state)
        self.pull = []
        # self.terminate
        info = { "state": self.state }

        return self.state, info

    def step(self, action):
        self.state = self.data[CONST.features_labels].iloc[self.index].values
        answer = self.data[CONST.reference_label].iloc[self.index]

        self.index += 1
        if action == answer:
            reward = 1
            if action == 1:
                confusion = "TP"
                confusion_value = 0
            else:
                confusion = "TN"
                confusion_value = 1
        else:
            reward = -1
            if action == 1:
                confusion = "FP"
                confusion_value = 2
            else:
                confusion = "FN"
                confusion_value = 3

        print(f"{action}:{answer} => reward: {reward}, confusion: {confusion}")
        try:
            observation = self.data[CONST.features_labels].iloc[self.index].values
        except IndexError:
            self.index = 0
            observation = self.data[CONST.features_labels].iloc[self.index].values
        info = { "confusion": confusion, "value": confusion_value }

        terminated = self.index % 100 == 0
        return observation, reward, terminated, False, info

    def render(self, mode=None):
        return

    def close(self):
        return
