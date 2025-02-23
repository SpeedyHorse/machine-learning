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


class FlowEnv(gym.Env):
    def __init__(self, render_mode=None, data=None):
        super(FlowEnv, self).__init__()

        self.render_mode = render_mode
        if data is not None:
            self.features = data.drop(columns=["Binary Label"])
            self.labels = data["Binary Label"]
        # self.data = [] if data is None else data

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features.columns),), dtype=np.float64
        )

        self.state = {}
        self.done = False
        self.reward = 0.0
        self.rng = np.random.default_rng(0)
        self.data_len = len(self.features)
        self.index_array = np.arange(0, self.data_len - 1)
        self.index = 0
        self.terminate = random.random() < 0.05

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data_len = len(self.features)
        self.index_array = np.arange(0, self.data_len - 1)
        self.index = self.rng.choice(self.index_array, 1)[0]

        np.delete(self.index_array, self.index)
        self.state = self.features.iloc[self.index].values
        # self.terminate
        info = { "state": self.state, "row": len(self.features), "column": len(self.features.columns) }

        return self.state, info

    def step(self, action):
        answer = self.labels.iloc[self.index]

        self.index = self.rng.choice(self.index_array, 1)[0]

        reward = 1 if action == answer else -1

        """
        1, 1: TP : action == answer == 1
        0, 1: FN : action == 0, answer == 1
        1, 0: FP : action == 1, answer == 0
        0, 0: TN : action == answer == 0
        
        if action == answer:
            # (0, 0) or (1, 1)
            if action == 1:
                matrix_position = (0, 0)
            else:
                matrix_position = (1, 1)
        else:
            # (0, 1) or (1, 0)
            if action == 1:
                matrix_position = (1, 0)
            else:
                matrix_position = (0, 1)
        """
        if action == answer:
            # (0, 0) or (1, 1)
            if action == 1:
                matrix_position = (1, 1)
            else:
                matrix_position = (0, 0)
        else:
            # (0, 1) or (1, 0)
            if action == 1:
                matrix_position = (1, 0)
            else:
                matrix_position = (0, 1)
        # matrix_position = (action, answer)
        info = {
            "confusion_position": matrix_position,
            "action": action,
            "answer": answer,
        }

        try:
            observation = self.features.iloc[self.index].values
        except IndexError:
            self.index = 0
            observation = self.features.iloc[self.index].values
        # print(self.data_len, len(self.index_array), self.data_len - len(self.index_array))

        # terminated = random.random() < 0.05 and self.data_len - len(self.index_array) >= 100
        terminated = random.random() < 0.01
        return observation, reward, terminated, False, info

    def render(self, mode=None):
        return

    def close(self):
        return
