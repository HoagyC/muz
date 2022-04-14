import random

import numpy as np

from training import Memory


class ASpace:
    def __init__(self, n):
        self.n = n


class TestEnv:
    def __init__(self):
        self.last_val = 0
        self.action_space = ASpace(2)

    def step(self, action):
        img = np.full([210, 120, 3], self.last_val * 30)

        self.last_val += 1
        if self.last_val > 6:
            self.last_val = 0

        if random.random() < 0.02:
            over = True
        else:
            over = False

        return img, self.last_val, over, False

    def reset(self):
        self.last_val = 0
        return np.full([210, 120, 3], 0)

    def close(self):
        pass


class TestEnvD:
    def __init__(self):
        self.last_val = 0
        self.action_space = ASpace(2)

    def step(self, action):
        img = np.full([4], self.last_val)

        self.last_val += 1
        if self.last_val > 6:
            self.last_val = 0

        if random.random() < 0.02:
            over = True
        else:
            over = False

        return img, self.last_val, over, False

    def reset(self):
        self.last_val = 0
        return np.full([4], 0)

    def close(self):
        pass
