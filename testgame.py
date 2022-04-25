import random

import numpy as np

<<<<<<< HEAD
from training import Memory
=======
from memory import Memory
>>>>>>> origin/testgame


class ASpace:
    def __init__(self, n):
        self.n = n


class TestEnv:
    def __init__(self):
<<<<<<< HEAD
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
=======
        self.counter = 0
        self.action_space = ASpace(2)
        self.over = False

    def step(self, action):
        self.counter += 1

        img = np.full([210, 160, 3], (self.counter % 7) * 30, dtype=np.float32)
        img[0, :] = self.counter
        reward = (self.counter - 1) % 7
        if self.counter > 50:
            self.over = True

        return img, reward, self.over, False

    def reset(self):
        self.counter = 0
        self.over = 0
        return np.full([210, 160, 3], (self.counter % 7) * 30, dtype=np.float32)

    def close(self):
        pass


class TestEnvD:
    def __init__(self):
        self.counter = 0
        self.action_space = ASpace(2)
        self.over = False

    def step(self, action):
        self.counter += 1

        img = np.full([4], self.counter % 7, dtype=np.float32)
        img[0] = self.counter
        reward = (self.counter - 1) % 7
        if self.counter > 50:
            self.over = True

        return img, reward, self.over, False

    def reset(self):
        self.counter = 0
        self.over = 0
        return np.full([4], self.counter % 7, dtype=np.float32)
>>>>>>> origin/testgame

    def close(self):
        pass
