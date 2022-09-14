import gym
import cv2

import numpy as np


class WrappedEnv(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.env = env
        self.full_image_size = self.observation_space.shape

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def convert_to_int(self, obs):
        return ((obs * 64) + 128).astype(np.ubyte)


def make_env(config):
    env = WrappedEnv(gym.make(config["env_name"]), config)
    return env