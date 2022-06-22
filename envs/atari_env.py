import gym
import cv2
import numpy as np


class WrappedEnv(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.env = env
        self.full_image_size = [
            (config["obs_size"][2] + 1) * config["last_n_frames"],
            *config["obs_size"][:2],
        ]

    def reset(self):
        obs = self.env.reset()
        return self.normalize_atari_image(obs)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.normalize_atari_image(next_state)
        return next_state, reward, done, info

    def normalize_atari_image(self, image):
        image_a = np.array(image, dtype=np.float32)
        # resize for neat convolutions, taken from openmuZ
        image_a = cv2.resize(image_a, (96, 96), interpolation=cv2.INTER_AREA)
        image_a = image_a / 256
        return image_a.transpose(2, 0, 1)


def make_env(config):
    env = WrappedEnv(gym.make(config["env_name"]), config)
    return env
