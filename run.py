import time

import gym

from MCTS import MCTS

env = gym.make("BreakoutNoFrameskip-v4")
env.reset()

print(env.action_space)

n_steps = 1000
mcts = MCST()

for _ in range(n_steps):
    env.render("human")

    frame_rate = 30
    time.sleep(1 / frame_rate)

    env.step(env.action_space.sample())

env.close()
