import time

import gym

from mcts import MCTS
from models import CartDyna, CartPred, CartRepr

env = gym.make("CartPole-v0")

action_size = env.action_space.n
obs_size = env.observation_space.shape[0]

mcts = MCTS(
    action_size=action_size,
    obs_size=obs_size,
    repr_net=CartRepr,
    dyna_net=CartDyna,
    pred_net=CartPred
)

n_steps = 100
n_simulations = 100

frame = env.reset()
for i in range(n_steps):
    tree = mcts.search(n_simulations, frame)
    action = tree.pick_action()
    
    env.render("human")
    frame, score, over, _ = env.step(action)
    
    mcts.update()

    print(f'Completed step {i:5} with action {action} and score {score}.')

env.close()
