import time

import gym

from mcts import MCTS
from models import MuZeroCartNet
from training import GameRecord, ReplayBuffer

LATENT_SIZE = 10
MAX_FRAMES = 200

env = gym.make("CartPole-v0")

action_size = env.action_space.n
obs_size = env.observation_space.shape[0]

muzero_network = MuZeroCartNet(action_size, obs_size, LATENT_SIZE)

mcts = MCTS(action_size=action_size, obs_size=obs_size, mu_net=muzero_network)

n_steps = 100
n_simulations = 100
n_games = 6

memory = ReplayBuffer(size=5)
# open muz implementation uses a GameHistory class
# with observation_history, action_history, reward_history
# to_play which is who is to play in case it's a multiplayer, turn-based game
# also stores search stats ie the number of times each child/action node was visited
# which becomes the policy
# and the value which is the average value as calculated by the MCTS
# can also store the reanalysed predicted root values

total_games = 0

while True:
    frames = 0
    over = False
    game_record = GameRecord(action_size)

    frame = env.reset()
    game_record.observations.append(frame)

    while not over and frames < MAX_FRAMES:
        tree = mcts.search(n_simulations, frame)
        action = tree.pick_action()

        # env.render("human")
        frame, score, over, _ = env.step(action)

        game_record.add_step(frame, score, action, tree)

        # mcts.update()

        frames += 1

    memory.save_game(game_record)
    batch = memory.get_batch()
    loss = mcts.train(batch)
    print(f"Completed game {total_games + 1:4} with score {frames:3}. Loss was {loss.item():5.3f}.")
    total_games += 1

env.close()
