import datetime
import os
import sys
import time
import yaml

import gym

from torch.utils.tensorboard import SummaryWriter

from mcts import MCTS
from models import MuZeroCartNet, MuZeroAtariNet
from training import GameRecord, ReplayBuffer

config = yaml.safe_load(open("config-breakout.yaml", "r"))

env = gym.make(config["env_name"])

action_size = env.action_space.n

obs_size = env.observation_space.shape
if len(obs_size) == 1:
    obs_size = obs_size[0]
net_type_dict = {"CartPole-v0": MuZeroCartNet, "Breakout-v0": MuZeroAtariNet}

muzero_class = net_type_dict[config["env_name"]]

muzero_network = muzero_class(action_size, obs_size, config)
learning_rate = config["learning_rate"]
muzero_network.init_optim(learning_rate)

mcts = MCTS(action_size=action_size, mu_net=muzero_network, config=config)

log_name = os.path.join(
    config["log_dir"], datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
)
tb_writer = SummaryWriter(log_dir=log_name)

memory = ReplayBuffer(config)
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
    frame = env.reset()

    game_record = GameRecord(
        action_size=action_size, init_frame=frame, discount=config["discount"]
    )

    temperature = 10 / (total_games + 10)
    score = 0

    if total_games % 10 == 0:
        learning_rate = learning_rate * config["learning_rate_decay"]
        mcts.mu_net.init_optim(learning_rate)

    while not over:
        tree = mcts.search(config["n_simulations"], frame)
        action = tree.pick_game_action(temperature=temperature)

        if config["render"]:
            env.render("human")

        frame, reward, over, _ = env.step(action)

        game_record.add_step(frame, action, reward, tree)

        # mcts.update()

        frames += 1
        score += reward

    memory.save_game(game_record)
    metrics_dict = mcts.train(memory, config["n_batches"])

    for key, val in metrics_dict.items():
        tb_writer.add_scalar(key, val, total_games)

    tb_writer.add_scalar("Score", score, total_games)

    # print(mcts.mu_net.dyna_net.fc1.weight.data)
    # print(mcts.mu_net.dyna_net.fc1.weight.grad)
    print(
        f"Completed game {total_games + 1:4} with score {score:3}. Loss was {metrics_dict['Loss/total'].item():5.3f}."
    )
    total_games += 1


env.close()
