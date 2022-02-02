import datetime
import os
import time
import yaml

import gym

from torch.utils.tensorboard import SummaryWriter

from mcts import MCTS
from models import MuZeroCartNet
from training import GameRecord, ReplayBuffer

config = yaml.safe_load(open("config.yaml", "r"))

env = gym.make(config["env_name"])

action_size = env.action_space.n
obs_size = env.observation_space.shape[0]

muzero_network = MuZeroCartNet(action_size, obs_size, config)

mcts = MCTS(
    action_size=action_size, obs_size=obs_size, mu_net=muzero_network, config=config
)

log_name = os.path.join(
    config["log_dir"], datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
)
tb_writer = SummaryWriter(log_dir=log_name)

memory = ReplayBuffer(size=config["buffer_size"])
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
    game_record = GameRecord(action_size, discount=config["discount"])

    frame = env.reset()
    game_record.observations.append(frame)

    temperature = 20 / (total_games + 20)

    while not over:
        tree = mcts.search(config["n_simulations"], frame)
        action = tree.pick_game_action(temperature=temperature)

        env.render("human")
        frame, score, over, _ = env.step(action)

        game_record.add_step(frame, action, score, tree)

        # mcts.update()

        frames += 1

    memory.save_game(game_record)
    metrics_dict = mcts.train(memory, config["n_batches"])

    for key, val in metrics_dict.items():
        tb_writer.add_scalar(key, val, total_games)

    tb_writer.add_scalar("Score", frames, total_games)

    # print(mcts.mu_net.dyna_net.fc1.weight.data)
    # print(mcts.mu_net.dyna_net.fc1.weight.grad)
    print(
        f"Completed game {total_games + 1:4} with score {frames:3}. Loss was {metrics_dict['Loss/total'].item():5.3f}."
    )
    total_games += 1


env.close()
