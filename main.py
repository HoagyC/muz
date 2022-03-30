import datetime
import os
import sys
import time
import yaml

import gym
import numpy as np
import ray

import torch
from torch.utils.tensorboard import SummaryWriter

from mcts import Trainer
from player import Player
from models import MuZeroCartNet, MuZeroAtariNet
from training import GameRecord, Memory, Reanalyser


def run(config):
    env = gym.make(config["env_name"])

    action_size = env.action_space.n

    obs_size = env.observation_space.shape

    if config["obs_type"] == "discrete":
        obs_size = obs_size[0]

    net_type_dict = {
        "discrete": MuZeroCartNet,
        "image": MuZeroAtariNet,
    }

    muzero_class = net_type_dict[config["obs_type"]]
    print(muzero_class)
    muzero_network = muzero_class(action_size, obs_size, config)
    muzero_network.init_optim(config["learning_rate"])

    if config["log_name"] == "last":
        runs = [x for x in os.listdir(config["log_dir"]) if config["env_name"] in x]
        if runs:
            config["log_name"] = sorted(runs)[-1]
        else:
            config["log_name"] = "None"
    if config["log_name"] == "None":
        config["log_name"] = (
            datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + config["env_name"]
        )

    print(f"Logging to '{config['log_name']}'")

    log_dir = os.path.join(config["log_dir"], config["log_name"])

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if "data.yaml" not in os.listdir(log_dir):
        init_dict = {"games": 0, "steps": 0}
        yaml.dump(init_dict, open(os.path.join(log_dir, "data.yaml"), "w+"))

    tb_writer = SummaryWriter(log_dir=log_dir)

    memory_gpus = 0.1 if torch.cuda.is_available() else 0
    memory = Memory.options(num_cpus=0.1, num_gpus=memory_gpus).remote(config, log_dir)
    # open muz implementation uses a GameHistory class
    # with observation_history, action_history, reward_history
    # to_play which is who is to play in case it's a multiplayer, turn-based game
    # also stores search stats ie the number of times each child/action node was visited
    # which becomes the policy
    # and the value which is the average value as calculated by the MCTS
    # can also store the reanalysed predicted root values

    start_time = time.time()
    scores = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    player = Player.options(num_cpus=0.2).remote(log_dir=log_dir)

    train_cpus = 0 if torch.cuda.is_available() else 0.2
    train_gpus = 0.9 if torch.cuda.is_available() else 0
    trainer = Trainer.options(num_cpus=train_cpus, num_gpus=train_gpus).remote()

    player.play.remote(
        config=config,
        mu_net=muzero_network,
        log_dir=log_dir,
        device=torch.device("cpu"),
        memory=memory,
        env=env,
    )

    trainer.train.remote(
        mu_net=muzero_network,
        memory=memory,
        config=config,
        device=device,
        log_dir=log_dir,
    )

    if config["reanalyse"]:
        analyser = Reanalyser.options(num_cpus=0.1).remote(
            config=config, log_dir=log_dir
        )
        analyser.reanalyse.remote(mu_net=muzero_network, memory=memory)

    while True:
        time.sleep(10)

        # metrics_dict = train(memory, config["n_batches"], device=device)
        # time_per_batch = (time.time() - train_start_time) / config["n_batches"]

        # for key, val in metrics_dict.items():
        #     tb_writer.add_scalar(key, val, total_games)

        # tb_writer.add_scalar("Score", score, total_games)

        # scores.append(score)
        # total_games += 1
        # total_frames += frames

        # print(
        #     f"Game: {total_games:4}. Total frames: {total_frames:6}. "
        #     + f"Time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}. Score: {score:6}. "
        #     + f"Loss: {metrics_dict['Loss/total'].item():7.2f}. "
        #     + f"Value mean, std: {np.mean(np.array(vals)):6.2f}, {np.std(np.array(vals)):5.2f}. "
        #     + f"s/move: {time_per_move:5.3f}. s/batch: {time_per_batch:6.3f}."
        # )

    env.close()
    return scores


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            config = yaml.safe_load(open("config-" + sys.argv[1] + ".yaml", "r"))
        except:
            raise ValueError(f"No config file for game '{sys.argv[1]}'")
    else:
        print("Specify game name")

    if len(sys.argv) > 2 and sys.argv[2] == "colab":
        config["render"] = False
        config["debug"] = False
        config["log_dir"] = "/content/gdrive/My Drive/muz"
        config["batch_size"] = 50

    run(config)
