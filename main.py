import datetime
import importlib
import os
import sys
import time
import yaml


import gym
import numpy as np
import ray

import torch
from torch.utils.tensorboard import SummaryWriter

from trainer import Trainer
from buffer import Buffer
from player import Player
from models import MuZeroCartNet, MuZeroAtariNet, TestNet
from memory import GameRecord, Memory
from reanalyser import Reanalyser
from envs import testgame_env, testgamed_env, atari_env, cartpole_env


ENV_DICT = {"image": atari_env, "cartpole": cartpole_env, "test": testgame_env}
NET_DICT = {
    "cartpole": MuZeroCartNet,
    "image": MuZeroAtariNet,
    "test": TestNet,
}


def run(config, train_only=False):
    env = ENV_DICT[config["obs_type"]].make_env(config)
    config["full_image_size"] = env.full_image_size

    action_size = env.action_space.n
    config["action_size"] = action_size

    obs_size = config["obs_size"]
    print(obs_size)
    if config["obs_type"] in ["cartpole", "test"]:
        obs_size = obs_size[0]

    print(f"Observation size: {obs_size}")

    muzero_class = NET_DICT[config["obs_type"]]
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

    os.makedirs("buffers", exist_ok=True)

    print(f"Logging to '{config['log_name']}'")

    log_dir = os.path.join(config["log_dir"], config["log_name"])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if "data.yaml" not in os.listdir(log_dir):
        init_dict = {"games": 0, "steps": 0, "batches": 0}
        yaml.dump(init_dict, open(os.path.join(log_dir, "data.yaml"), "w+"))

    tb_writer = SummaryWriter(log_dir=log_dir)

    workers = []

    use_cuda = config["try_cuda"] and torch.cuda.is_available()

    buffer_gpus = 0.1 if use_cuda else 0
    memory = Memory.options(num_cpus=0.1).remote(config, log_dir)
    buffer = Buffer.options(num_cpus=0.1, num_gpus=buffer_gpus).remote(config, memory)

    # open muz implementation uses a GameHistory class
    # with observation_history, action_history, reward_history
    # to_play which is who is to play in case it's a multiplayer, turn-based game
    # also stores search stats ie the number of times each child/action node was visited
    # which becomes the policy
    # and the value which is the average value as calculated by the MCTS
    # can also store the reanalysed predicted root values

    start_time = time.time()
    scores = []

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Training on device: {device}")

    player = Player.options(num_cpus=0.3).remote(log_dir=log_dir)

    train_cpus = 0 if use_cuda else 0.1
    train_gpus = 0.9 if use_cuda else 0
    trainer = Trainer.options(num_cpus=train_cpus, num_gpus=train_gpus).remote()

    if not train_only:
        workers.append(
            player.play.remote(
                config=config,
                mu_net=muzero_network,
                log_dir=log_dir,
                device=torch.device("cpu"),
                memory=memory,
                buffer=buffer,
                env=env,
            )
        )

    workers.append(
        trainer.train.remote(
            mu_net=muzero_network,
            memory=memory,
            buffer=buffer,
            config=config,
            device=device,
            log_dir=log_dir,
        )
    )

    if config["reanalyse"]:
        print("adding reanalyser")
        analyser = Reanalyser.options(num_cpus=0.1).remote(
            config=config, log_dir=log_dir
        )
        workers.append(
            analyser.reanalyse.remote(
                mu_net=muzero_network, memory=memory, buffer=buffer
            )
        )

    ray.get(workers)

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
    scores = ray.get(memory.get_scores.remote())
    return scores


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            config_path = os.path.join("configs", "config-" + sys.argv[1] + ".yaml")
            config = yaml.safe_load(open(config_path, "r"))
        except FileNotFoundError:
            print(f"No config file for game '{sys.argv[1]}'")
    else:
        print("Specify game name")

    if len(sys.argv) > 2 and sys.argv[2] == "colab":
        config["render"] = False
        config["channel_list"] = [128, 256]
        config["debug"] = False
        config["log_dir"] = "/content/gdrive/My Drive/muz"
        config["batch_size"] = 128  # 256 causes insufficient memory errors
        config["repr_channels"] = 128

    run(config)
