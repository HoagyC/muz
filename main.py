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

from mcts import MCTS
from models import MuZeroCartNet, MuZeroAtariNet
from training import GameRecord, ReplayBuffer


def run(config):
    env = gym.make(config["env_name"])

    action_size = env.action_space.n

    obs_size = env.observation_space.shape

    if config["obs_type"] == "discrete":
        obs_size = obs_size[0]

    net_type_dict = {
        "CartPole-v1": MuZeroCartNet,
        "BreakoutNoFrameskip-v4": MuZeroAtariNet,
        "Freeway-v0": MuZeroAtariNet,
    }

    muzero_class = net_type_dict[config["env_name"]]
    print(muzero_class)
    muzero_network = muzero_class(action_size, obs_size, config)
    learning_rate = config["learning_rate"]
    muzero_network.init_optim(learning_rate)

    if config["log_name"] == "None":
        config["log_name"] = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    else:
        muzero_network

    log_dir = os.path.join(config["log_dir"], config["log_name"])
    tb_writer = SummaryWriter(log_dir=log_dir)

    mcts = MCTS(
        action_size=action_size, mu_net=muzero_network, config=config, log_dir=log_dir
    )

    memory = ReplayBuffer(config)
    # open muz implementation uses a GameHistory class
    # with observation_history, action_history, reward_history
    # to_play which is who is to play in case it's a multiplayer, turn-based game
    # also stores search stats ie the number of times each child/action node was visited
    # which becomes the policy
    # and the value which is the average value as calculated by the MCTS
    # can also store the reanalysed predicted root values

    total_games = 0
    scores = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    while total_games < config["max_games"]:
        try:
            frames = 0
            over = False
            frame = env.reset()

            game_record = GameRecord(
                config=config,
                action_size=action_size,
                init_frame=frame,
                discount=config["discount"],
                last_analysed=total_games,
            )
            if config["temp_time"] <= 0:
                temperature = 0
            else:
                temperature = config["temp_time"] / (total_games + config["temp_time"])
            score = 0

            if total_games % 10 == 0 and total_games > 0:
                learning_rate = learning_rate * config["learning_rate_decay"]
                mcts.mu_net.init_optim(learning_rate)

            vals = []
            while not over and frames < config["max_frames"]:
                tree = mcts.search(config["n_simulations"], frame)
                action = tree.pick_game_action(temperature=temperature)

                if config["render"]:
                    env.render("human")

                frame, reward, over, _ = env.step(action)

                game_record.add_step(frame, action, reward, tree)

                # mcts.update()

                frames += 1
                score += reward
                vals.append(tree.val_pred)

            # print("Root value std: ", np.std(np.array(vals)))

            game_record.add_priorities(n_steps=config["reward_depth"])
            if total_games > 1 and config["reanalyse"]:
                memory.reanalyse(mcts, current_game=total_games)
            memory.save_game(game_record)
            metrics_dict = mcts.train(memory, config["n_batches"], device=device)

            for key, val in metrics_dict.items():
                tb_writer.add_scalar(key, val, total_games)

            tb_writer.add_scalar("Score", score, total_games)

            print(
                f"Completed game {total_games + 1:4} with score {score:6}. Loss was {metrics_dict['Loss/total'].item():5.2f}."
            )
            scores.append(score)
            total_games += 1

        except KeyboardInterrupt:
            breakpoint()
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

    run(config)
