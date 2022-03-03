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
from models import MuZeroCartNet, MuZeroAtariNet, normalize
from training import GameRecord, ReplayBuffer


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
    learning_rate = config["learning_rate"]
    muzero_network.init_optim(learning_rate)

    if config["log_name"] == "None":
        config["log_name"] = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

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
    total_frames = 0
    start_time = time.time()
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
            game_start_time = time.time()
            while not over and frames < config["max_frames"]:
                tree = mcts.search(config["n_simulations"], frame, device=device)
                action = tree.pick_game_action(temperature=temperature)

                if config["render"]:
                    env.render("human")

                frame, reward, over, _ = env.step(action)

                if config["env_name"] == "CartPole-v1" and frames % 20 > 0:
                    reward = 0

                if config["obs_type"] == "image":
                    frame = normalize(frame)
                else:
                    frame = np.array(frame)

                game_record.add_step(frame, action, reward, tree)

                # mcts.update()

                frames += 1
                score += reward
                vals.append(float(tree.val_pred))

            time_per_move = (time.time() - game_start_time) / frames

            game_record.add_priorities(n_steps=config["reward_depth"])
            if total_games > 1 and config["reanalyse"]:
                memory.reanalyse(
                    mcts, current_game=total_games, n=config["reanalyse_n"]
                )
            memory.save_game(game_record)

            train_start_time = time.time()
            metrics_dict = mcts.train(memory, config["n_batches"], device=device)
            time_per_batch = (time.time() - train_start_time) / config["n_batches"]

            for key, val in metrics_dict.items():
                tb_writer.add_scalar(key, val, total_games)

            tb_writer.add_scalar("Score", score, total_games)

            scores.append(score)
            total_games += 1
            total_frames += frames

            print(
                f"Game: {total_games:4}. Total frames: {total_frames:6}. "
                + f"Time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}. Score: {score:6}. "
                + f"Loss: {metrics_dict['Loss/total'].item():7.2f}. "
                + f"Value mean, std: {np.mean(np.array(vals)):6.2f}, {np.std(np.array(vals)):5.2f}. "
                + f"s/move: {time_per_move:5.3f}. s/batch: {time_per_batch:6.3f}."
            )

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
