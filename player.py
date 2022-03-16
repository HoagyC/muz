import datetime
import math
import os
import random
import time
import yaml

import numpy as np
import ray


import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from training import GameRecord
from models import scalar_to_support, support_to_scalar, normalize
from utils import load_model
from mcts import search


@ray.remote(max_restarts=-1)
class Player:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if os.path.exists(os.path.join(self.log_dir, "data.yaml")):
            data = yaml.safe_load(open(os.path.join(self.log_dir, "data.yaml"), "r"))
            self.total_games = data["games"]
            self.total_frames = data["steps"]
        else:
            self.total_games = 0
            self.total_frames = 0

    def play(self, config, mu_net, device, log_dir, memory, env):
        total_games = 0
        total_frames = 0
        minmax = ray.get(memory.get_minmax.remote())
        start_time = time.time()
        while True:
            if "latest_model_dict.pt" in os.listdir(log_dir):
                mu_net = ray.get(
                    memory.load_model.remote(log_dir, mu_net, device=device)
                )
            else:
                memory.save_model.remote(mu_net, log_dir)

            frames = 0
            over = False
            frame = env.reset()

            if config["obs_type"] == "image":
                frame = normalize(frame)
            else:
                frame = np.array(frame)

            game_record = GameRecord(
                config=config,
                action_size=mu_net.action_size,
                init_frame=frame,
                discount=config["discount"],
                last_analysed=self.total_games,
            )

            if config["temp_time"] <= 0:
                temperature = 0
            else:
                temperature = config["temp_time"] / (total_games + config["temp_time"])
            score = 0

            if total_games % 10 == 0 and total_games > 0:
                learning_rate = config["learning_rate"] * (
                    config["learning_rate_decay"] ** total_games // 10
                )
                mu_net.init_optim(learning_rate)

            vals = []
            game_start_time = time.time()
            while not over and frames < config["max_frames"]:
                if config["obs_type"] == "image":
                    frame_input = game_record.get_last_n(config["last_n_frames"])
                else:
                    frame_input = frame
                tree = search(
                    config, mu_net, frame_input, minmax, log_dir, device=device
                )
                action = tree.pick_game_action(temperature=temperature)
                if config["debug"]:
                    if tree.children[action]:
                        print(float(tree.children[action].reward))

                if config["render"]:
                    env.render("human")

                frame, reward, over, _ = env.step(action)

                # if config["env_name"] == "CartPole-v1" and frames % 20 > 0:
                #     reward = 0

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

            memory.save_game.remote(game_record)

            self.total_games += 1
            self.total_frames += frames

            print(
                f"Game: {self.total_games:4}. Total frames: {self.total_frames:6}. "
                + f"Time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}. Score: {score:6}. "
                + f"Value mean, std: {np.mean(np.array(vals)):6.2f}, {np.std(np.array(vals)):5.2f}. "
                + f"s/move: {time_per_move:5.3f}."
            )

            with open(os.path.join(log_dir, "data.yaml"), "w+") as f:
                yaml.dump({"steps": self.total_frames, "games": self.total_games}, f)
