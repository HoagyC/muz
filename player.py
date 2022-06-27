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

from memory import GameRecord
from models import scalar_to_support, support_to_scalar
from mcts import search


@ray.remote
class Player:
    def __init__(self, log_dir, writer=None):
        self.log_dir = log_dir
        self.writer = writer

    def play(self, config, mu_net, device, log_dir, memory, buffer, env):
        minmax = ray.get(memory.get_minmax.remote())
        start_time = time.time()

        while not ray.get(memory.is_finished.remote()):
            data = ray.get(memory.get_data.remote())
            self.total_games = data["games"]
            self.total_frames = data["frames"]

            if "latest_model_dict.pt" in os.listdir(log_dir):
                mu_net = ray.get(memory.load_model.remote(log_dir, mu_net))
            else:
                memory.save_model.remote(mu_net, log_dir)

            frames = 0
            over = False
            frame = env.reset()

            game_record = GameRecord(
                config=config,
                action_size=mu_net.action_size,
                init_frame=frame,
                discount=config["discount"],
                last_analysed=self.total_games,
            )

            if self.total_frames < config["temp1"]:
                temperature = 1
            elif self.total_frames < config["temp2"]:
                temperature = 0.5
            else:
                temperature = 0
            score = 0

            if self.total_games % 10 == 0 and self.total_games > 0:
                learning_rate = config["learning_rate"] * (
                    config["learning_rate_decay"] ** self.total_games // 10
                )
                mu_net.init_optim(learning_rate)

            vals = []
            game_start_time = time.time()
            while not over and frames < config["max_frames"]:
                if config["obs_type"] == "image":
                    frame_input = game_record.get_last_n(
                        n=config["last_n_frames"], pos=-1
                    )
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

                game_record.add_step(frame, action, reward, tree)

                frames += 1
                score += reward
                vals.append(float(tree.val_pred))

            time_per_move = (time.time() - game_start_time) / frames

            game_record.add_priorities(n_steps=config["reward_depth"])
            stats = ray.get(memory.get_data.remote())
            if self.writer:
                self.writer.add_scalar("score", score, stats["frames"])

            game_data = ray.get(memory.done_game.remote(frames, score))
            buffer.save_game.remote(game_record, frames, score, game_data)

            print(
                f"Game: {self.total_games + 1:4}. Total frames: {self.total_frames + frames:6}. "
                + f"Time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}. Score: {score:6}. "
                + f"Value mean, std: {np.mean(np.array(vals)):6.2f}, {np.std(np.array(vals)):5.2f}. "
                + f"s/move: {time_per_move:5.3f}."
            )
