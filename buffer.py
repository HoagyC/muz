import datetime
import os
import pickle

from functools import reduce
from operator import add

import torch
import ray

import numpy as np


@ray.remote
class Buffer:
    def __init__(self, config, memory):
        self.config = config
        self.memory = memory

        self.last_time = datetime.datetime.now()  # Used if profiling speed of batching

        self.size = config["buffer_size"]  # How many game records to store
        self.priority_alpha = config["priority_alpha"]
        self.tau = config["tau"]

        # List of start points of each game if the whole buffer were concatenated
        self.game_starts_list = []

        if self.config["load_buffer"] and os.path.exists(
            os.path.join("buffers", config["env_name"])
        ):
            self.load_buffer()
        else:
            self.buffer = []
            self.buffer_ndxs = []

        self.prioritized_replay = config["priority_replay"]

        self.priorities = []

    def save_buffer(self):
        with open(os.path.join("buffers", self.config["env_name"]), "wb") as f:
            pickle.dump((self.buffer, self.buffer_ndxs), f)

    def load_buffer(self):
        with open(os.path.join("buffers", self.config["env_name"]), "rb") as f:
            self.buffer, self.buffer_ndxs = pickle.load(f)
        self.update_stats()

    def update_vals(self, ndx, vals):
        try:
            buf_ndx = self.buffer_ndxs.index(ndx)
            self.buffer[buf_ndx].values = vals
            total_games = ray.get(self.memory.get_total_games.remote())
            self.buffer[buf_ndx].last_analysed = total_games
        except ValueError:
            print(f"No buffer item with index {ndx}")

    def add_priorities(self, ndx, reanalysing=False):
        try:
            buf_ndx = self.buffer_ndxs.index(ndx)
            self.buffer[buf_ndx].add_priorities(
                n_steps=self.config["reward_depth"], reanalysing=reanalysing
            )
        except ValueError:
            print(f"No buffer item with index {ndx}")

    def update_stats(self):
        # Maintain stats for the total length of all games in the buffer
        # and where each game would begin if all games were concatenated
        # so that each step of each game can be uniquely indexed

        lengths = [len(x.values) for x in self.buffer]
        self.game_starts_list = [sum(lengths[0:i]) for i in range(len(self.buffer))]
        self.total_vals = sum(lengths)
        self.priorities = reduce(add, [x.priorities for x in self.buffer], [])
        self.priorities = [float(p**self.priority_alpha) for p in self.priorities]
        sum_priorities = sum(self.priorities)
        self.priorities = [p / sum_priorities for p in self.priorities]

    def get_batch(self, batch_size=40, device=torch.device("cpu")):
        self.print_timing("start")
        batch = []

        # Get a random list of points across the length of the buffer to take training examples
        if self.prioritized_replay:
            probabilities = self.priorities
        else:
            probabilities = None

        if probabilities and len(probabilities) != self.total_vals:
            breakpoint()
        start_vals = np.random.choice(
            list(range(self.total_vals)), size=batch_size, p=probabilities
        )
        self.print_timing("get ndxs")

        images_l = []
        actions_l = []
        target_values_l = []
        target_rewards_l = []
        target_policies_l = []
        weights_l = []
        depths_l = []

        for val in start_vals:
            # Get the index of the game in the buffer (game_ndx) and a location in the game (frame_ndx)
            game_ndx, frame_ndx = self.get_ndxs(val)

            game = self.buffer[game_ndx]

            reward_depth = self.get_reward_depth(
                val,
                self.tau,
                self.config["total_training_steps"],
                self.config["reward_depth"],
            )

            # Gets a series of actions, values, rewards, policies, up to a depth of rollout_depth
            (
                images,
                actions,
                target_values,
                target_rewards,
                target_policies,
                depth,
            ) = game.make_target(
                frame_ndx,
                reward_depth=self.config["reward_depth"],
                rollout_depth=self.config["rollout_depth"],
            )

            # Add tuple to batch
            if self.prioritized_replay:
                weight = 1 / self.priorities[val]
            else:
                weight = 1

            images_l.append(images)
            actions_l.append(actions)
            target_values_l.append(target_values)
            target_rewards_l.append(target_rewards)
            target_policies_l.append(target_policies)

            weights_l.append(weight)
            depths_l.append(depth)
        self.print_timing("make_lists")

        images_t = torch.tensor(np.stack(images_l, axis=0), dtype=torch.float32)
        actions_t = torch.tensor(np.stack(actions_l, axis=0), dtype=torch.int64)
        target_values_t = torch.tensor(
            np.stack(target_values_l, axis=0), dtype=torch.float32
        )
        target_policies_t = torch.tensor(
            np.stack(target_policies_l, axis=0), dtype=torch.float32
        )
        target_rewards_t = torch.tensor(
            np.stack(target_rewards_l, axis=0), dtype=torch.float32
        )
        weights_t = torch.tensor(weights_l)
        weights_t = weights_t / max(weights_t)
        self.print_timing("make_tensors")

        return (
            images_t.to(device),
            actions_t.to(device),
            target_values_t.to(device),
            target_rewards_t.to(device),
            target_policies_t.to(device),
            weights_t.to(device),
            depths_l,
        )

    def get_buffer_ndx(self, ndx):
        buf_ndx = self.buffer_ndxs.index(ndx)
        return self.buffer[buf_ndx]

    def get_buffer_len(self):
        return len(self.buffer)

    def get_buffer(self):
        return self.buffer

    def get_buffer_ndxs(self):
        return self.buffer_ndxs

    def get_reanalyse_probabilities(self):
        total_games = ray.get(self.memory.get_total_games.remote())
        p = np.array([total_games - x.last_analysed for x in self.buffer]).astype(
            np.float32
        )
        if sum(p) > 0:
            return p / sum(p)
        else:
            return np.array([])

    def save_game(self, game, n_frames, score, game_data):
        # If reached the max size, remove the oldest GameRecord, and update stats accordingly
        while len(self.buffer) >= self.size:
            self.buffer.pop(0)
            self.buffer_ndxs.pop(0)

        self.buffer.append(game)
        self.update_stats()
        self.save_buffer()
        self.buffer_ndxs.append(game_data["games"] - 1)

    def get_ndxs(self, val):
        if val >= self.total_vals:
            raise ValueError("Trying to get a value beyond the length of the buffer")

        # Assumes len_list is sorted, gets the last entry in starts_list which is below val
        # by iterating through game_starts_list until one is above val, at which point
        # it returns the previous value in game_starts_list
        # and the position in the game is gap between the game's start position and val
        for i, l in enumerate(self.game_starts_list):
            if l > val:
                return i - 1, val - self.game_starts_list[i - 1]
        return len(self.buffer) - 1, val - self.game_starts_list[-1]

    def get_reward_depth(self, val, tau=0.3, total_steps=100_000, max_depth=5):
        if self.config["off_policy_correction"]:
            # Varying reward depth depending on the length of time since the trajectory was generated
            # Follows the formula in A.4 of EfficientZero paper
            steps_ago = self.total_vals - val
            depth = max_depth - np.floor((steps_ago / (tau * total_steps)))
            depth = int(np.clip(depth, 1, max_depth))
        else:
            depth = max_depth
        return depth

    def print_timing(self, tag, min_time=0.05):
        if self.config["get_batch_profiling"]:
            now = datetime.datetime.now()
            print(f"{tag:20} {now - self.last_time}")
            self.last_time = now
