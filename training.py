import os
import pickle
import time
import yaml

from functools import reduce
from operator import add
from random import randrange, random

import numpy as np
import torch
import ray

from mcts import search, MinMax
from utils import convert_to_int, convert_from_int


class GameRecord:
    # This class stores the relevant history of a single game
    def __init__(
        self,
        config,
        action_size: int,
        init_frame,
        discount: float = 0.8,
        last_analysed=0,
    ):
        self.config = config
        self.action_size = action_size  # Number of available actions
        self.discount = discount  # Discount rate to be applied to future rewards

        # List of states received from the game
        self.observations = [convert_to_int(init_frame, self.config["obs_type"])]
        # List of actions taken in the game
        self.actions = []
        # List of rewards received after taking action in game (single step)
        self.rewards = []
        # List of the number of times each possible action was sampled at the root of the search tree
        self.search_stats = []
        # List of *estimated* total future reward from the node,
        # as measured by the average value at the root of the search tree
        self.values = []

        self.priorities = []
        self.last_analysed = last_analysed

    def add_step(self, obs: np.ndarray, action: int, reward: int, root):
        # Root is a TreeNode object at the root of the search tree for the given state

        # Note that when taking a step you get the action, reward and new observation
        # but for training purposes we want to connect the reward with the action and *old* observation.
        # We therefore add the first frame when we initialize the class, so connected frame-action-reward
        # tuples have the same index

        int_obs = convert_to_int(obs, self.config["obs_type"])
        self.observations.append(int_obs)
        self.actions.append(action)
        self.rewards.append(float(reward))

        self.search_stats.append([c.num_visits if c else 0 for c in root.children])
        self.values.append(float(root.average_val))

    def get_last_n(self, n=None, pos=-1):
        if not n:
            n = self.config["last_n_frames"]

        if pos == -1:
            last_n = np.concatenate(self.observations[-n:], axis=0)
            last_n_actions = self.actions[-n:]
        else:
            last_n = np.concatenate(
                self.observations[max(0, pos - n + 1) : pos + 1], axis=0
            )
            last_n_actions = self.actions[max(0, pos - n + 1) : pos + 1]

        last_n = convert_from_int(last_n, self.config["obs_type"])

        last_n_actions = [-1] * (n - len(last_n_actions)) + last_n_actions
        last_n_actions = torch.tensor(last_n_actions, dtype=torch.float32) + 1
        last_n_actions /= self.action_size

        action_planes = torch.ones(last_n.shape[1:])
        action_planes = torch.einsum("hw, a->ahw", [action_planes, last_n_actions])

        pad_len = (n * 3) - last_n.shape[0]
        pad_a = np.zeros((pad_len, *last_n.shape[1:]))
        last_n = np.concatenate((action_planes, pad_a, last_n), axis=0)
        return last_n

    def add_priorities(self, n_steps=5, reanalysing=False):
        # Add this in case trainer has restarted before adding priorities
        if len(self.values) != len(self.priorities):
            self.priorities = []
            reanalysing = False

        for i, r in enumerate(self.values):
            if i + n_steps < len(self.values):
                value_target = self.values[i + n_steps]
            else:
                value_target = 0
            for j in range(n_steps):
                if len(self.values) < i + j:
                    value_target += self.values[i + j]
                else:
                    break

            priority = abs(r - value_target)
            if reanalysing:
                self.priorities[i] = priority
            else:
                self.priorities.append(priority)

    def pad_target(self, target_l, pad_len):
        target_a = np.array(target_l)
        pad_l = [(0, pad_len)] + [(0, 0)] * (target_a.ndim - 1)
        target_a = np.pad(target_a, pad_l, mode="constant")
        return target_a

    def make_target(self, ndx: int, reward_depth: int = 5, rollout_depth: int = 3):
        # ndx is where in the record of the game we start
        # reward_depth is how far into the future we use the actual reward - beyond this we use predicted value

        # rollout_depth is how many iterations of the dynamics function we take,
        # from where we apply the represent function
        # it acts like an additional dimension of batching when we make the target but the crucial difference is that
        # when we train, we will use a hidden state by repeated application of the dynamics function from the init_image
        # rather than creating a new hidden state from the game obs at the time
        # this is necessary to train the dynamics function to give useful information for predicting the value

        with torch.no_grad():
            # We get a reward, value and policy for each step in the rollout of our hidden state
            target_rewards = []
            target_values = []
            target_policies = []

            game_len = len(self.search_stats)

            # Make sure we don't try to roll out beyond end of game
            actual_rollout_depth = min(rollout_depth, game_len - ndx)

            for i in range(actual_rollout_depth):
                target_rewards.append(self.rewards[ndx + i])

                # If we have an estimated value at the current index + reward_depth
                # then this is our base value (after discounting)
                # else we start at 0
                bootstrap_index = ndx + reward_depth + i

                if bootstrap_index < len(self.values):
                    target_value = self.values[bootstrap_index] * (
                        self.discount**reward_depth
                    )

                else:
                    target_value = 0

                # We then add intermediate rewards, breaking if we hit the end of the game
                for j in range(reward_depth):
                    if ndx + i + j < len(self.rewards):
                        target_value += self.rewards[ndx + i + j] * (self.discount**j)
                    else:
                        break

                target_values.append(target_value)

                total_searches = sum(self.search_stats[ndx + i])

                # The target policy is the fraction of searches which went down each action at the root of the tree
                target_policies.append(
                    [x / total_searches for x in self.search_stats[ndx + i]]
                )

            # include all observations for consistency loss
            if self.config["obs_type"] == "image":
                images = [
                    self.get_last_n(pos=x)
                    for x in range(ndx, ndx + actual_rollout_depth)
                ]
            else:
                images = self.observations[ndx : ndx + actual_rollout_depth]
                images = [convert_from_int(x, self.config["obs_type"]) for x in images]

            actions = self.actions[ndx : ndx + actual_rollout_depth]

            unused_rollout = rollout_depth - actual_rollout_depth

            images_a = self.pad_target(images, unused_rollout)
            actions_a = self.pad_target(actions, unused_rollout)
            target_policies_a = self.pad_target(target_policies, unused_rollout)
            target_values_a = self.pad_target(target_values, unused_rollout)
            target_rewards_a = self.pad_target(target_rewards, unused_rollout)

            return (
                images_a,
                actions_a,
                target_values_a,
                target_rewards_a,
                target_policies_a,
                actual_rollout_depth,
            )


@ray.remote
class Memory:
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = log_dir
        self.size = config["buffer_size"]  # How many game records to store
        self.total_vals = 0  # How many total steps are stored

        data = yaml.safe_load(open(os.path.join(self.log_dir, "data.yaml"), "r"))
        self.total_games = data["games"]
        self.total_frames = data["steps"]
        self.total_batches = data["batches"]

        self.prioritized_replay = config["priority_replay"]
        self.priority_alpha = config["priority_alpha"]

        # List of start points of each game if the whole buffer were concatenated
        self.game_starts_list = []

        self.reward_depth = config["reward_depth"]
        self.rollout_depth = config["rollout_depth"]
        self.priorities = []
        self.minmax = MinMax()
        if os.path.exists(os.path.join("buffers", config["env_name"])):
            self.load_buffer()
            print(self.buffer_ndxs)
        else:
            self.buffer = []
            self.buffer_ndxs = []

    def save_buffer(self):
        with open(os.path.join("buffers", self.config["env_name"]), "wb") as f:
            pickle.dump((self.buffer, self.buffer_ndxs), f)

    def load_buffer(self):
        with open(os.path.join("buffers", self.config["env_name"]), "rb") as f:
            self.buffer, self.buffer_ndxs = pickle.load(f)
        self.update_stats()

    def get_data(self):
        return {
            "games": self.total_games,
            "frames": self.total_frames,
            "batches": self.total_batches,
        }

    def get_buffer(self):
        return self.buffer

    def get_buffer_ndxs(self):
        return self.buffer_ndxs

    def add_priorities(self, ndx, reanalysing=False):
        try:
            buf_ndx = self.buffer_ndxs.index(ndx)
            self.buffer[buf_ndx].add_priorities(
                n_steps=self.config["reward_depth"], reanalysing=reanalysing
            )
        except ValueError:
            print(f"No buffer item with index {ndx}")

    def get_reanalyse_probabilities(self):
        p = np.array([self.total_games - x.last_analysed for x in self.buffer]).astype(
            np.float32
        )
        if sum(p) > 0:
            return p / sum(p)
        else:
            return np.array([])

    def update_vals(self, ndx, vals):
        try:
            buf_ndx = self.buffer_ndxs.index(ndx)
            self.buffer[buf_ndx].values = vals
            self.buffer[buf_ndx].last_analysed = self.total_games
        except ValueError:
            print(f"No buffer item with index {ndx}")

    def get_buffer_ndx(self, ndx):
        buf_ndx = self.buffer_ndxs.index(ndx)
        return self.buffer[buf_ndx]

    def get_buffer_len(self):
        return len(self.buffer)

    def get_minmax(self):
        return self.minmax

    def save_model(self, model, log_dir):
        path = os.path.join(log_dir, "latest_model_dict.pt")
        torch.save(model.state_dict(), path)

    def load_model(self, log_dir, model, device=torch.device("cpu")):
        it = time.time()
        path = os.path.join(log_dir, "latest_model_dict.pt")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"no dict to load at {path}")

        return model

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

    def done_batch(self):
        self.total_batches += 1
        self.save_core_stats()

    def save_core_stats(self, total_batches=None):
        with open(os.path.join(self.log_dir, "data.yaml"), "w+") as f:
            yaml.dump(
                {
                    "steps": self.total_frames,
                    "games": self.total_games,
                    "batches": self.total_batches,
                },
                f,
            )

    def save_game(self, game, n_frames):
        # If reached the max size, remove the oldest GameRecord, and update stats accordingly
        while len(self.buffer) >= self.size:
            print(self.buffer_ndxs)
            self.buffer.pop(0)
            self.buffer_ndxs.pop(0)

        self.buffer.append(game)
        self.buffer_ndxs.append(self.total_games)
        self.update_stats()
        self.total_games += 1
        self.total_frames += n_frames
        self.save_buffer()
        self.save_core_stats

    def get_batch(self, batch_size=40):
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

        images_l = []
        actions_l = []
        target_values_l = []
        target_rewards_l = []
        target_policies_l = []
        weights_l = []
        depths_l = []

        for val in start_vals:
            # Get the index of the game in the buffer (buf_ndx) and a location in the game (game_ndx)
            buf_ndx, game_ndx = self.get_ndxs(val)

            game = self.buffer[buf_ndx]

            # Gets a series of actions, values, rewards, policies, up to a depth of rollout_depth
            (
                images,
                actions,
                target_values,
                target_rewards,
                target_policies,
                depth,
            ) = game.make_target(
                game_ndx,
                reward_depth=self.reward_depth,
                rollout_depth=self.rollout_depth,
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

        images_t = torch.tensor(np.stack(images_l), dtype=torch.float32)
        actions_t = torch.tensor(np.stack(actions_l), dtype=torch.int64)
        target_values_t = torch.tensor(np.stack(target_values_l), dtype=torch.float32)
        target_policies_t = torch.tensor(
            np.stack(target_policies_l), dtype=torch.float32
        )
        target_rewards_t = torch.tensor(np.stack(target_rewards_l), dtype=torch.float32)
        weights_t = torch.tensor(weights_l)
        weights_t = weights_t / max(weights_t)

        return (
            images_t,
            actions_t,
            target_values_t,
            target_rewards_t,
            target_policies_t,
            weights_t,
            depths_l,
        )

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


@ray.remote
class Reanalyser:
    def __init__(self, config, log_dir, device=torch.device("cpu")):
        self.device = device
        self.config = config
        self.log_dir = log_dir

    def reanalyse(self, mu_net, memory):
        while True:
            if "latest_model_dict.pt" in os.listdir(self.log_dir):
                mu_net = ray.get(
                    memory.load_model.remote(self.log_dir, mu_net, device=self.device)
                )

            # No point reanalysing until there are multiple games in the history
            while True:
                buffer_len = ray.get(memory.get_buffer_len.remote())
                train_stats = ray.get(memory.get_data.remote())
                current_game = train_stats["games"]
                if buffer_len >= 1 and current_game >= 2:
                    break

                time.sleep(1)

            mu_net.train()
            mu_net = mu_net.to(self.device)

            p = ray.get(memory.get_reanalyse_probabilities.remote())

            if len(p) > 0:
                ndxs = ray.get(memory.get_buffer_ndxs.remote())
                ndx = np.random.choice(ndxs, p=p)
                game_rec = ray.get(memory.get_buffer_ndx.remote(ndx))
                minmax = ray.get(memory.get_minmax.remote())

                vals = game_rec.values

                for i in range(len(game_rec.observations) - 1):
                    if self.config["obs_type"] == "image":
                        obs = game_rec.get_last_n(pos=i)
                    else:
                        obs = convert_from_int(
                            game_rec.observations[i], self.config["obs_type"]
                        )

                    new_root = search(
                        config=self.config,
                        mu_net=mu_net,
                        current_frame=obs,
                        minmax=minmax,
                        log_dir=self.log_dir,
                        device=torch.device("cpu"),
                    )
                    vals[i] = new_root.average_val

                memory.update_vals.remote(ndx=ndx, vals=vals)
                memory.add_priorities.remote(ndx=ndx, reanalysing=True)
                print(f"Reanalysed game {ndx}")
            else:
                time.sleep(5)
