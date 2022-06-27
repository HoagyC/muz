import os
import pickle
import time
import yaml

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
            target_rewards = (
                []
            )  # If using value prefix then these are cumulative rewards
            target_values = []
            target_policies = []

            game_len = len(self.search_stats)

            # Make sure we don't try to roll out beyond end of game
            actual_rollout_depth = min(rollout_depth, game_len - ndx)

            for i in range(actual_rollout_depth):
                if self.config["value_prefix"]:
                    target_rewards.append(sum(self.rewards[ndx : ndx + i + 1]))
                else:
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
            if len(target_rewards_a) > 5:
                print(target_rewards_a, target_rewards_a.shape)

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
        self.session_start_time = time.time()
        self.log_dir = log_dir
        self.total_vals = 0  # How many total steps are stored

        data = yaml.safe_load(open(os.path.join(self.log_dir, "data.yaml"), "r"))
        self.total_games = data["games"]
        self.total_frames = data["steps"]
        self.total_batches = data["batches"]

        self.reward_depth = config["reward_depth"]
        self.total_training_steps = config["total_training_steps"]
        self.rollout_depth = config["rollout_depth"]
        self.minmax = MinMax()
        self.finished = False
        self.game_stats = []

    def get_data(self):
        return {
            "games": self.total_games,
            "frames": self.total_frames,
            "batches": self.total_batches,
        }

    def get_minmax(self):
        return self.minmax

    def save_model(self, model, log_dir):
        path = os.path.join(log_dir, "latest_model_dict.pt")
        print("saving", next(model.pred_net.parameters()).device)
        torch.save(model.state_dict(), path)

    def load_model(self, log_dir, model):
        print("loading from, ", log_dir)
        it = time.time()
        path = os.path.join(log_dir, "latest_model_dict.pt")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        else:
            print(f"no dict to load at {path}")
        print("loaded memory")

        return model

    def done_game(self, n_frames, score):
        self.total_games += 1
        self.total_frames += n_frames
        self.save_core_stats()
        self.game_stats.append(
            {
                "total games": self.total_games,
                "score": score,
                "total frames": self.total_frames,
                "elapsed time": self.get_elapsed_time(),
                "total batches": self.total_batches,
            }
        )
        if (
            self.total_games >= self.config["max_games"]
            or self.total_frames >= self.config["max_total_frames"]
        ):
            print("Reached designated end of run, sending shutdown message")
            self.finished = True

        return self.get_data()

    def done_batch(self):
        self.total_batches += 1
        self.save_core_stats()

    def save_core_stats(self, total_batches=None):
        stat_dict = {
            "steps": self.total_frames,
            "games": self.total_games,
            "batches": self.total_batches,
        }
        with open(os.path.join(self.log_dir, "data.yaml"), "w+") as f:
            yaml.dump(stat_dict, f)

    def is_finished(self):
        return self.finished

    def get_scores(self):
        return self.game_stats

    def get_total_games(self):
        return self.total_games

    def get_elapsed_time(self):
        return time.time() - self.session_start_time
