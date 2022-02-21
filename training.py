from functools import reduce
from operator import add
from random import randrange, random

import numpy as np
import torch
import ray


class GameRecord:
    # This class stores the relevant history of a single game
    def __init__(self, config, action_size: int, init_frame, discount: float = 0.8):
        self.config = config
        self.action_size = action_size  # Number of available actions
        self.discount = discount  # Discount rate to be applied to future rewards

        # List of states received from the game
        self.observations = [init_frame]
        # List of actions taken in the game
        self.actions = []
        # List of rewards received after taking action in game (single step)
        self.rewards = []
        # List of the number of times each possible action was sampled at the root of the search tree
        self.search_stats = []
        # List of *estimated* total future reward from the node, as measured by the average value at the root of the search tree
        self.values = []

        self.priorities = []

    def add_step(self, obs: np.ndarray, action: int, reward: int, root):
        # Root is a TreeNode object at the root of the search tree for the given state

        # Note that when taking a step you get the action, reward and new observation
        # but for training purposes we want to connect the reward with the action and *old* observation.
        # We therefore add the first frame when we initialize the class, so connected frame-action-reward
        # tuples have the same index

        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

        self.search_stats.append([c.num_visits if c else 0 for c in root.children])
        self.values.append(root.average_val)

    def add_priorities(self, n_steps=5):
        assert len(self.priorities) == 0

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

            priority = (r - value_target) ** 2
            self.priorities.append(priority)

    def make_target(self, ndx: int, reward_depth: int = 5, rollout_depth: int = 3):
        # ndx is where in the record of the game we start
        # reward_depth is how far into the future we use the actual reward - beyond this we use predicted value

        # rollout_depth is how many iterations of the dynamics function we take, from where we apply the represent function
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
            rollout_depth = min(rollout_depth, game_len - ndx)

            for i in range(rollout_depth):
                target_rewards.append(self.rewards[ndx + i])

                # If we have an estimated value at the current index + reward_depth
                # then this is our base value (after discounting)
                # else we start at 0
                bootstrap_index = ndx + reward_depth + i
                # #print(
                #     f"bootstrap {bootstrap_index} ndx {ndx} reward depth {reward_depth} i {i} game_len {game_len}"
                # )
                if bootstrap_index < len(self.values):
                    target_value = self.values[bootstrap_index] * (
                        self.discount**reward_depth
                    )
                    # print(target_value)
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
            images = self.observations[ndx : ndx + rollout_depth]
            actions = self.actions[ndx : ndx + rollout_depth]

            return images, actions, target_values, target_rewards, target_policies

    def reanalyse(self, mcts):
        for i, obs in enumerate(self.observations[:-1]):
            new_root = mcts.search(self.config["n_simulations"], obs)
            self.values[i] = new_root.average_val

        return self


class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.size = config["buffer_size"]  # How many game records to store
        self.buffer = []  # List of stored game records
        self.total_vals = 0  # How many total steps are stored

        self.prioritized_replay = config["priority_replay"]
        self.priority_alpha = config["priority_alpha"]

        # List of start points of each game if the whole buffer were concatenated
        self.game_starts_list = []

        self.reward_depth = config["reward_depth"]
        self.rollout_depth = config["rollout_depth"]
        self.priorities = []

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

    def save_game(self, game):
        # If reached the max size, remove the oldest GameRecord, and update stats accordingly
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)

        self.buffer.append(game)
        self.update_stats()

    def get_batch(self, batch_size=40):
        batch = []

        # Get a random list of points across the length of the buffer to take training examples

        if self.prioritized_replay:
            probabilities = self.priorities
        else:
            probabilities = None

        start_vals = np.random.choice(
            list(range(self.total_vals)), size=batch_size, p=probabilities
        )

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
            ) = game.make_target(
                game_ndx,
                reward_depth=self.reward_depth,
                rollout_depth=self.rollout_depth,
            )

            # Add tuple to batch
            if self.priorities:
                weight = 1 / (self.priorities[val] * self.total_vals)
            else:
                weight = 1

            batch.append(
                (
                    images,
                    actions,
                    target_values,
                    target_rewards,
                    target_policies,
                    weight,
                )
            )

        return batch

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

    def reanalyse(self, mcts):
        # print("go")
        for i, game in enumerate(self.buffer):
            # Reanalyse on average every 50 games at max size
            if random() < 2 / len(self.buffer):
                new_game = game.reanalyse(mcts)
                self.buffer[i] = new_game


@ray.remote
class Reanalyser:
    def __init__(self):
        pass
