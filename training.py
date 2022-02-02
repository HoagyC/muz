from random import randrange

import numpy as np
import torch


class GameRecord:
    def __init__(self, action_size: int, discount: float = 0.8):
        self.action_size = action_size
        self.discount = discount

        self.observations = []
        self.actions = []
        self.rewards = []
        self.search_stats = []
        self.values = []

    def add_step(self, obs: np.ndarray, action: int, reward: int, root):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

        self.search_stats.append([c.num_visits if c else 0 for c in root.children])
        self.values.append(root.average_val)

    def make_target(self, ndx: int, reward_depth: int = 5, rollout_depth: int = 5):
        # ndx is where in the record of the game we start
        # reward_depth is how far into the future we use the actual reward - beyond this we use predicted value

        # rollout_depth is how many
        # it acts like an additional dimension of batching when we make the target but the crucial difference is that
        # when we train, we will use a hidden state by repeated application of the dynamics function from the init_image
        # rather than creating a new hidden state from the game obs at the time
        # this is necessary to train the dynamics function to give useful information for predicting the value

        # When we predict the reward, we're predicting the reward from reaching the state
        # Not the reward from performing a subsequent action
        # We therefore need to get the previous reward

        with torch.no_grad():
            # We get a reward, value and policy for each step in the rollout of our hidden state
            target_rewards = []
            target_values = []
            target_policies = []

            game_len = len(self.search_stats)
            rollout_depth = min(
                rollout_depth, game_len - ndx
            )  # Make sure we don't try to roll out beyond end of game

            for i in range(rollout_depth):
                if ndx + i == 0:
                    target_rewards.append(torch.tensor(0, dtype=torch.float32))
                else:
                    target_rewards.append(self.rewards[ndx + i])

                bootstrap_index = ndx + reward_depth + i
                if bootstrap_index < len(self.values):
                    target_value = self.values[bootstrap_index] * (
                        self.discount**reward_depth
                    )
                else:
                    target_value = 0

                for j in range(reward_depth):
                    if ndx + i + j < len(self.rewards):
                        target_value += self.rewards[ndx + i + j] * (self.discount**j)
                    else:
                        break

                if target_value > 1 / (1 - self.discount):
                    print(target_value, self.values[bootstrap_index])
                    target_value = 1 / (1 - self.discount)

                target_values.append(target_value)

                total_searches = sum(self.search_stats[ndx + i])
                target_policies.append(
                    [x / total_searches for x in self.search_stats[ndx + i]]
                )

            init_obs = self.observations[ndx]
            actions = self.actions[ndx : ndx + rollout_depth]

            return init_obs, actions, target_values, target_rewards, target_policies


class ReplayBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buffer = []
        self.total_vals = 0
        self.len_list = []

    def save_game(self, game):
        if len(self.buffer) >= self.size:
            r = self.buffer.pop(0)
            remove_len = len(r.values)
            self.total_vals -= remove_len
            self.len_list.pop(0)
            self.len_list = [x - remove_len for x in self.len_list]

        game_len = len(game.values)
        self.len_list.append(self.total_vals)
        self.total_vals += game_len
        self.buffer.append(game)

    def get_batch(self, batch_size=40):
        # checking that the indexing is correct
        lens = [len(x.values) for x in self.buffer]
        assert [sum(lens[0:i]) for i in range(len(self.len_list))] == self.len_list

        batch = []

        start_vals = [randrange(self.total_vals) for _ in range(batch_size)]
        for val in start_vals:
            buf_ndx, game_ndx = self.get_ndxs(val)
            game = self.buffer[buf_ndx]
            (
                image,
                actions,
                target_values,
                target_rewards,
                target_policies,
            ) = game.make_target(game_ndx)

            batch.append(
                (image, actions, target_values, target_rewards, target_policies)
            )

        return batch

    def get_ndxs(self, val):
        if val >= self.total_vals:
            raise ValueError("Trying to get a value beyond the length of the buffer")
        # Assumes the in_list is
        for i, l in enumerate(self.len_list):
            if l > val:
                return i - 1, val - self.len_list[i - 1]
            else:
                pass
        return len(self.buffer) - 1, val - self.len_list[-1]
