from random import randrange

import numpy as np
import torch 

from mcts import TreeNode


class GameRecord:
    def __init__(self, action_size: int, discount: float = 1):
        self.action_size = action_size
        self.discount = discount

        self.observations = []
        self.actions = []
        self.rewards = []
        self.search_stats = []
        self.values = []

    def add_step(self, obs: np.ndarray, action: int, reward: int, root: TreeNode):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

        self.search_stats.append([c.num_visits for c in root.children if c])
        self.values.append(root.average_val)

    def make_target(self, ndx: int, td_depth: int = 5):
        # When we predict the reward, we're predicting the reward from reaching the state
        # Not the reward from performing a subsequent action
        # We therefore need to get the previous reward
        with torch.no_grad():
            if ndx == 0:
                target_reward = 0
            else:
                target_reward = self.rewards[ndx - 1]

            bootstrap_index = ndx + td_depth
            if bootstrap_index < len(self.values):
                target_value = self.values[bootstrap_index] * self.discount ** td_depth
            else:
                target_value = 0

            for i in range(td_depth):
                if ndx + i < len(self.rewards):
                    target_value += self.rewards[ndx] * self.discount ** td_depth
                else:
                    break

            total_searches = sum(self.search_stats[ndx])
            target_policy = [x / total_searches for x in self.search_stats[ndx]]
            init_obs = self.observations[ndx]
            action = self.actions[ndx]

            return init_obs, action, (target_value, target_reward, target_policy)


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
            image, action, target = game.make_target(game_ndx)
            batch.append((image, action, target))

        return batch

    def get_ndxs(self, val):
        if val >= self.total_vals:
            raise ValueError('Trying to get a value beyond the length of the buffer')
        # Assumes the in_list is
        for i, l in enumerate(self.len_list):
            if l > val:
                return i - 1, val - self.len_list[i - 1]
            else: 
                pass
        return len(self.buffer) - 1, val - self.len_list[-1] 