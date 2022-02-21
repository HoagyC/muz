# Replicating MuZero
# In very basic form

import torch
import typing

from enum import Enum
from torch import nn as nn
from torch.nn import functional as F


class Game:
    def __init__(
        self,
    ):
        pass


class Move:
    def __init__(self, row, col):
        assert row in [1, 2, 3] and col in [1, 2, 3], "Incorrect move"
        self.row = row
        self.col = col


class GameState:
    def __init__(self):
        self.state = [0] * 9
        self.winning_triplets = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]

    def get_pos(self, row, col):
        return (col * 3) + row

    def get_state(self, row, col):
        return self.state[self.get_pos(row, col)]

    def move(self, row, col, i):
        assert all([x in [0, 1, 2] for x in [row, col, i]]), "Incorrect game state"
        if self.state[self.get_pos(row, col)] != 0:
            return -1
        self.state[self.get_pos(row, col)] = i
        return 0

    def check_winner(self):
        for wt in self.winning_triplets:
            p1win, p2win = False, False
            squares = [self.state[i] for i in wt]
            if all([x == 1 for x in squares]):
                p1win = True

            if all([x == 2 for x in squares]):
                p2win = True

            assert not (p1win and p2win), "Both players have won"

            if p1win:
                return 1
            elif p2win:
                return 2

        return 0


def noughts_dynamics(game_state: GameState, move):
    pass


noughts = GameState()

noughts.move(1, 2, 1)
noughts.move(2, 0, 2)
noughts.move(2, 1, 2)
noughts.move(2, 2, 2)

print(noughts.state, noughts.check_winner())


m = Move(1, 2)
print(m.col, m.row)
n = Move(4, 2)


# The muzero algorithm has a represention function, a dynamics function, and a prediction function
# dynamics takes in the current state, and an action, and learns the next state BUT not directly - the state are just internal states, no enforced semantic content
# so can we just

ENV_SIZE = 9
N_PLAYER = 2

hidden_w = 20
input_w = ENV_SIZE + 1


class DynamicNet(nn.Module):
    def __init__(self):
        super(DynamicNet, self).__init__()

        self.fc1 = nn.Linear(input_w + hidden_w, hidden_w)
        self.fc2 = nn.Linear(hidden_w, hidden_w)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)


class RepresentationNet(nn.module):
    def __init__(self):
        super(RepresentationNet, self).__init__()

        self.fc1 = nn.Linear(input_w, hidden_w)
        self.fc2 = nn.Linear(hidden_w, hidden_w)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)


class PredictionNet(nn.Module):
    def __init__(self):
        super(PredictionNet, self).__init__()

        self.fc1 = nn.Linear(hidden_w, hidden_w)
        self.fc2 = nn.Linear(hidden_w, input_w + 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        p = F.softmax(x[:-1])
        v = x[-1]
        return p, v


dynamic_net = DynamicNet()
representation_net = RepresentationNet()
prediction_net = PredictionNet()
