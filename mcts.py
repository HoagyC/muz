import math
import random

import torch
from torch import nn


class CartRepr(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, state):
        out = self.fc1(state)
        out = torch.relu(out)
        out = self.fc2(out)
        return out


class CartDyna(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, action, latent):
        out = torch.cat(action, latent)
        out = torch.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out


class CartPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, latent):
        out = self.fc1(latent)
        out = torch.relu(out)
        out = self.fc2(out)
        action = torch.softmax(out[:2], 0)
        value = out[2]
        return action, value


class MCTS:
    def __init__(self, action_size, repr_net, dyna_net, pred_net):
        self.action_size = action_size
        self.max_reward = 1
        self.min_reward = 0
        self.prediction_net = pred_net
        self.dynamics_net = dyna_net
        self.representation_net = repr_net

    def choose_action(self, policy):
        return random.randrange(self.action_size)

    def representation_net(self, frame):
        return [0] * 5

    def dynamics_net(self, latent, action):
        return [random.random() for _ in range(5)]

    def prediction_net(self, latent):
        value = random.random() * 10
        policy = [random.random() for _ in range(self.action_size)]
        return value, policy

    def search(self, n_simulations, current_frame):
        # with torch.no_grad():
        #     model.eval()

        init_latent = self.representation_net(current_frame)
        init_val, init_policy = self.prediction_net(init_latent)
        root_node = TreeNode(
            latent=init_latent,
            action_size=self.action_size,
            val_pred=init_val,
            pol_pred=init_policy,
        )

        for i in range(n_simulations):
            current_node = root_node
            new_node = False

            while not new_node:
                value_pred, policy_pred, latent = (
                    current_node.val_pred,
                    current_node.pol_pred,
                    current_node.latent,
                )
                action = self.choose_action(policy_pred)
                if current_node.children[action] is None:
                    new_latent = self.dynamics_net(latent, action)
                    new_val, new_policy = self.prediction_net(new_latent)
                    current_node.insert(
                        action_n=action,
                        latent=new_latent,
                        val_pred=new_val,
                        pol_pred=new_policy,
                    )
                    new_node = True
                else:
                    current_node = current_node.children[action]

        return root_node


class TreeNode:
    def __init__(self, latent, action_size, val_pred=None, pol_pred=None, parent=None):
        self.action_size = action_size
        self.children = [None] * action_size
        self.latent = latent
        self.val_pred = val_pred
        self.pol_pred = pol_pred
        self.parent = parent
        self.average_val = 0
        self.num_visits = 0

    def insert(self, action_n, latent, val_pred, pol_pred):
        if self.children[action_n] is None:
            new_child = TreeNode(
                latent=latent,
                val_pred=val_pred,
                pol_pred=pol_pred,
                action_size=self.action_size,
                parent=self,
            )

            self.children[action_n] = new_child
            self.increment()
            self.update_val(val_pred)

        else:
            raise ValueError("This node has already been traversed")

    def increment(self):
        self.num_visits += 1
        if self.parent:
            self.parent.increment()

    def update_val(self, curr_val, discount=1):
        print(self.num_visits, curr_val, self.val_pred)
        nmtr = self.average_val * self.num_visits + (curr_val + self.val_pred)
        dnmtr = self.num_visits + 1
        self.average_val = nmtr / dnmtr

        if self.parent is not None:
            self.parent.update_val(
                curr_val * discount
            )  # send this down to the parent so that it also updates

    def action_score(self, action_n, total_visit_count):
        c1 = 1.25
        c2 = 19652

        child = self.children[action_n]

        n = child.num_visits if child else 0
        q = child.average_val if child else 0
        p = self.policy[action_n]

        vis_frac = math.sqrt(total_visit_count) / (1 + n)
        balance_term = c1 + math.log((total_visit_count + c2 + 1) / c2)

        score = q + (p * vis_frac * balance_term)
        return score

    def pick_action(self):
        total_visit_count = sum([a.num_visits for a in self.children])

        scores = [self.action_score(c, total_visit_count) for c in self.children]

        return scores.index(max(scores))


if __name__ == "__main__":
    mcts = MCTS(
        action_size=4,
        repr_net=CartRepr(),
        dyna_net=CartDyna(),
        pred_net=CartPred(),
    )
