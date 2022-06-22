# import datetime
import math
import os
import random
import time
import datetime
import yaml
import pickle
import numpy as np
import ray
from matplotlib import pyplot as plt

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from models import scalar_to_support, support_to_scalar


def search(
    config,
    mu_net,
    current_frame,
    minmax,
    log_dir,
    device=torch.device("cpu"),
):
    """
    This function takes a frame and creates a tree of possible actions that could
    be taken from the frame, assessing the expected value at each location
    and returning this tree which contains the data needed to choose an action

    The models expect inputs where the first dimension is a batch dimension,
    but the way in which we traverse the tree means we only pass a single
    input at a time. There is therefore the need to consistently squeeze and unsqueeze
    (ie add and remove the first dimension) so as not to confuse things by carrying around
    extraneous dimensions

    Note that mu_net returns logits for the policy, value and reward
    and that the value and reward are represented categorically rather than
    as a scalar
    """
    mu_net.eval()
    mu_net = mu_net.to(device)

    with torch.no_grad():

        frame_t = torch.tensor(current_frame, device=device)
        init_latent = mu_net.represent(frame_t.unsqueeze(0))[0]
        init_policy, init_val = [x[0] for x in mu_net.predict(init_latent.unsqueeze(0))]

        # Getting probabilities from logits and a scalar value from the categorical support
        init_policy_probs = torch.softmax(init_policy, 0)
        init_val = support_to_scalar(torch.softmax(init_val, 0))

        init_policy_probs = add_dirichlet(
            init_policy_probs,
            config["root_dirichlet_alpha"],
            config["explore_frac"],
        )

        if config["value_prefix"]:
            # Hidden size must be (num_layers, batch_size, hidden_size)
            init_lstm_hiddens = (
                torch.zeros(1, 1, config["lstm_hidden_size"]).detach(),
                torch.zeros(1, 1, config["lstm_hidden_size"]).detach(),
            )
        else:
            init_lstm_hiddens = None

        # initialize the search tree with a root node
        root_node = TreeNode(
            latent=init_latent,
            action_size=mu_net.action_size,
            val_pred=init_val,
            pol_pred=init_policy_probs,
            minmax=minmax,
            config=config,
            num_visits=0,
            lstm_hiddens=init_lstm_hiddens,
        )

        for i in range(config["n_simulations"]):
            # vital to have with(torch.no_grad() or the size of the computation graph quickly becomes gigantic
            current_node = root_node
            new_node = False

            # search list tracks the route of the simulation through the tree
            search_list = []
            while not new_node:
                search_list.append(current_node)
                value_pred = current_node.val_pred
                policy_pred = current_node.pol_pred
                latent = current_node.latent
                action = current_node.pick_action()
                lstm_hiddens = current_node.lstm_hiddens

                # if we pick an action that's been picked before we don't need to run the model to explore it
                if current_node.children[action] is None:
                    # Convert to a 2D tensor one-hot encoding the action
                    action_t = nn.functional.one_hot(
                        torch.tensor([action], device=device),
                        num_classes=mu_net.action_size,
                    )

                    # apply the dynamics function to get a representation of the state after the action,
                    # and the reward gained
                    # then estimate the policy and value at this new state

                    if config["value_prefix"]:
                        latent, reward, new_hiddens = mu_net.dynamics(
                            latent.unsqueeze(0), action_t, lstm_hiddens
                        )
                        latent = latent.squeeze_(0)
                        reward = reward.squeeze_(0)

                    else:
                        latent, reward = [
                            x[0] for x in mu_net.dynamics(latent.unsqueeze(0), action_t)
                        ]
                        new_hiddens = None
                    new_policy, new_val = [
                        x[0] for x in mu_net.predict(latent.unsqueeze(0))
                    ]

                    # convert logits to scalars and probaility distributions
                    reward = support_to_scalar(torch.softmax(reward, 0))
                    new_val = support_to_scalar(torch.softmax(new_val, 0))
                    policy_probs = torch.softmax(new_policy, 0)
                    current_node.insert(
                        action_n=action,
                        latent=latent,
                        val_pred=new_val,
                        pol_pred=policy_probs,
                        reward=reward,
                        minmax=minmax,
                        config=config,
                        lstm_hiddens=new_hiddens,
                    )

                    # We have reached a new node and therefore this is the end of the simulation
                    new_node = True
                else:
                    # If we have already explored this node then we take the child as our new current node
                    current_node = current_node.children[action]

            # Updates the visit counts and average values of the nodes that have been traversed
            backpropagate(search_list, new_val, minmax, config["discount"])
    return root_node


def backpropagate(search_list, value, minmax, discount):
    """Going backward through the visited nodes, we increase the visit count of each by one
    and set the value, discounting the value at the node ahead, but then adding the reward"""
    for node in search_list[::-1]:
        node.num_visits += 1
        value = node.reward + (value * discount)
        node.update_val(value)
        minmax.update(value)


class TreeNode:
    """
    TreeNode is an individual node of a search tree.
    It has one potential child for each potential action which, if it exists, is another TreeNode
    Its function is to hold the relevant statistics for deciding which action to take.
    """

    def __init__(
        self,
        latent,
        action_size,
        val_pred=None,
        pol_pred=None,
        parent=None,
        reward=0,
        minmax=None,
        config=None,
        num_visits=1,
        lstm_hiddens=None,
    ):

        self.action_size = action_size
        self.children = [None] * action_size
        self.latent = latent
        self.val_pred = val_pred
        self.pol_pred = pol_pred
        self.parent = parent
        self.average_val = val_pred
        self.num_visits = num_visits
        self.reward = reward

        self.minmax = minmax
        self.config = config
        self.lstm_hiddens = lstm_hiddens

    def insert(
        self,
        action_n,
        latent,
        val_pred,
        pol_pred,
        reward,
        minmax,
        config,
        lstm_hiddens=None,
    ):
        # The implementation here differs from the open MuZero (werner duvaud)
        # by only initializing tree nodes when they are chosen, rather than when their parent is chosen
        if self.children[action_n] is None:
            new_child = TreeNode(
                latent=latent,
                val_pred=val_pred,
                pol_pred=pol_pred,
                action_size=self.action_size,
                parent=self,
                reward=reward,
                minmax=minmax,
                config=self.config,
                lstm_hiddens=lstm_hiddens,
            )

            self.children[action_n] = new_child

        else:
            raise ValueError("This node has already been traversed")

    def update_val(self, curr_val):
        """Updates the average value of a node when a new value is receivied
        copies the formula of the muzero paper rather than the neater form of
        just tracking the sum and dividng as needed
        """
        nmtr = self.average_val * self.num_visits + curr_val
        dnmtr = self.num_visits + 1
        self.average_val = nmtr / dnmtr

    def action_score(self, action_n, total_visit_count):
        """
        Scoring function for the different potential actions, following the formula in Appendix B of muzero
        """
        c1 = 1.25
        c2 = 19652

        child = self.children[action_n]

        n = child.num_visits if child else 0

        q = self.minmax.normalize(child.average_val) if child else 0

        # p here is the prior - the expectation of what the the policy will look like
        prior = self.pol_pred[action_n]

        # This term increases the prior on those actions which have been taken only a small fraction
        # of the current number of visits to this node
        explore_term = math.sqrt(total_visit_count) / (1 + n)

        # This is intended to more heavily weight the prior as we take more and more actions.
        # Its utility is questionable, because with on the order of 100 simulations, this term will always be
        # close to 1.
        balance_term = c1 + math.log((total_visit_count + c2 + 1) / c2)
        score = q + (prior * explore_term * balance_term)

        return score

    def pick_action(self):
        """Gets the score each of the potential actions and picks the one with the highest"""
        total_visit_count = sum([a.num_visits if a else 0 for a in self.children])

        scores = [
            self.action_score(a, total_visit_count) for a in range(self.action_size)
        ]
        maxscore = max(scores)

        # Need to be careful not to always pick the first action as it common that two are scored identically
        action = np.random.choice(
            [a for a in range(self.action_size) if scores[a] == maxscore]
        )
        return action

    def pick_game_action(self, temperature):
        """
        Picks the action to actually be taken in game,
        taken by the root node after the full tree has been generated.
        Note that it only uses the visit counts, rather than the score or prior,
        these impact the decision only through their impact on where to visit
        """

        visit_counts = [a.num_visits if a else 0 for a in self.children]

        # zero temperature means always picking the highest visit count
        if temperature == 0:
            max_vis = max(visit_counts)
            action = np.random.choice(
                [a for a in range(self.action_size) if visit_counts[a] == max_vis]
            )

        # If temperature is non-zero, raise (visit_count + 1) to power (1 / T)
        # scale these to a probability distribution and use to select action
        else:
            scores = [(vc + 1) ** (1 / temperature) for vc in visit_counts]
            total_score = sum(scores)
            adjusted_scores = [score / total_score for score in scores]

            action = np.random.choice(self.action_size, p=adjusted_scores)

        # Prints a lot of useful information for how the algorithm is making decisions
        if self.config["debug"]:
            val_preds = [c.val_pred if c else 0 for c in self.children]
            print(visit_counts, self.val_pred, val_preds)

        return action


class MinMax:
    """
    This class tracks the smallest and largest values that have been seen
    so that it can normalize the values
    this is for when deciding which branch of the tree to explore
    by putting the values on a 0-1 scale, they become comparable with the probabilities
    given by the prior

    It comes pretty much straight from the MuZero pseudocode
    """

    def __init__(self):
        # initialize at +-inf so that any value will supercede the max/min
        self.max_value = -float("inf")
        self.min_value = float("inf")

    def update(self, val):
        self.max_value = max(float(val), self.max_value)
        self.min_value = min(float(val), self.min_value)

    def normalize(self, val):
        # places val between 0 - 1 linearly depending on where it sits between min_value and max_value
        if self.max_value > self.min_value:
            return (val - self.min_value) / (self.max_value - self.min_value)
        else:
            return val


def add_dirichlet(prior, dirichlet_alpha, explore_frac):
    noise = torch.tensor(
        np.random.dirichlet([dirichlet_alpha] * len(prior)), device=prior.device
    )
    new_prior = (1 - explore_frac) * prior + explore_frac * noise
    return new_prior


last_time = datetime.datetime.now()


if __name__ == "__main__":
    pass
