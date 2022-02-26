import math
import os
import random

import numpy as np

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from training import ReplayBuffer
from models import scalar_to_support, support_to_scalar


class MCTS:
    """
    MCTS is the class where contains the main algorithm - it contains within it the
    models used to estimate the quantities of interest, as well as the functions to
    create the tree of potential actions, and to train the relevant models
    """

    def __init__(self, action_size, mu_net, config, log_dir):
        self.action_size = action_size
        self.config = config
        self.log_dir = log_dir

        # a class which holds the three functions as set out in MuZero paper: representation, prediction, dynamics
        # and has an optimizer which updates the parameters for all three simultaneously
        self.mu_net = mu_net

        # weighting of the value loss relative to policy and reward - paper recommends 0.25
        self.val_weight = config["val_weight"]
        self.consistency_weight = config["consistency_weight"]
        self.discount = config["discount"]
        self.batch_size = config["batch_size"]
        self.debug = config["debug"]

        self.root_dirichlet_alpha = config["root_dirichlet_alpha"]
        self.explore_frac = config["explore_frac"]

        # keeps track of the highest and lowest values found
        self.minmax = MinMax()

    def search(self, n_simulations, current_frame):
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
        self.load_model()

        with torch.no_grad():

            frame_t = torch.tensor(current_frame)
            if self.config["obs_type"] == "image":
                frame_t = torch.einsum("hwc->chw", [frame_t])
            init_latent = self.mu_net.represent(frame_t.unsqueeze(0))[0]
            init_policy, init_val = [
                x[0] for x in self.mu_net.predict(init_latent.unsqueeze(0))
            ]

            # Getting probabilities from logits and a scalar value from the categorical support
            init_policy_probs = torch.softmax(init_policy, 0)
            init_val = support_to_scalar(torch.softmax(init_val, 0))

            init_policy_probs = add_dirichlet(
                init_policy_probs, self.root_dirichlet_alpha, self.explore_frac
            )

            # initialize the search tree with a root node
            root_node = TreeNode(
                latent=init_latent,
                action_size=self.action_size,
                val_pred=init_val,
                pol_pred=init_policy_probs,
                discount=self.discount,
                minmax=self.minmax,
                debug=self.debug,
                num_visits=0,
            )

            for i in range(n_simulations):
                # vital to have to grad or the size of the computation graph quickly becomes gigantic
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

                    # if we pick an action that's been picked before we don't need to run the model to explore it
                    if current_node.children[action] is None:
                        # Convert to a 2D tensor one-hot encoding the action
                        action_t = nn.functional.one_hot(
                            torch.tensor([action]), num_classes=self.action_size
                        )

                        # apply the dynamics function to get a representation of the state after the action, and the reward gained
                        # then estimate the policy and value at this new state

                        latent, reward = [
                            x[0]
                            for x in self.mu_net.dynamics(latent.unsqueeze(0), action_t)
                        ]
                        new_policy, new_val = [
                            x[0] for x in self.mu_net.predict(latent.unsqueeze(0))
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
                            minmax=self.minmax,
                            debug=self.debug,
                        )

                        # We have reached a new node and therefore this is the end of the simulation
                        new_node = True
                    else:
                        # If we have already explored this node then we take the child as our new current node
                        current_node = current_node.children[action]

                # Updates the visit counts and average values of the nodes that have been traversed
                self.backpropagate(search_list, new_val)
        return root_node

    def train(self, buffer: ReplayBuffer, n_batches: int):
        """
        The train function simultaneously trains the prediction, dynamics and representation functions
        each batch has a series of values, rewards and policies, that must be predicted only
        from the initial_image, and the actions.

        This unrolled training is how the dynamics function
        is trained - is it akin to training through a recurrent neural network with the prediction function
        as a head
        """

        (
            total_loss,
            total_policy_loss,
            total_reward_loss,
            total_value_loss,
            total_consistency_loss,
        ) = (0, 0, 0, 0, 0)

        for _ in range(n_batches):
            (
                batch_policy_loss,
                batch_reward_loss,
                batch_value_loss,
                batch_consistency_loss,
            ) = (0, 0, 0, 0)

            (
                images,
                actions,
                target_values,
                target_rewards,
                target_policies,
                weights,
                depths,
            ) = buffer.get_batch(batch_size=self.batch_size)

            if self.config["priority_replay"]:
                batch_weight = 0

            assert (
                len(actions)
                == len(target_policies)
                == len(target_rewards)
                == len(target_values)
                == len(images)
            )
            assert self.config["rollout_depth"] == actions.shape[1]

            if self.config["priority_replay"]:
                batch_weight += weight

            # This is how far we will deny the use of the representation function,
            # requiring the dynamics function to learn to represent the s, a -> s function
            # All batch tensors are index first by batch x rollout
            init_images = images[:, 0]

            if self.config["obs_type"] == "image":
                init_images = torch.einsum("bhwc->bchw", init_images)

            latents = self.mu_net.represent(init_images)
            val_diff = 0
            for i in range(self.config["rollout_depth"]):
                # We must do tthis sequentially, as the input to the dynamics function requires the output
                # from the previous dynamics function

                target_value_stepi = target_values[:, i]
                target_reward_stepi = target_rewards[:, i]
                target_policy_stepi = target_policies[:, i]

                if self.config["consistency_loss"]:
                    if self.config["obs_type"] == "image":
                        images_chw = torch.einsum("bhwc->bchw", images[:, i])
                        target_latents = self.mu_net.represent(images_chw).detach()
                    else:
                        target_latents = self.mu_net.represent(images[:, i]).detach()

                one_hot_actions = nn.functional.one_hot(
                    actions[:, i],
                    num_classes=self.action_size,
                )

                pred_policy_logits, pred_value_logits = self.mu_net.predict(latents)

                new_latents, pred_reward_logits = self.mu_net.dynamics(
                    latents, one_hot_actions
                )

                # We scale down the gradient, I believe so that the gradient at the base of the unrolled
                # network converges to a maximum rather than increasing linearly with depth
                new_latents.register_hook(lambda grad: grad * 0.5)

                target_reward_sup_i = scalar_to_support(
                    target_reward_stepi, half_width=self.config["support_width"]
                )

                target_value_sup_i = scalar_to_support(
                    target_value_stepi, half_width=self.config["support_width"]
                )

                screen_t = torch.tensor(depths) > i

                # Cutting off cases where there's not enough data for a full rollout

                # The muzero paper calculates the loss as the squared difference between scalars
                # but CrossEntropyLoss is used here for a more stable value loss when large values are encountered
                value_loss = self.mu_net.value_loss(
                    pred_value_logits[screen_t], target_value_sup_i[screen_t]
                )
                reward_loss = self.mu_net.reward_loss(
                    pred_reward_logits[screen_t], target_reward_sup_i[screen_t]
                )
                breakpoint()
                # print(pred_policy_logits, target_policy_stepi)
                policy_loss = self.mu_net.policy_loss(
                    pred_policy_logits[screen_t], target_policy_stepi[screen_t]
                )
                # print(policy_loss)
                if self.config["consistency_loss"]:
                    consistency_loss = self.mu_net.consistency_loss(
                        latents[screen_t], target_latents[screen_t]
                    )
                else:
                    consistency_loss = 0

                batch_policy_loss += policy_loss
                batch_value_loss += value_loss
                batch_reward_loss += reward_loss
                if self.config["consistency_loss"]:
                    batch_consistency_loss += consistency_loss

                latents = new_latents

            # Aggregate the losses to a single measure
            batch_loss = (
                batch_policy_loss
                + batch_reward_loss
                + (batch_value_loss * self.val_weight)
                + (batch_consistency_loss * self.consistency_weight)
            ) / self.config["batch_size"]

            if self.config["priority_replay"]:
                average_weight = batch_weight / len(batch)
                batch_loss /= average_weight

            if self.config["debug"]:
                print(
                    f"v {batch_value_loss}, r {batch_reward_loss}, p {batch_policy_loss}, c {consistency_loss}"
                )
                # print(
                #     f"v {pred_value_logits}, r {pred_reward_logits}, p {pred_policy_logits}"
                # )

            # Zero the gradients in the computation graph and then propagate the loss back through it
            self.mu_net.optimizer.zero_grad()
            batch_loss.backward()
            if self.config["grad_clip"] != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.mu_net.parameters(), self.config["grad_clip"]
                )
            self.mu_net.optimizer.step()

            total_loss += batch_loss
            total_value_loss += batch_value_loss
            total_policy_loss += batch_policy_loss
            total_reward_loss += batch_reward_loss
            total_consistency_loss += batch_consistency_loss

        metrics_dict = {
            "Loss/total": total_loss,
            "Loss/policy": total_policy_loss,
            "Loss/reward": total_reward_loss,
            "Loss/value": (total_value_loss * self.val_weight),
            "Loss/consistency": (total_consistency_loss * self.consistency_weight),
        }

        self.save_model()

        return metrics_dict

    def backpropagate(
        self,
        search_list,
        value,
    ):
        """Going backward through the visited nodes, we increase the visit count of each by one
        and set the value, discounting the value at the node ahead, but then adding the reward"""
        for node in search_list[::-1]:
            node.num_visits += 1
            node.update_val(value)
            value = node.reward + (value * self.discount)
            self.minmax.update(value)

    def save_model(self):
        path = os.path.join(self.log_dir, "latest_model_dict.pt")
        torch.save(self.mu_net.state_dict(), path)

    def load_model(self):
        path = os.path.join(self.log_dir, "latest_model_dict.pt")
        if os.path.exists(path):
            self.mu_net.load_state_dict(torch.load(path))


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
        discount=1,
        minmax=None,
        debug=False,
        num_visits=1,
    ):

        self.action_size = action_size
        self.children = [None] * action_size
        self.latent = latent
        self.val_pred = val_pred
        self.pol_pred = pol_pred
        self.parent = parent
        self.average_val = 0
        self.num_visits = num_visits
        self.reward = reward

        self.discount = discount
        self.minmax = minmax

        self.debug = debug

    def insert(self, action_n, latent, val_pred, pol_pred, reward, minmax, debug):
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
                discount=self.discount,
                minmax=minmax,
                debug=debug,
            )

            self.children[action_n] = new_child

        else:
            raise ValueError("This node has already been traversed")

    def update_val(self, curr_val):
        """Updates the average value of a node when a new value is receivied
        copies the formula of the muzero paper rather than the neater form of just tracking the sum and dividng as needed
        """
        nmtr = self.average_val * self.num_visits + (curr_val + self.val_pred)
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
        q = child.average_val if child else 0

        # p here is the prior - the expectation of what the the policy will look like
        prior = self.pol_pred[action_n]

        # This term increases the prior on those actions which have been taken only a small fraction
        # of the current number of visits to this node
        explore_term = math.sqrt(total_visit_count) / (1 + n)

        # This is intended to more heavily weight the prior as we take more and more actions.
        # Its utility is questionable, because with on the order of 100 simulations, this term will always be
        # close to 1.
        balance_term = c1 + math.log((total_visit_count + c2 + 1) / c2)

        score = self.minmax.normalize(q) + (prior * explore_term * balance_term)

        return score

    def pick_action(self):
        """Gets the score each of the potential actions and picks the one with the highest"""
        total_visit_count = sum([a.num_visits if a else 0 for a in self.children])

        scores = [
            self.action_score(a, total_visit_count) for a in range(self.action_size)
        ]
        maxscore = max(scores)

        # Need to be careful not to always pick the first action is it common that two are scored identically
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
        if self.debug:
            val_preds = [c.val_pred if c else 0 for c in self.children]
            print(
                visit_counts,
                self.val_pred,
                val_preds,
                "L" if val_preds[0] > val_preds[1] else "R",
                "   ",
                "T"
                if action == 0
                and val_preds[0] > val_preds[1]
                or action == 1
                and val_preds[0] <= val_preds[1]
                else "F",
            )

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
        self.max_value = max(val, self.max_value)
        self.min_value = min(val, self.min_value)

    def normalize(self, val):
        # places val between 0 - 1 linearly depending on where it sits between min_value and max_value
        if self.max_value > self.min_value:
            return (val - self.min_value) / (self.max_value - self.min_value)
        else:
            return val


def add_dirichlet(prior, dirichlet_alpha, explore_frac):
    noise = np.random.dirichlet([dirichlet_alpha] * len(prior))
    new_prior = (1 - explore_frac) * prior + explore_frac * noise
    return new_prior


if __name__ == "__main__":
    pass
