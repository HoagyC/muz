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
<<<<<<< HEAD

=======
        if random.random() < 0.01:
            print(
                f"fr mean: {torch.mean(frame_t)}, fr var{torch.var(frame_t)}"
                + f"la mean: {torch.mean(init_latent)}, la var {torch.var(init_latent)}"
            )
>>>>>>> origin/testgame
        init_policy, init_val = [x[0] for x in mu_net.predict(init_latent.unsqueeze(0))]

        # Getting probabilities from logits and a scalar value from the categorical support
        init_policy_probs = torch.softmax(init_policy, 0)
        init_val = support_to_scalar(torch.softmax(init_val, 0))

        init_policy_probs = add_dirichlet(
            init_policy_probs,
            config["root_dirichlet_alpha"],
            config["explore_frac"],
        )

        # initialize the search tree with a root node
        root_node = TreeNode(
            latent=init_latent,
            action_size=mu_net.action_size,
            val_pred=init_val,
            pol_pred=init_policy_probs,
            minmax=minmax,
            config=config,
            num_visits=0,
        )

        for i in range(config["n_simulations"]):
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
                        torch.tensor([action], device=device),
                        num_classes=mu_net.action_size,
                    )

                    # apply the dynamics function to get a representation of the state after the action,
                    # and the reward gained
                    # then estimate the policy and value at this new state

                    latent, reward = [
                        x[0] for x in mu_net.dynamics(latent.unsqueeze(0), action_t)
                    ]
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
                    )

                    # We have reached a new node and therefore this is the end of the simulation
                    new_node = True
                else:
                    # If we have already explored this node then we take the child as our new current node
                    current_node = current_node.children[action]

            # Updates the visit counts and average values of the nodes that have been traversed
            backpropagate(search_list, new_val, minmax, config["discount"])
    return root_node


<<<<<<< HEAD
@ray.remote
class Trainer:
    def __init__(self):
        pass

    def train(
        self,
        mu_net,
        memory,
        config,
        log_dir,
        device=torch.device("cpu"),
        writer=None,
    ):
        """
        The train function simultaneously trains the prediction, dynamics and representation functions
        each batch has a series of values, rewards and policies, that must be predicted only
        from the initial_image, and the actions.

        This unrolled training is how the dynamics function
        is trained - is it akin to training through a recurrent neural network with the prediction function
        as a head
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        next_batch = None
        total_batches = ray.get(memory.get_data.remote())["batches"]
        if "latest_model_dict.pt" in os.listdir(log_dir):
            mu_net = ray.get(memory.load_model.remote(log_dir, mu_net))

        while ray.get(memory.get_buffer_len.remote()) == 0:
            time.sleep(1)
        ms = time.time()

        while True:
            print_timing("start")
            st = time.time()

            (
                total_loss,
                total_policy_loss,
                total_reward_loss,
                total_value_loss,
                total_consistency_loss,
            ) = (0, 0, 0, 0, 0)
            if not next_batch:
                next_batch = memory.get_batch.remote(batch_size=config["batch_size"])
            print_timing("next batch command")

            val_diff = 0

            print_timing("load model")
            mu_net.train()
            print_timing("to train")
            mu_net = mu_net.to(device)
            print_timing("to device")
            (
                batch_policy_loss,
                batch_reward_loss,
                batch_value_loss,
                batch_consistency_loss,
            ) = (0, 0, 0, 0)
            print_timing("init")
            (
                images,
                actions,
                target_values,
                target_rewards,
                target_policies,
                weights,
                depths,
            ) = ray.get(next_batch)
            next_batch = memory.get_batch.remote(batch_size=config["batch_size"])
            print_timing("get batch")

            images = images.to(device=device)
            actions = actions.to(device=device)
            target_rewards = target_rewards.to(device=device)
            target_values = target_values.to(device=device)
            target_policies = target_policies.to(device=device)
            weights = weights.to(device=device)
            print_timing("uploading weights")

            assert (
                len(actions)
                == len(target_policies)
                == len(target_rewards)
                == len(target_values)
                == len(images)
            )
            assert config["rollout_depth"] == actions.shape[1]
            print_timing("asserting")

            # This is how far we will deny the use of the representation function,
            # requiring the dynamics function to learn to represent the s, a -> s function
            # All batch tensors are index first by batch x rollout
            init_images = images[:, 0]
            print_timing("images0")

            latents = mu_net.represent(init_images)
            print_timing("represent")
            for i in range(config["rollout_depth"]):
                print_timing("rollout start")
                screen_t = torch.tensor(depths) > i
                if torch.sum(screen_t) < 1:
                    continue
                print_timing("for init")

                # We must do tthis sequentially, as the input to the dynamics function requires the output
                # from the previous dynamics function

                target_value_step_i = target_values[:, i]
                target_reward_step_i = target_rewards[:, i]
                target_policy_step_i = target_policies[:, i]
                print_timing("make target")

                if config["consistency_loss"]:
                    target_latents = mu_net.represent(images[:, i]).detach()
                print_timing("repreSENT")
                one_hot_actions = nn.functional.one_hot(
                    actions[:, i],
                    num_classes=mu_net.action_size,
                ).to(device=device)

                pred_policy_logits, pred_value_logits = mu_net.predict(latents)

                new_latents, pred_reward_logits = mu_net.dynamics(
                    latents, one_hot_actions
                )
                print_timing("forward pass")

                # We scale down the gradient, I believe so that the gradient at the base of the unrolled
                # network converges to a maximum rather than increasing linearly with depth
                new_latents.register_hook(lambda grad: grad * 0.5)

                # target_reward_sup_i = scalar_to_support(
                #     target_reward_stepi, half_width=config["support_width"]
                # )

                # target_value_sup_i = scalar_to_support(
                #     target_value_stepi, half_width=config["support_width"]
                # )

                print_timing("discrete to scalar")

                # Cutting off cases where there's not enough data for a full rollout

                # The muzero paper calculates the loss as the squared difference between scalars
                # but CrossEntropyLoss is used here for a more stable value loss when large values are encountered

                pred_values = support_to_scalar(
                    torch.softmax(pred_value_logits[screen_t], dim=1)
                )
                pred_rewards = support_to_scalar(
                    torch.softmax(pred_reward_logits[screen_t], dim=1)
                )
                vvar = torch.var(pred_rewards)
                # if vvar > 0.01:
                #     print(vvar, torch.var(target_reward_step_i))
                # if vvar > 10:
                #     print(pred_rewards, target_reward_step_i)
                #     print(i)
                val_diff += sum(
                    target_value_step_i[screen_t]
                    - support_to_scalar(
                        torch.softmax(pred_value_logits[screen_t], dim=1)
                    )
                )
                val_loss = torch.nn.MSELoss()
                reward_loss = torch.nn.MSELoss()
                value_loss = val_loss(pred_values, target_value_step_i[screen_t])
                reward_loss = reward_loss(pred_rewards, target_reward_step_i[screen_t])
                policy_loss = mu_net.policy_loss(
                    pred_policy_logits[screen_t], target_policy_step_i[screen_t]
                )

                if config["consistency_loss"]:
                    consistency_loss = mu_net.consistency_loss(
                        latents[screen_t], target_latents[screen_t]
                    )
                else:
                    consistency_loss = 0

                batch_policy_loss += (policy_loss * weights[screen_t]).mean()
                batch_value_loss += (value_loss * weights[screen_t]).mean()
                batch_reward_loss += (reward_loss * weights[screen_t]).mean()
                batch_consistency_loss += (consistency_loss * weights[screen_t]).mean()
                latents = new_latents
                # print(batch_value_loss, batch_consistency_loss)

                # print(
                #     target_value_stepi,
                #     pred_value_logits[screen_t],
                #     torch.softmax(pred_value_logits[screen_t], dim=1),
                #     support_to_scalar(
                #         torch.softmax(pred_value_logits[screen_t], dim=1)
                #     ),
                #     target_value_sup_i,
                # )
                print_timing("done losses")
            # Aggregate the losses to a single measure
            batch_loss = (
                batch_policy_loss
                + batch_reward_loss
                + (batch_value_loss * config["val_weight"])
                + (batch_consistency_loss * config["consistency_weight"])
            )
            batch_loss = batch_loss.mean()
            print_timing("batch loss")

            if config["debug"]:
                print(
                    f"v {batch_value_loss}, r {batch_reward_loss}, p {batch_policy_loss}, c {batch_consistency_loss}"
                )

            # Zero the gradients in the computation graph and then propagate the loss back through it
            mu_net.optimizer.zero_grad()
            batch_loss.backward()
            if config["grad_clip"] != 0:
                torch.nn.utils.clip_grad_norm_(mu_net.parameters(), config["grad_clip"])
            mu_net.optimizer.step()
            print_timing("optimizer")

            total_loss += batch_loss
            total_value_loss += batch_value_loss
            total_policy_loss += batch_policy_loss
            total_reward_loss += batch_reward_loss
            total_consistency_loss += batch_consistency_loss
            print_timing("loss")

            metrics_dict = {
                "Loss/total": total_loss,
                "Loss/policy": total_policy_loss,
                "Loss/reward": total_reward_loss,
                "Loss/value": (total_value_loss * config["val_weight"]),
                "Loss/consistency": (
                    total_consistency_loss * config["consistency_weight"]
                ),
            }
            mu_net = mu_net.to(device=torch.device("cpu"))
            print_timing("to_cpu")

            memory.done_batch.remote()
            if total_batches % 50 == 0:
                memory.save_model.remote(mu_net, log_dir)
            total_batches += 1

            if self.writer:
                for key, val in metrics_dict.items():
                    self.writer.add_scalar(key, val, total_batches)

            if total_batches % 100 == 0:
                print(
                    f"Completed {total_batches} total batches of size {config['batch_size']}, took {(time.time() - st)}"
                )
            print_timing("saving/end")

        return metrics_dict


=======
>>>>>>> origin/testgame
def backpropagate(search_list, value, minmax, discount):
    """Going backward through the visited nodes, we increase the visit count of each by one
    and set the value, discounting the value at the node ahead, but then adding the reward"""
    for node in search_list[::-1]:
        node.num_visits += 1
        node.update_val(value)
        value = node.reward + (value * discount)
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

    def insert(self, action_n, latent, val_pred, pol_pred, reward, minmax, config):
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
