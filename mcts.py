import math
import random

import numpy as np

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from training import ReplayBuffer
from models import scalar_to_support, support_to_scalar


class MinMax:
    def __init__(self):
        self.max_reward = -float("inf")
        self.min_reward = float("inf")

    def update(self, val):
        self.max_reward = max(val, self.max_reward)
        self.min_reward = min(val, self.min_reward)

    def normalize(self, val):
        if self.max_reward > self.min_reward:
            return (val - self.min_reward) / (self.max_reward - self.min_reward)
        else:
            return val


class MCTS:
    def __init__(self, action_size, obs_size, mu_net, config):
        self.action_size = action_size
        self.obs_size = obs_size
        self.mu_net = mu_net

        self.val_weight = config["val_weight"]
        self.discount = config["discount"]
        self.batch_size = config["batch_size"]
        self.grad_clip = config["grad_clip"]

        self.minmax = MinMax()

    def search(self, n_simulations, current_frame):
        # with torch.no_grad():
        #     model.eval()

        frame_t = torch.tensor(current_frame).unsqueeze(0)
        init_latent = self.mu_net.represent(frame_t)[0]
        init_policy, init_val = [
            x[0] for x in self.mu_net.predict(init_latent.unsqueeze(0))
        ]

        init_val = support_to_scalar(torch.softmax(init_val, 0))
        root_node = TreeNode(
            latent=init_latent,
            action_size=self.action_size,
            val_pred=init_val,
            pol_pred=init_policy,
            discount=self.discount,
            minmax=self.minmax,
        )

        for i in range(n_simulations):
            with torch.no_grad():
                current_node = root_node
                new_node = False
                search_list = [current_node]
                while not new_node:
                    value_pred, policy_pred, latent = (
                        current_node.val_pred,
                        current_node.pol_pred,
                        current_node.latent,
                    )
                    action = current_node.pick_action()
                    if current_node.children[action] is None:
                        action_t = nn.functional.one_hot(
                            torch.tensor([action]), num_classes=self.action_size
                        )

                        new_latent, reward = [
                            x[0]
                            for x in self.mu_net.dynamics(latent.unsqueeze(0), action_t)
                        ]
                        reward = support_to_scalar(torch.softmax(reward, 0))

                        new_policy, new_val = [
                            x[0] for x in self.mu_net.predict(new_latent.unsqueeze(0))
                        ]
                        new_val = support_to_scalar(torch.softmax(new_val, 0))

                        current_node.insert(
                            action_n=action,
                            latent=new_latent,
                            val_pred=new_val,
                            pol_pred=new_policy,
                            reward=reward,
                            minmax=self.minmax,
                        )
                        # print(action, reward)
                        new_node = True
                    else:
                        current_node = current_node.children[action]

                    search_list.append(current_node)

                self.backpropagate(search_list, new_val)
        return root_node

    def train(self, buffer: ReplayBuffer, n_batches: int):
        total_loss, total_policy_loss, total_reward_loss, total_value_loss = 0, 0, 0, 0

        for _ in range(n_batches):
            batch_policy_loss, batch_reward_loss, batch_value_loss = 0, 0, 0

            batch = buffer.get_batch(batch_size=self.batch_size)
            for (
                init_image,
                actions,
                target_values,
                target_rewards,
                target_policies,
            ) in batch:
                assert (
                    len(actions)
                    == len(target_policies)
                    == len(target_rewards)
                    == len(target_values)
                )
                rollout_depth = len(actions)
                hidden_state = self.mu_net.represent(
                    torch.tensor(init_image).unsqueeze(0)
                )[0]

                for i in range(rollout_depth):
                    target_value = torch.tensor(target_values[i], dtype=torch.float32)
                    target_reward = torch.tensor(target_rewards[i], dtype=torch.float32)
                    target_policy = torch.tensor(
                        target_policies[i], dtype=torch.float32
                    )

                    one_hot_action = nn.functional.one_hot(
                        torch.tensor([actions[i]]).to(dtype=torch.int64),
                        num_classes=self.action_size,
                    )

                    pred_policy, pred_value = [
                        x[0] for x in self.mu_net.predict(hidden_state.unsqueeze(0))
                    ]

                    hidden_state, pred_reward = [
                        x[0]
                        for x in self.mu_net.dynamics(
                            hidden_state.unsqueeze(0), one_hot_action
                        )
                    ]

                    policy_loss = self.mu_net.policy_loss(
                        pred_policy.unsqueeze(0), target_policy.unsqueeze(0)
                    )

                    hidden_state.register_hook(lambda grad: grad * 0.5)

                    target_reward_s = scalar_to_support(target_reward)
                    target_value_s = scalar_to_support(target_value)
                    # print(
                    #     "reward",
                    #     target_reward,
                    #     pred_reward,
                    #     "value",
                    #     target_value,
                    #     pred_value,
                    # )
                    value_loss = self.mu_net.value_loss(
                        pred_value.unsqueeze(0), target_value_s.unsqueeze(0)
                    )
                    reward_loss = self.mu_net.reward_loss(
                        pred_reward.unsqueeze(0), target_reward_s.unsqueeze(0)
                    )
                    # if i == 0:
                    #     print(
                    #         support_to_scalar(torch.softmax(pred_value, 0)),
                    #         target_value,
                    #         hidden_state,
                    #     )

                    # print(f'r {pred_reward:3.3}, {target_reward:3.3}, v {pred_value:5.3}, {target_value:5.3}')

                    batch_policy_loss += policy_loss
                    batch_value_loss += value_loss
                    batch_reward_loss += reward_loss

                    # print(f'p {policy_loss}, v {value_loss}, r {reward_loss}')

            batch_loss = (
                batch_policy_loss
                + batch_reward_loss
                + (batch_value_loss * self.val_weight)
            )
            # total_loss = total_reward_loss
            self.mu_net.optimizer.zero_grad()
            batch_loss.backward()

            # nn.utils.clip_grad_norm_(self.mu_net.pred_net.parameters(), self.grad_clip)
            # nn.utils.clip_grad_norm_(self.mu_net.dyna_net.parameters(), self.grad_clip)
            # nn.utils.clip_grad_norm_(self.mu_net.repr_net.parameters(), self.grad_clip)

            self.mu_net.optimizer.step()
            # score = self.minmax.normalize(q) + (p * vis_frac * balance_term)

            total_loss += batch_loss
            total_value_loss += batch_value_loss
            total_policy_loss += batch_policy_loss
            total_reward_loss += batch_reward_loss

        metrics_dict = {
            "Loss/total": total_loss,
            "Loss/policy": total_policy_loss,
            "Loss/reward": total_reward_loss,
            "Loss/value": (total_value_loss * self.val_weight),
        }

        return metrics_dict

    def backpropagate(
        self,
        search_list,
        value,
    ):
        for node in search_list[::-1]:
            node.num_visits += 1
            node.update_val(value)
            value = node.reward + (value * self.discount)
            self.minmax.update(value)


class TreeNode:
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
    ):
        self.action_size = action_size
        self.children = [None] * action_size
        self.latent = latent
        self.val_pred = val_pred
        self.pol_pred = pol_pred
        self.parent = parent
        self.average_val = 0
        self.num_visits = 0
        self.reward = reward

        self.discount = discount
        self.minmax = minmax

    def insert(self, action_n, latent, val_pred, pol_pred, reward, minmax):
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
            )

            self.children[action_n] = new_child

        else:
            raise ValueError("This node has already been traversed")

    def increment(self):
        self.num_visits += 1
        if self.parent:
            self.parent.increment()

    def update_val(self, curr_val):
        nmtr = self.average_val * self.num_visits + (curr_val + self.val_pred)
        dnmtr = self.num_visits + 1
        self.average_val = nmtr / dnmtr

        if self.parent is not None:
            self.parent.update_val(
                curr_val * self.discount
            )  # send this down to the parent so that it also updates

    def action_score(self, action_n, total_visit_count):
        c1 = 1.25
        c2 = 19652

        child = self.children[action_n]

        n = child.num_visits if child else 0
        q = child.average_val if child else 0

        p = self.pol_pred[
            action_n
        ]  # p here is the prior - the expectation of what the the policy will look like
        # p = 1 / self.action_size

        vis_frac = math.sqrt(total_visit_count) / (1 + n)
        balance_term = c1 + math.log((total_visit_count + c2 + 1) / c2)

        # score = self.minmax.normalize(q) + (p * vis_frac * balance_term)

        score = self.minmax.normalize(q) + (vis_frac * balance_term)

        return score

    def pick_action(self):
        total_visit_count = sum([a.num_visits if a else 0 for a in self.children])

        scores = [
            self.action_score(a, total_visit_count) for a in range(self.action_size)
        ]
        maxscore = max(scores)

        action = np.random.choice(
            [a for a in range(self.action_size) if scores[a] == maxscore]
        )
        return action

    def pick_game_action(self, temperature):
        visit_counts = [a.num_visits if a else 0 for a in self.children]
        val_preds = [c.val_pred if c else 0 for c in self.children]
        # print(
        #     visit_counts,
        #     self.val_pred,
        #     val_preds,
        #     "L" if val_preds[0] > val_preds[1] else "R",
        # )

        if temperature == 0:
            max_vis = max(visit_counts)
            action = np.random.choice(
                [a for a in range(self.action_size) if visit_counts[a] == max_vis]
            )

        else:
            scores = [(vc + 1) ** (1 / temperature) for vc in visit_counts]
            total_score = sum(scores)
            adjusted_scores = [score / total_score for score in scores]

            action = np.random.choice(self.action_size, p=adjusted_scores)

        return action


if __name__ == "__main__":
    pass
