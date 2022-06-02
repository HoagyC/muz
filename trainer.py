import datetime
import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
import ray
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from models import scalar_to_support, support_to_scalar


@ray.remote(max_restarts=-1)
class Trainer:
    def __init__(self):
        self.last_time = datetime.datetime.now()

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
        self.config = config
        torch.autograd.set_detect_anomaly(True)
        self.writer = SummaryWriter(log_dir=log_dir)
        next_batch = None
        total_batches = ray.get(memory.get_data.remote())["batches"]
        if "latest_model_dict.pt" in os.listdir(log_dir):
            mu_net = ray.get(memory.load_model.remote(log_dir, mu_net))
        mu_net.to(device)

        while ray.get(memory.get_buffer_len.remote()) == 0:
            time.sleep(1)
        ms = time.time()
        metrics_dict = {}

        while not ray.get(memory.is_finished.remote()):
            self.print_timing("start")
            st = time.time()

            (
                total_loss,
                total_policy_loss,
                total_reward_loss,
                total_value_loss,
                total_consistency_loss,
            ) = (0, 0, 0, 0, 0)
            if not next_batch:
                next_batch = memory.get_batch.remote(
                    batch_size=config["batch_size"], device=device
                )
            self.print_timing("next batch command")
            val_diff = 0
            mu_net.train()
            self.print_timing("to train")
            (
                batch_policy_loss,
                batch_reward_loss,
                batch_value_loss,
                batch_consistency_loss,
            ) = (0, 0, 0, 0)
            self.print_timing("init")
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
            self.print_timing("get batch")

            images = images.to(device=device)
            actions = actions.to(device=device)
            target_rewards = target_rewards.to(device=device)
            target_values = target_values.to(device=device)
            target_policies = target_policies.to(device=device)
            weights = weights.to(device=device)
            self.print_timing("changing to device")

            assert (
                len(actions)
                == len(target_policies)
                == len(target_rewards)
                == len(target_values)
                == len(images)
            )
            assert config["rollout_depth"] == actions.shape[1]
            self.print_timing("asserting")

            # This is how far we will deny the use of the representation function,
            # requiring the dynamics function to learn to represent the s, a -> s function
            # All batch tensors are index first by batch x rollout
            init_images = images[:, 0]
            self.print_timing("images0")

            latents = mu_net.represent(init_images)
            self.print_timing("represent")
            output_hiddens = None
            for i in range(config["rollout_depth"]):
                self.print_timing("rollout start")
                screen_t = torch.tensor(depths) > i
                if torch.sum(screen_t) < 1:
                    continue
                self.print_timing("for init")

                # We must do tthis sequentially, as the input to the dynamics function requires the output
                # from the previous dynamics function

                target_value_step_i = target_values[:, i]
                target_reward_step_i = target_rewards[:, i]
                target_policy_step_i = target_policies[:, i]
                self.print_timing("make target")

                if config["consistency_loss"]:
                    target_latents = mu_net.represent(images[:, i]).detach()
                self.print_timing("repreSENT")
                one_hot_actions = nn.functional.one_hot(
                    actions[:, i],
                    num_classes=mu_net.action_size,
                ).to(device=device)

                pred_policy_logits, pred_value_logits = mu_net.predict(latents)
                if config["value_prefix"]:
                    new_latents, pred_reward_logits, output_hiddens = mu_net.dynamics(
                        latents, one_hot_actions, output_hiddens
                    )
                else:
                    new_latents, pred_reward_logits = mu_net.dynamics(
                        latents, one_hot_actions
                    )
                self.print_timing("forward pass")

                # We scale down the gradient, I believe so that the gradient at the base of the unrolled
                # network converges to a maximum rather than increasing linearly with depth
                new_latents.register_hook(lambda grad: grad * 0.5)

                # target_reward_sup_i = scalar_to_support(
                #     target_reward_stepi, half_width=config["support_width"]
                # )

                # target_value_sup_i = scalar_to_support(
                #     target_value_stepi, half_width=config["support_width"]
                # )

                # Cutting off cases where there's not enough data for a full rollout

                # The muzero paper calculates the loss as the squared difference between scalars
                # but CrossEntropyLoss is used here for a more stable value loss when large values are encountered

                pred_values = support_to_scalar(
                    torch.softmax(pred_value_logits[screen_t], dim=1)
                )
                pred_rewards = support_to_scalar(
                    torch.softmax(pred_reward_logits[screen_t], dim=1)
                )
                self.print_timing("support to scalar")
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
                self.print_timing("done losses")
            # Aggregate the losses to a single measure
            batch_loss = (
                batch_policy_loss
                + batch_reward_loss
                + (batch_value_loss * config["val_weight"])
                + (batch_consistency_loss * config["consistency_weight"])
            )
            batch_loss = batch_loss.mean()
            self.print_timing("batch loss")

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
            self.print_timing("optimizer")

            total_loss += batch_loss
            total_value_loss += batch_value_loss
            total_policy_loss += batch_policy_loss
            total_reward_loss += batch_reward_loss
            total_consistency_loss += batch_consistency_loss
            self.print_timing("loss")

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
            self.print_timing("to_cpu")

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
            if self.config["train_speed_profiling"]:
                print(f"WHOLE BATCH: {time.time() - st}")
            self.print_timing("saving/end")

        return metrics_dict

    def print_timing(self, tag, min_time=0.05):
        if self.config["train_speed_profiling"]:
            now = datetime.datetime.now()
            print(f"{tag:20} {now - self.last_time}")
            self.last_time = now


def test_whole_game(mu_net, memory):
    ndx = ray.get(memory.get_buffer_ndxs.remote())[0]
    game = ray.get(memory.get_buffer_ndx.remote(ndx))
    for i in range(50):
        ims, acts, vals, rewards, _, _ = game.make_target(i, 5, 5)

        obs = torch.tensor(ims[0], dtype=torch.float32).unsqueeze(0)
        if i == 20:
            pickle.dump(obs, open("whole_game20.pkl", "wb"))
        acts = torch.tensor(acts[1], dtype=torch.int64).unsqueeze(0)
        acts = nn.functional.one_hot(acts, num_classes=mu_net.action_size)

        latent = mu_net.represent(obs)
        _, val_logits = mu_net.predict(latent)
        _, reward_logits = mu_net.dynamics(latent, acts)
        val_logits = val_logits.detach()
        reward_logits = reward_logits.detach()
        rew = support_to_scalar(torch.softmax(reward_logits, dim=1))
        val = support_to_scalar(torch.softmax(val_logits, dim=1))

        print(
            f"t reward: {rewards[0]}, p reward: {rew}, t val: {vals[0]}, p val: {val}"
        )


def get_test_graph(mu_net, memory, discrete=False):
    xvals = np.arange(0, 7)
    yvals = [get_test_numbers(mu_net, i, discrete, memory) for i in xvals]
    print(yvals)
    plt.plot(xvals, yvals)
    plt.ylim(0, 6)


def get_test_numbers(mu_net, i, discrete, memory):
    # if discrete:
    #     val = i
    #     shape = [4]
    #     obs = torch.full(shape, val, dtype=torch.float32).unsqueeze(0)
    # else:
    ndx = ray.get(memory.get_buffer_ndxs.remote())[10]
    game = ray.get(memory.get_buffer_ndx.remote(ndx))
    ims, acts, _, rewards, _, _ = game.make_target(21 + i, 5, 5)

    print(rewards[0], ims[0].shape, type(ims[0]))
    # print([np.mean(im, axis=(1, 2)) for im in ims])
    obs = torch.tensor(ims[4], dtype=torch.float32).unsqueeze(0)
    acts = torch.tensor(acts[4], dtype=torch.int64).unsqueeze(0)
    acts = nn.functional.one_hot(acts, num_classes=mu_net.action_size)

    print(i, obs.shape, torch.mean(obs))
    reward_logits = mu_net.dynamics(mu_net.represent(obs), acts)[1].detach()
    print(reward_logits.shape, reward_logits)
    return support_to_scalar(
        torch.softmax(
            reward_logits,
            dim=1,
        )
    )
