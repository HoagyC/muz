import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class CartRepr(nn.Module):
    def __init__(self, obs_size, latent_size):
        super().__init__()
        self.obs_size = obs_size
        self.fc1 = nn.Linear(obs_size, latent_size)
        self.fc2 = nn.Linear(latent_size, latent_size)

    def forward(self, state):
        assert state.dim() == 2
        assert state.shape[1] == self.obs_size
        state = state.to(dtype=torch.float32)
        out = self.fc1(state)
        out = torch.relu(out)
        out = self.fc2(out)
        return out


class CartDyna(nn.Module):
    def __init__(self, action_size, latent_size, support_width):
        self.latent_size = latent_size
        self.action_size = action_size
        self.full_width = (2 * support_width) + 1
        super().__init__()
        self.fc1 = nn.Linear(latent_size + action_size, latent_size)
        self.fc2 = nn.Linear(latent_size, latent_size + self.full_width)

    def forward(self, latent, action):
        assert latent.dim() == 2 and action.dim() == 2
        assert (
            latent.shape[1] == self.latent_size and action.shape[1] == self.action_size
        )

        out = torch.cat([action, latent], dim=1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        new_latent = out[:, : self.latent_size]
        reward_logits = out[:, self.latent_size :]
        return new_latent, reward_logits


class CartPred(nn.Module):
    def __init__(self, action_size, latent_size, support_width):
        super().__init__()
        self.action_size = action_size
        self.latent_size = latent_size
        self.full_width = (support_width * 2) + 1
        self.fc1 = nn.Linear(latent_size, latent_size)
        self.fc2 = nn.Linear(latent_size, action_size + self.full_width)

    def forward(self, latent):
        assert latent.dim() == 2
        assert latent.shape[1] == self.latent_size
        out = self.fc1(latent)
        out = torch.relu(out)
        out = self.fc2(out)
        policy_logits = out[:, : self.action_size]
        value_logits = out[:, self.action_size :]
        return policy_logits, value_logits


class MuZeroCartNet(nn.Module):
    def __init__(self, action_size: int, obs_size, config: dict):
        super().__init__()
        self.config = config
        self.action_size = action_size
        self.obs_size = obs_size
        self.latent_size = config["latent_size"]
        self.support_width = config["support_width"]

        self.pred_net = CartPred(self.action_size, self.latent_size, self.support_width)
        self.dyna_net = CartDyna(self.action_size, self.latent_size, self.support_width)
        self.repr_net = CartRepr(self.obs_size, self.latent_size)

        self.policy_loss = nn.CrossEntropyLoss(reduction="sum")
        self.reward_loss = nn.CrossEntropyLoss(reduction="sum")
        self.value_loss = nn.CrossEntropyLoss(reduction="sum")
        self.cos_sim = nn.CosineSimilarity(dim=0)

    def consistency_loss(self, x1, x2):
        assert x1.shape == x2.shape
        return -self.cos_sim(x1.view(-1), x2.view(-1))

    def init_optim(self, lr):
        params = (
            list(self.pred_net.parameters())
            + list(self.dyna_net.parameters())
            + list(self.repr_net.parameters())
        )
        self.optimizer = torch.optim.SGD(
            params, lr=lr, weight_decay=self.config["weight_decay"], momentum=0.9
        )

    def predict(self, latent):
        policy, value = self.pred_net(latent)
        return policy, value

    def dynamics(self, latent, action):
        latent, reward = self.dyna_net(latent, action)
        return latent, reward

    def represent(self, observation):
        latent = self.repr_net(observation)
        return latent


class MuZeroAtariNet(nn.Module):
    def __init__(self, action_size, obs_size, config):
        super().__init__()
        self.config = config

        self.x_pad, self.y_pad = 0, 0
        assert len(obs_size) == 3

        self.y_size, self.x_size, self.n_channels = obs_size

        if self.x_size % 16 != 0:
            self.x_pad = 16 - (obs_size[1] % 16)

        if self.y_size % 16 != 0:
            self.y_pad = 16 - (obs_size[0] % 16)

        self.x_size_final = math.ceil(obs_size[1] / 16)
        self.y_size_final = math.ceil(obs_size[0] / 16)

        self.action_size = action_size
        self.obs_size = obs_size
        self.latent_depth = config["latent_size"]
        self.support_width = config["support_width"]
        self.latent_area = self.x_size_final * self.y_size_final

        self.dyna_net = AtariDynamicsNet(
            self.latent_depth, self.support_width, self.latent_area, self.action_size
        )
        self.pred_net = AtariPredictionNet(
            self.latent_depth, self.support_width, self.latent_area, self.action_size
        )
        self.repr_net = AtariRepresentationNet(
            self.x_pad, self.y_pad, self.latent_depth
        )

        self.policy_loss = nn.CrossEntropyLoss()
        self.reward_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.CrossEntropyLoss()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def consistency_loss(self, x1, x2):
        assert x1.shape == x2.shape
        return -self.cos_sim(x1.view(-1), x2.view(-1))

    def init_optim(self, lr):
        params = (
            list(self.pred_net.parameters())
            + list(self.dyna_net.parameters())
            + list(self.repr_net.parameters())
        )
        self.optimizer = torch.optim.SGD(
            params, lr=lr, weight_decay=self.config["weight_decay"], momentum=0.9
        )

    def predict(self, latent):
        policy, value = self.pred_net(latent)
        return policy, value

    def dynamics(self, latent, action):
        latent, reward = self.dyna_net(latent, action)
        return latent, reward

    def represent(self, observation):
        latent = self.repr_net(observation)
        return latent


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ResBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


# class ResBlock(nn.Module):
#     def __init__(
#         self, in_channels, out_channels=None, downsample=None, momentum=0.1, stride=1
#     ):
#         super().__init__()
#         if not out_channels:
#             out_channels = in_channels

#         self.conv1 = nn.Conv2d(
#             in_channels, out_channels, stride=stride, padding=1, kernel_size=3
#         )[
#         self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels, momentum=momentum)

#         self.conv2 = nn.Conv2d(
#             out_channels, out_channels, stride=stride, padding=1, kernel_size=3
#         )
#         self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x
#         out = torch.relu(self.batch_norm1(self.conv1(x)))
#         out = self.batch_norm2(self.conv2(out))]

#         if self.downsample is not None:
#             identity = self.downsample(iout)

#         out = out + identity
#         out = torch.relu(out)
#         return out


class AtariRepresentationNet(nn.Module):
    def __init__(self, x_pad, y_pad, latent_depth):
        super().__init__()

        self.pad = (0, x_pad, 0, y_pad)

        self.conv1 = nn.Conv2d(3, 10, stride=2, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=10, momentum=0.1)

        self.res1 = ResBlock(10)

        self.conv2 = nn.Conv2d(10, 10, stride=2, kernel_size=3, padding=1)

        self.res2 = ResBlock(10)
        self.av_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.res3 = ResBlock(10)
        self.av_pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.res4 = ResBlock(10)

        self.conv3 = nn.Conv2d(10, latent_depth, stride=1, kernel_size=3, padding=1)

    def forward(self, x):  # inputs are 96x96??
        x = x.to(dtype=torch.float32)
        out = F.pad(x, self.pad, "constant", 0)
        out = torch.relu(self.batch_norm1(self.conv1(out)))  # outputs 48x48

        out = self.res1(out)

        out = self.conv2(out)  # outputs 24x24

        out = self.res2(out)
        out = self.av_pool1(out)  # outputs 12x12
        out = self.res3(out)
        out = self.av_pool2(out)  # outputs 6x6
        out = self.res4(out)
        out = self.conv3(out)
        return out


class AtariDynamicsNet(nn.Module):
    def __init__(
        self,
        latent_depth,
        support_width,
        latent_area,
        action_space_size,
        reward_head_width=50,
    ):
        super().__init__()
        self.latent_depth = latent_depth
        self.full_support_width = (support_width * 2) + 1
        self.latent_area = latent_area

        self.conv1 = nn.Conv2d(
            latent_depth + action_space_size, latent_depth, kernel_size=3, padding=1
        )
        self.res1 = ResBlock(latent_depth)

        self.fc1 = nn.Linear(latent_area * latent_depth, reward_head_width)
        self.fc2 = nn.Linear(reward_head_width, self.full_support_width)

    def forward(self, latent, actions_one_hot):
        # Receives 2D actions of batch_size x action_space_size
        action_images = torch.ones(latent.shape[0], latent.shape[2], latent.shape[3])

        action_images_spread = torch.einsum(
            "bhw,ba->bahw", action_images, actions_one_hot
        )  # Spread the one-hot action over the first dim to make a channel for each possible action

        res_input = torch.cat((latent, action_images_spread), dim=1)

        batch_size = latent.shape[0]
        out = self.conv1(res_input)
        new_latent = self.res1(out)

        out = new_latent.view(batch_size, -1)
        reward_logits = self.fc2(torch.relu(self.fc1(out)))

        return new_latent, reward_logits


class AtariPredictionNet(nn.Module):
    def __init__(
        self,
        latent_depth,
        support_width,
        latent_area,
        action_size,
        prediction_head_width=50,
    ):
        super().__init__()

        self.latent_depth = latent_depth
        self.full_support_width = (support_width * 2) + 1
        self.latent_area = latent_area

        self.fc1 = nn.Linear(latent_area * latent_depth, prediction_head_width)
        self.fc_policy = nn.Linear(prediction_head_width, action_size)
        self.fc_value = nn.Linear(prediction_head_width, self.full_support_width)

    def forward(self, x):
        batch_size = x.shape[0]
        out = x.view(batch_size, -1)

        out = torch.relu(self.fc1(out))
        policy_logits = self.fc_policy(out)
        value_logits = self.fc_value(out)
        return policy_logits, value_logits


def support_to_scalar(support, epsilon=0.001):
    half_width = int((len(support) - 1) / 2)
    vals = torch.Tensor(range(-half_width, half_width + 1))

    # Dot product of the two
    out_val = torch.einsum("i,i -> ", vals, support)

    sign_out = 1 if out_val >= 0 else -1

    num = torch.sqrt(1 + 4 * epsilon * (torch.abs(out_val) + 1 + epsilon)) - 1
    res = (num / (2 * epsilon)) ** 2

    output = sign_out * (res - 1)

    return output


def scalar_to_support(scalar: torch.Tensor, epsilon=0.001, half_width: int = 10):
    # Scaling the value function and converting to discrete support as found in
    # Appendix F if MuZero

    sign_x = torch.where(scalar >= 0, 1, -1)
    h_x = sign_x * (torch.sqrt(torch.abs(scalar) + 1) - 1 + epsilon * scalar)

    h_x.clamp_(-half_width, half_width)

    upper_ndxs = (torch.ceil(h_x) + half_width).to(dtype=torch.int64)
    lower_ndxs = (torch.floor(h_x) + half_width).to(dtype=torch.int64)
    ratio = h_x % 1
    support = torch.zeros(*scalar.shape, 2 * half_width + 1)

    support.scatter_(1, upper_ndxs.unsqueeze(1), ratio.unsqueeze(1))
    # do lower ndxs second as if lower==upper, ratio = 0, 1 - ratio = 1
    support.scatter_(1, lower_ndxs.unsqueeze(1), (1 - ratio).unsqueeze(1))

    return support
