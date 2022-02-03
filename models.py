import torch
import torch.nn as nn
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
        self.action_size = action_size
        self.obs_size = obs_size
        self.latent_size = config["latent_size"]
        self.support_width = config["support_width"]

        self.pred_net = CartPred(self.action_size, self.latent_size, self.support_width)
        self.dyna_net = CartDyna(self.action_size, self.latent_size, self.support_width)
        self.repr_net = CartRepr(self.obs_size, self.latent_size)

        self.policy_loss = nn.CrossEntropyLoss()
        self.reward_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.CrossEntropyLoss()

    def init_optim(self, lr):
        params = (
            list(self.pred_net.parameters())
            + list(self.dyna_net.parameters())
            + list(self.repr_net.parameters())
        )
        self.optimizer = torch.optim.SGD(params, lr=lr, weight_decay=1e-4, momentum=0.9)

    def predict(self, latent):
        policy, value = self.pred_net(latent)
        return policy, value

    def dynamics(self, latent, action):
        latent, reward = self.dyna_net(latent, action)
        return latent, reward

    def represent(self, observation):
        latent = self.repr_net(observation)
        return latent


class ResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, downsample=None, momentum=0.1, stride=1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, stride=stride, padding=1, kernel_size=3
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels, momentum=momentum)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, stride=stride, padding=1, kernel_size=3
        )
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = torch.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = torch.relu(out)
        return out


class RepresentationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, stride=2, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=32, momentum=0.1)

        self.res1 = ResBlock(32)

        self.conv2 = nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1)
        self.res_down = ResBlock(32, 64, downsample=self.conv2)

        self.res2 = ResBlock(64)
        self.av_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.res3 = ResBlock(64)
        self.av_pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.res4 = ResBlock(64)

    def forward(self, x):  # inputs are 96x96??
        out = torch.relu(self.batch_norm1(self.conv1(x)))  # outputs 48x48

        out = self.res1(out)

        out = self.res_down(out)  # outputs 24x24

        out = self.res2(out)
        out = self.av_pool1(out)  # outputs 12x12
        out = self.res3(out)
        out = self.av_pool2(out)  # outputs 6x6
        out = self.res4(out)


# this needs to take a a block representing the current or future state
# which in this model is a 6x6x64 tensor
# and return a policy and a value
# value is intended to predict the n-step reward
# policy predicts the policy that will actually be undertaken at that future time
# so that we know where to go in the future
class PredictionNetwork(nn.Module):
    def __init__(self, action_size):
        pass


# this takes a block representing a game state, and an action
# and returns a block of the same shape
class DynamicsNetwork(nn.Module):
    def __init__(self, action_size):
        block_size = 6 * 6 * 64
        self.fc1 = nn.Linear(block_size + action_size, block_size)
        self.fc2 = nn.Linear(block_size, block_size)

    def forward(self, block, action):
        flat_block = block.flatten()
        out = torch.cat(flat_block, action)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.view(6, 6, 64)
        return out


def support_to_scalar(support, epsilon=0.001):
    half_width = int((len(support) - 1) / 2)
    vals = torch.Tensor(range(-half_width, half_width + 1))
    out_val = torch.einsum("i,i -> ", vals, support)

    sign_out = 1 if out_val >= 0 else -1

    num = torch.sqrt(1 + 4 * epsilon * (torch.abs(out_val) + 1 + epsilon)) - 1
    res = (num / (2 * epsilon)) ** 2

    output = sign_out * (res - 1)

    return output


def scalar_to_support(scalar: torch.Tensor, epsilon=0.001, max_val: int = 10):
    # Scaling the value function and converting to discrete support as found in
    # Appendix F if MuZero

    sign_x = 1 if scalar >= 0 else -1
    h_x = torch.sign(scalar) * (
        torch.sqrt(torch.abs(scalar) + 1) - 1 + epsilon * scalar
    )

    h_x.clamp_(-max_val, max_val)

    upper_ndx = (torch.ceil(h_x) + max_val).to(dtype=torch.int64)
    lower_ndx = (torch.floor(h_x) + max_val).to(dtype=torch.int64)
    ratio = h_x % 1
    support = torch.zeros(2 * max_val + 1)

    if upper_ndx == lower_ndx:
        support[upper_ndx] = 1
    else:
        support[lower_ndx] = 1 - ratio
        support[upper_ndx] = ratio

    return support
