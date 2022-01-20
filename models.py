import torch
import torch.nn as nn


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

    def forward(x):  # inputs are 96x96??
        out = torch.relu(self.batch_norm1(self.conv1(x)))  # outputs 48x48

        out = self.res1(out)

        out = self.res_down(out)  # outputs 24x24

        out = self.res2(out)
        out = self.av_pool1(out)  # outputs 12x12
        out = self.res3(out)
        out = self.av_pool2(out)  # outputs 6x6
        out = self.res4(out)
