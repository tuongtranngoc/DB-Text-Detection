from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, use_1x1_conv=False, strides=1) -> None:
        super().__init__()

        self.conv_1 = nn.LazyConv2d(out_channels=num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv_2 = nn.LazyConv2d(out_channels=num_channels, kernel_size=3, padding=1)
        self.bn_1 = nn.LazyBatchNorm2d()
        self.bn_2 = nn.LazyBatchNorm2d()

        if use_1x1_conv:
            self.conv_1x1 = nn.LazyConv2d(out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv_1x1 = None

    def forward(self, x):
        y = F.relu(self.bn_1(self.conv_1(x)))
        y = self.bn_2(self.conv_2(y))
        if self.conv_1x1:
            x = self.conv_1x1(x)
        y += x
        y = F.relu(y)
        return y

if __name__ == "__main__":
    from torchsummary import summary
    residual = ResidualBlock(6, True, 2)
    x = torch.randn(1, 3, 32, 32)
    y = residual(x)
    import ipdb; ipdb.set_trace()    

