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


class ResNet18(nn.Module):
    def __init__(self):
        cfg = [(2, 64), (2, 128), (2, 256), (2, 512)]
        super(ResNet18, self).__init__()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(cfg):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
    
    def b1(self): 
        return nn.Sequential( 
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), 
            nn.LazyBatchNorm2d(), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 or not first_block:
                blk.append(ResidualBlock(num_channels, use_1x1_conv=True, strides=2))
            else:
                blk.append(ResidualBlock(num_channels))
        return nn.Sequential(*blk)
    
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    from torchsummary import summary
    net = ResNet18()
    x = torch.randn(1, 3, 640, 640)
    y = net(x)
    import ipdb; ipdb.set_trace()    

