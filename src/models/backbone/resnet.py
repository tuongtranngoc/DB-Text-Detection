from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class BackboneDB(nn.Module):
    def __init__(self) -> None:
        super(BackboneDB, self).__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        # Features
        self.layer1 = backbone.layer1 # 1/4
        self.layer2 = backbone.layer2 # 1/8
        self.layer3 = backbone.layer3 # 1/16
        self.layer4 = backbone.layer4 # 1/32
        # Out channels
        self.out_channels = [self.layer1[-1].conv2.out_channels,
                             self.layer2[-1].conv2.out_channels,
                             self.layer3[-1].conv2.out_channels,
                             self.layer4[-1].conv2.out_channels]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        return x2, x3, x4, x5


if __name__ == "__main__":
    x = torch.randn(1, 3, 640, 640)
    net = BackboneDB()
    y = net(x)
    import ipdb; ipdb.set_trace()