# Understanding of Architecture Network

The Differential Binarization (DB) Algorithm is one of the cutting-edge approaches to effectively detect curved text. 
+ Improved Text Detection: The algorithm excels at accurately identifying text within images, even when it's curved or distorted.
+ Accurate Text Recognition: It paves the way for more precise text recognition, ensuring that the text is correctly extracted and understood.
  
DB works quite well when using a lightweight backbone, which significantly enhances the detection performance with a backbone of ResNet-18. In the Neck network, the features are up-sampled to the same scale and cascaded to produce feature $F$. Then, feature $F$ is used to predict both the probability map $P$ and threshold map $T$ in the Head network.
After that, the approximate binary map $B$ is calculated by $P$ and $F$.

<p align="center">
    <img src="../../images/architecture.png">
</p>

## 1. Backbone

First, we use extracted features from the different scales of layers (layers 1, 2, 3, 4) in the ResNet-18 backbone

```python
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
```

## 2. Neck Network
Next, the features are up-sampled to the same scale by using Conv+Bn+ReLu module