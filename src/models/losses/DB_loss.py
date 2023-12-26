from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffBinarizationLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10):
        self.probability_map_loss = nn.BCELoss()
        self.binary_map_loss = nn.BCELoss()
    
    def forward(self, x):
        pass