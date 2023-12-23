from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
import torch.nn.functional as F

from . import *

class DifferentialBinarization(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackboneDB()
        

    def forward(self, x):
        pass