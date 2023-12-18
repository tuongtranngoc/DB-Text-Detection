from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeckDB(nn.Module):
    def __init__(self, backbone) -> None:
        super(NeckDB, self).__init__()

    def forward(self, x):
        pass


