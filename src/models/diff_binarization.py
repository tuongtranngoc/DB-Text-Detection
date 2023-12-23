from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
import torch.nn.functional as F

from . import *


class DiffBinarization(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackboneDB()
        self.neck = NeckDB(self.backbone.out_channels)
        self.head = HeadDB(self.neck.out_channels)
    
    def forward(self, x):
        __, __, H, W  = x.size()
        y = self.backbone(x)
        y = self.neck(y)
        y = self.head(y)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y
    

if __name__ == "__main__":
    DB = DiffBinarization()
    x = torch.randn(2, 3, 640, 640)
    y = DB(x)
    import ipdb; ipdb.set_trace()