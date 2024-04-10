from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
import torch.nn.functional as F

from . import *

set_logger_tag(logger, 'MODEL')


class DiffBinarization(nn.Module):
    def __init__(self, pretrained=True, backbone='deformable_resnet50'):
        super(DiffBinarization, self).__init__()
        if backbone == 'deformable_resnet50':
            self.backbone = deformable_resnet50(pretrained=pretrained, in_channels=3)
        elif backbone == 'deformable_resnet50':
            self.backbone = deformable_resnet18(pretrained=pretrained, in_channels=3)
        elif backbone == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained, in_channels=3)
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained, in_channels=3)
        else:
            logger.warning(f'Not exist backbone {backbone}')
            
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
    import ipdb; ipdb.set_trace()
    y = DB(x)