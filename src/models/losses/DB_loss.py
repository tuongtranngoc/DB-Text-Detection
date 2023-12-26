from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import *


class DiffBinarizationLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, hn_ratio=3, eps=1e-6):
        self.hn_ratio = hn_ratio
        self.alpha = alpha
        self.beta = beta
        self.probability_map_loss = BCELoss(self.hn_ratio, eps=eps)
        self.binary_map_loss = DiceLoss(eps=eps)
        self.threshold_map_loss = L1Loss(eps=eps)
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        gt_shrink_maps, gt_shrink_masks, gt_thresh_maps, gt_thresh_masks = gt
        pred_shrink_maps = pred[:, 0, ...]
        pred_thresh_maps = pred[:, 1, ...]
        pred_binary_maps = pred[:, 2, ...]

        loss_shrink_maps = self.probability_map_loss(pred_shrink_maps, gt_shrink_maps, gt_shrink_masks)
        loss_thresh_maps = self.threshold_map_loss(pred_thresh_maps, gt_thresh_maps, gt_thresh_masks)
        if pred.size()[1] > 2:
            loss_binary_maps = self.binary_map_loss(pred_binary_maps, gt_shrink_maps, gt_shrink_masks)
            