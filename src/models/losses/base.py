from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, negative_ratio, eps=1e-6):
        super(BCELoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt:torch.Tensor, mask:torch.Tensor):
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = F.binary_cross_entropy(pred, gt, reduction="none")
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, __ = negative_loss.view(-1).topk(negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        return balance_loss
    

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, weights=None):
        pred = pred[:, 0, :, :]
        gt = gt[:, 0, :, :]
        if weights is not None:
            mask = weights * mask
        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss
    

class L1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        self.eps = eps
    
    def forward(self, pred:torch.Tensor, gt: torch.Tensor, mask:torch.Tensor):
        loss = (torch.abs(pred-gt) * mask).sum() / (mask.sum() + self.eps)
        return loss
