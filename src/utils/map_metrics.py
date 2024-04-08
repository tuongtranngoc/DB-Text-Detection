from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import cfg

import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from shapely.geometry import Polygon


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class AccTorchMetric(object):
    def __init__(self):
        self.map_mt = MeanAveragePrecision(iou_type='bbox', box_format='xyxy', class_metrics=True)
    
    def compute_acc(self, pred_mask, pred_score, pred_class, gt_mask, gt_score, gt_class):
        """Mean Average Precision (mAP)
        Reference: https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
        """
        preds = [{
            "boxes": torch.tensor(pred_mask, dtype=torch.float32),
            "scores": torch.tensor(pred_score, dtype=torch.float32),
            "labels": torch.tensor(pred_class, dtype=torch.long)
        }]
        target = [{
            "boxes": torch.tensor(gt_mask, dtype=torch.float32),
            "scores": torch.tensor(gt_score, dtype=torch.float32),
            "labels": torch.tensor(gt_class, dtype=torch.long)
        }]
        self.map_mt.update(preds, target)