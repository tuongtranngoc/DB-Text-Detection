from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import cfg

import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from shapely.geometry import Polygon


class BatchMeter(object):
    """Calculate average/sum value after each time
    """
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.value = 0
        self.count = 0
    
    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        
    def get_value(self, summary_type=None):
        if summary_type == 'mean':
            return self.avg
        elif summary_type == 'sum':
            return self.sum
        else:
           return self.value


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