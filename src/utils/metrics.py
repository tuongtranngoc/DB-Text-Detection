from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import cfg

import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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


class AccuracyMetric(object):
    def __init__(self):
        self.map_mt = MeanAveragePrecision(box_format='xyxy', iou_type='segm')

    def compute_acc(self, pred_mask, pred_score, pred_class, gt_mask, gt_score, gt_class):
        """Mean Average Precision (mAP)
        Reference: https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
        """
        preds = [{
            "masks": pred_mask,
            "scores": pred_score,
            "labels": pred_class
        }]
        target = [{
            "masks": gt_mask,
            "scores": gt_score,
            "labels": gt_class
        }]
        self.map_mt.update(preds, target)
