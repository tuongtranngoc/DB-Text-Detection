from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as cfg

import os
import argparse
from tqdm import tqdm

from src.utils.logger import Logger
from src.utils.data_utils import DataUtils
from src.data.dataset import ICDAR2015Dataset
from src.utils.post_processing import PostProcessor
from src.utils.metrics import BatchMeter, AccuracyMetric
from src.models.diff_binarization import DiffBinarization
from src.models.losses.db_loss import DiffBinarizationLoss


logger = Logger.get_logger("EVALUATION")


class Evaluator:
    def __init__(self, valid_dataset, model, args) -> None:
        self.args = args
        self.model = model
        self.model.to(args.device)
        self.valid_dataset = valid_dataset
        self.loss_func = DiffBinarizationLoss()
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=self.args.shuffle,
                                       num_workers=self.args.num_workers,
                                       pin_memory=self.args.pin_memory)
        self.acc = AccuracyMetric()
        self.post_process = PostProcessor()

    def eval(self):
        metrics = {
            "shrink_maps_loss": BatchMeter(),
            "thresh_maps_loss": BatchMeter(),
            "binary_maps_loss": BatchMeter(),
            "total_loss": BatchMeter(),
            "map": BatchMeter(),
            "map_50": BatchMeter(),
            "map_75": BatchMeter()
        }
        self.model.eval()

        for i, (images, labels) in enumerate(self.valid_loader):
            with torch.no_grad():
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels)
                preds = self.model(images)
                loss = self.loss_func(preds, labels)
                metrics['shrink_maps_loss'].update(loss["loss_shrink_maps"])
                metrics['thresh_maps_loss'].update(loss["loss_thresh_maps"])
                metrics['binary_maps_loss'].update(loss["loss_binary_maps"])
                metrics['total_loss'].update(loss["total_loss"])

                boxes, scores = self.post_process(images, preds, True)
                classes = torch.ones_like(scores, dtype=scores.dtype)
                gt_score = torch.ones_like(scores, dtype=scores.dtype)
                self.acc.compute_acc(boxes, scores, classes, images, gt_score, classes)

        acc = self.acc.map_mt.compute()
        metrics['map'].update(acc['map'])
        metrics['map_50'].update(acc['map_50'])
        metrics['map_75'].update(acc['map_75'])

        logger.info(f'shrink_maps_loss: {metrics["shrink_maps_loss"].get_value("mean"): .3f},
                    thresh_maps_loss: {metrics["thresh_maps_loss"].get_value("mean"): .3f},
                    binary_maps_loss: {metrics["binary_maps_loss"].get_value("mean"): .3f},
                    total_loss: {metrics["total_loss"].get_value("mean"): .3f}')

        logger.info(f'mAP: {metrics["map"].get_value("mean"): .3f},
                    mAP_50: {metrics["map_50"].get_value("mean"): .3f},
                    mAP_75: {metrics["map_75"].get_value("mean"): .3f}')


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=cfg['Eval']['loader']['batch_size'])
    parser.add_argument("--shuffle", type=bool, default=cfg['Eval']['loader']['shuffle'])
    parser.add_argument("--num_workers", type=int,  default=cfg['Eval']['loader']['num_workers'])
    parser.add_argument("--pin_memory", type=bool, default=cfg['Eval']['loader']['use_shared_memory'])
    parser.add_argument("--device", type=str, default=cfg['Global']['device'])
    parser.add_argument("--lr", type=float, default=cfg['Optimizer']['lr'])
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'])
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    valid_dataset = ICDAR2015Dataset(mode="Eval")
    model = DiffBinarization()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device)['model'])
    evaluate = Evaluator(valid_dataset, model, args)
    evaluate.eval()
