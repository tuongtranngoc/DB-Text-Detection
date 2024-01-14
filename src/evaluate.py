from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2
import argparse
import numpy as np
from tqdm import tqdm

from . import config as cfg
from src.utils.logger import Logger
from src.utils.data_utils import DataUtils
from src.data.dataset import ICDAR2015Dataset
from src.utils.post_processing import DBPostProcess
from src.utils.metrics import BatchMeter, AccTorchMetric
from src.models.diff_binarization import DiffBinarization
from src.models.losses.db_loss import DiffBinarizationLoss


logger = Logger.get_logger("EVALUATION")


class Evaluator:
    def __init__(self, valid_dataset, model) -> None:
        self.args = cli()
        self.model = model
        self.model.to(self.args.device)
        self.valid_dataset = valid_dataset
        self.loss_func = DiffBinarizationLoss()
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=self.args.shuffle,
                                       num_workers=self.args.num_workers,
                                       pin_memory=self.args.pin_memory)
        self.acc = AccTorchMetric()
        self.post_process = DBPostProcess()
    
    def evaluate(self) -> dict:
        metrics = {
            "map": BatchMeter(),
            "map_50": BatchMeter(),
            "map_75": BatchMeter()
        }
        self.model.eval()
        for (images, labels) in tqdm(self.valid_loader):
            with torch.no_grad():
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels)
                preds = self.model(images)
                preds = preds.cpu().detach().numpy()
                images = images.cpu().detach().numpy()
                labels = [label.cpu().detach().numpy() for label in labels]
                pred_boxes, pred_scores = self.post_process(images, preds, False)
                gt_boxes, gt_scores = self.post_process(images, labels, False)
                for pred_box, pred_score, gt_box, gt_score in \
                    zip(pred_boxes, pred_scores, gt_boxes, gt_scores):
                    pred_class = np.zeros_like(pred_score, dtype=pred_score.dtype)
                    gt_class = np.zeros_like(gt_score, dtype=gt_score.dtype)
                    self.acc.compute_acc(pred_box, pred_score, pred_class, gt_box, gt_score, gt_class)

        avg_acc = self.acc.map_mt.compute()
        metrics['map'].update(avg_acc['map'])
        metrics['map_50'].update(avg_acc['map_50'])
        metrics['map_75'].update(avg_acc['map_75'])

        logger.info(f'map: {metrics["map"].get_value("mean"): .3f} - map_50: {metrics["map_50"].get_value("mean"): .3f} - map_75: {metrics["map_75"].get_value("mean"): .3f}')
        self.acc.map_mt.reset()
        
        return metrics


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
    evaluate = Evaluator(valid_dataset, model)
    evaluate.evaluate()