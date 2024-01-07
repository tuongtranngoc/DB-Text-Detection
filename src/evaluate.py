from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import argparse
import numpy as np
from tqdm import tqdm

from . import config as cfg
from src.utils.logger import Logger
from src.utils.data_utils import DataUtils
from src.data.dataset import ICDAR2015Dataset
from src.utils.post_processing import DBPostProcess
from src.models.diff_binarization import DiffBinarization
from src.models.losses.db_loss import DiffBinarizationLoss
from src.utils.metrics import BatchMeter, AccTorchMetric, PolygonEvaluator


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
        self.acc = PolygonEvaluator()
        self.post_process = DBPostProcess()

    def eval(self) -> dict:
        metrics = {
            "precision": BatchMeter(),
            "recall": BatchMeter(),
            "hmean": BatchMeter()
        }
        self.model.eval()
        accuracy = []
        for i, (images, labels) in enumerate(self.valid_loader):
            with torch.no_grad():
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels)
                preds = self.model(images)
                preds = preds.cpu().detach().numpy()
                images = images.cpu().detach().numpy()
                labels = [label.cpu().detach().numpy() for label in labels]
                pred_boxes, __ = self.post_process(images, preds)
                gt_boxes, __ = self.post_process(images, labels)

                for pred_box, gt_box in zip(pred_boxes, gt_boxes):
                    accuracy.append(self.acc.compute_acc(pred_box, gt_box))

        avg_acc = self.acc.combine_results(accuracy)
        metrics['precision'].update(avg_acc['precision'])
        metrics['recall'].update(avg_acc['recall'])
        metrics['hmean'].update(avg_acc['hmean'])

        logger.info(f'precision: {metrics["precision"].get_value("mean"): .3f} - recall: {metrics["recall"].get_value("mean"): .3f} - hmean: {metrics["hmean"].get_value("mean"): .3f}')
        
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
    evaluate.eval()
