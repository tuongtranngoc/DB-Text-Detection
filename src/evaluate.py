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
from src.utils.metrics import BatchMeter, AccuracyMetric
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
        self.acc = AccuracyMetric()
        self.post_process = DBPostProcess()

    def eval(self) -> dict:
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
                preds = preds.cpu().detach().numpy()
                images = images.cpu().detach().numpy()
                boxes, scores = self.post_process(images, preds)
                
                for box, score, label in zip(boxes, scores, labels):
                    mask = label[0].cpu().detach().numpy()
                    mask = mask.astype(np.int32)[np.newaxis, :, :]
                    if len(box) == 0 or len(score) == 0:
                        box = np.zeros_like(mask, dtype=mask.dtype)
                        score = np.array([1.0], dtype=np.float32)
                    else:
                        box = np.array(box, dtype=mask.dtype)
                        score = np.array(score, dtype=np.float32)

                    classes = np.zeros_like(score, dtype=np.int32)
                    self.acc.compute_acc(box, score, classes, mask, classes)

        acc = self.acc.map_mt.compute()
        metrics['map'].update(acc['map'])
        metrics['map_50'].update(acc['map_50'])
        metrics['map_75'].update(acc['map_75'])

        logger.info(f'mAP: {metrics["map"].get_value("mean"): .3f} - mAP_50: {metrics["map_50"].get_value("mean"): .3f} - mAP_75: {metrics["map_75"].get_value("mean"): .3f}')
        
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
