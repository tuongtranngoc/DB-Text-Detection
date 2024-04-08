from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2
import glob
import argparse

import time
import numpy as np

from . import config as cfg
from src.utils.logger import Logger
from src.utils.data_utils import DataUtils
from src.utils.visualization import Visualization
from src.utils.post_processing import DBPostProcess
from src.models.diff_binarization import DiffBinarization

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

logger = Logger.get_logger("PREDICTION")


class Predictor:
    def __init__(self, args) -> None:
        self.args = args
        self.model = DiffBinarization(pretrained=False)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)['model'])
        self.model.to(self.args.device)
        self.model.eval()
        self.post_process = DBPostProcess(box_thresh=0.5)
        self.image_size = cfg['Train']['dataset']['transforms']['image_shape']
        self.transform = A.Compose([
            A.Resize(self.image_size[1], self.image_size[2]),
            A.Normalize(always_apply=True),
            ToTensorV2()])
        
    def preprocess(self, img_path):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)[..., ::-1]
            img = self.transform(image=img)['image']
            img = img.unsqueeze(0)
            return img
        else:
            Exception("Not exist image path")

    def predict(self, img_path):
        save_dir = args.save_dir
        img = self.preprocess(img_path).to(self.args.device)
        with torch.no_grad():
            st = time.time()
            preds = self.model(img)
            print(time.time()-st)
            _img = img.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            boxes, scores = self.post_process(_img, preds, True)
            boxes, scores = boxes[0], scores[0]
            idxs = np.where(np.array(scores)>args.threshold)[0]
            boxes = [boxes[i].tolist() for i in idxs]
        
        img = DataUtils.image_to_numpy(img)
        img = Visualization.draw_polygon(img, boxes)
        basename = os.path.basename(img_path)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/{basename}", img)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=cfg['Debug']['prediction'])
    parser.add_argument("--image_path", type=str, default=None, help="Path to image file")
    parser.add_argument("--threshold", type=str, default=0.2, help="Bounding box threshold")
    parser.add_argument("--device", type=str, default='cuda', help="device inference (cuda or cpu)")
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'], help="Path to model checkpoint")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    predictor = Predictor(args)
    predictor.predict(args.image_path)
    