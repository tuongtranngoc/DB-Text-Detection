from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *

import cv2
import torch
import numpy as np


class DataUtils:
    @classmethod
    def polygon2xyxy(cls, polygon):
        xyxy = np.array(polygon, dtype=np.float32)
        xyxy = xyxy.reshape((4, 2))
        xmin = xyxy[:, 0].min()
        ymin = xyxy[:, 1].min()
        xmax = xyxy[:, 0].max()
        ymax = xyxy[:, 1].max()

        return [xmin, ymin, xmax, ymax]
    
    @classmethod
    def denormalize(cls, image):
        mean = np.array(cfg['Train']['dataset']['transforms']['mean'], dtype=np.float32)
        std = np.array(cfg['Train']['dataset']['transforms']['std'], dtype=np.float32)
        image *= (std * 255.)
        image += (mean * 255.)
        image = np.clip(image, 0, 255.)
        return image
    
    @classmethod
    def image_to_numpy(cls, image):
        if isinstance(image, torch.Tensor):
            if image.dim() > 3:
                image = image.squeeze()
            image = image.detach().cpu().numpy()
            image = image.transpose((1, 2, 0))
            image = cls.denormalize(image)
            image = np.ascontiguousarray(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        elif isinstance(image, np.ndarray):
            image = cls.denormalize(image)
            image = np.ascontiguousarray(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        else:
            raise Exception(f"{image} is a type of {type(image)}, not numpy/tensor type")
    
    @classmethod
    def order_points_clockwise(cls, points: np.ndarray):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect
    
    
