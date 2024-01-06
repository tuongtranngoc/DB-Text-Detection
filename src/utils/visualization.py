from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import numpy as np

import torch
import torch.nn as nn

from . import *


class Visualization:
    C, H, W = cfg['Train']['dataset']['transforms']['image_shape']
    device = cfg['Global']['device']
    
    @classmethod
    def draw_polygon(cls, image, polygon, color=(255, 0, 0)):
        # polygon: x1, y1, x2, y2, x3, y3, x4, y4
        isClosed = True
        thickness = 2
        image = np.array(image, np.uint8)
        polygon = np.array(polygon, np.int32)
        if polygon.ndim <= 2:
            polygon = [polygon]
        for pts in polygon:
            pts = pts.reshape((-1, 1, 2))
            image = cv2.polylines(image, [pts], isClosed, color, thickness)
        return image
    
    @classmethod
    def save_debug(cls, image, save_dir, basename):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, basename)
        cv2.imwrite(save_path, image)

    @classmethod
    def model_debug(cls, model:nn.Module):
        from torchview import draw_graph
        x = torch.randn(size=(cls.C, cls.H, cls.W)).to(cls.device)
        model.to(cls.device)
        draw_graph(model, input_size=x.unsqueeze(0).shape,
                   expand_nested=True,
                   save_graph=True,
                   directory=cfg['Debug']['model'],
                   graph_name='resnet18')