from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from . import *


class Visualization:
    C, H, W = cfg['Train']['dataset']['transforms']['image_shape']
    device = cfg['Global']['device']
    post_process = DBPostProcess()
    
    @classmethod
    def draw_polygon(cls, image, polygon, color=(0, 0, 255)):
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
                   graph_name='resnet34')
    
    @classmethod
    def draw_heatmap(cls, basename, preds, org_img, debug_type=None):
        if isinstance(org_img , torch.Tensor):
            org_img = DataUtils.image_to_numpy(org_img)
        pred_prob = preds[0]
        pred_prob[pred_prob <= cfg['Global']['prob_threshold']] = 0
        pred_prob[pred_prob > cfg['Global']['prob_threshold']] = 1
        scaled_img = ((org_img - org_img.min()) * (1 / (org_img.max() - org_img.min()) * 255)).astype(np.uint8)
        
        plt.imshow(scaled_img)
        plt.imshow(pred_prob)
        
        save_dir = os.path.join(cfg['Debug']['debug_dir'], debug_type)
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(os.path.join(save_dir, str(basename) + '_heatmap.png'), dpi=200, bbox_inches='tight')
        gc.collect()
        
        
    @classmethod
    def output_debug(cls, dataset, idxs, model, save_dir, debug_type):
        save_dir = os.path.join(save_dir, debug_type)
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        for i, idx in enumerate(idxs):
            img, __ = dataset[idx]
            img = DataUtils.to_device(img.unsqueeze(0))
            _img = img.cpu().detach().numpy()
            preds = model(img)
            
            preds = preds.cpu().detach().numpy()
            # cls.draw_heatmap(i, preds[0], img, debug_type=debug_type)
            boxes, scores = cls.post_process(_img, preds, True)
            boxes, scores = boxes[0], scores[0]
            idxs = np.where(np.array(scores)>cfg['Global']['prob_threshold'])[0]
            boxes = [boxes[i].tolist() for i in idxs]
            
            img = DataUtils.image_to_numpy(img)
            img = Visualization.draw_polygon(img, boxes)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f"{save_dir}/{i}_pred.png", img)