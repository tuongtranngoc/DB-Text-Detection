from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon


class DBPostProcess():
    def __init__(self, thresh=0.3, box_thresh=0.3, max_candidates=100, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, inputs, preds, is_output_polygon=False):
        if isinstance(preds, torch.Tensor) or isinstance(preds, np.ndarray):
            pred = preds[:, 0, :, :]
        else:
            pred = preds[0]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        __, __, H, W = inputs.shape
        for batch_idx in range(pred.shape[0]):
            if is_output_polygon:
                boxes, scores = self.bitmap2polygon(pred[batch_idx], segmentation[batch_idx], W, H)
            else:
                boxes, scores = self.bitmap2boxes(pred[batch_idx], segmentation[batch_idx], W, H)
            
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh
    
    def bitmap2polygon(self, pred, _bitmap, imgw, imgh):
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.copy()
        h, w = bitmap.shape
        boxes = []
        scores = []

        contours, __ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            # Reference: https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/
            eps = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps, True)
            points = approx.reshape((-1, 2))

            if points.shape[0] < 4: continue

            score = self.box_score_fast(pred, contour.squeeze(1))

            if score < self.box_thresh: continue

            if points.shape[0] > 2:
                try:
                    box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                except:
                    continue
                if len(box) > 1: continue
            else:
                continue
        
            box = box.reshape(-1, 2)
            if box.shape[0] == 0: continue
            box, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2: continue
            box = np.array(box)
            if not isinstance(imgw, int):
                imgw = imgw.item()
                imgh = imgh.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / w * imgw), 0, imgw)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * imgh), 0, imgh)
            boxes.append(box.astype(np.int32))
            scores.append(score)
        
        return boxes, scores

    def bitmap2boxes(self, pred, _bitmap, imgw, imgh):
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.copy()  # The first channel
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(imgw, int):
                imgw = imgw.item()
                imgh = imgh.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * imgw), 0, imgw)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * imgh), 0, imgh)
            boxes[index, :] = self.poly2xyxy(box.astype(np.int16))
            scores[index] = score

        return boxes, scores

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w-1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded
    
    def poly2xyxy(self, poly:np.ndarray):
        x1 = poly[:, 0].min()
        y1 = poly[:, 1].min()
        x2 = poly[:, 0].max()
        y2 = poly[:, 1].max()
        return np.array([x1, y1, x2, y2], dtype=np.int16)
    
    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])