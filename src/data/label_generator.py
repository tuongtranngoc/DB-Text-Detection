from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from . import *

import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon


def shrink_polygon_py(polygon, shrink_ratio):
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()
    polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
    polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
    return polygon


def shrink_polygon_pyclipper(polygon, shrink_ratio):
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked


class LabelGenerator(object):
    def __init__(self, shrink_ratio=0.4, min_text_size=8, shrink_type='pyclipper') -> None:
        shrink_func_dict = {
            'py': shrink_polygon_py,
            'pyclipper': shrink_polygon_pyclipper
        }
        self.shrink_func = shrink_func_dict[shrink_type]
        self.shrink_ratio = shrink_ratio
        self.min_text_size = min_text_size
        self.shrink_type = shrink_type

    def __call__(self, image, polygons):
        h,w = image.shape[:2]
        polygons = self.validate_polygons(polygons, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
            else:
                shrinked = self.shrink_func(polygon, self.shrink_ratio)
                if shrinked.size == 0:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
        return image, gt, mask

    def validate_polygons(self, polygons, h, w):
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w-1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h-1)

        for i in range(len(polygons)):
            area = cv2.contourArea(polygons[i])
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons
