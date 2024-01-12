from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from . import *

import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon

np.seterr(divide='ignore',invalid='ignore')


class ShrinkGenerator(object):
    def __init__(self, shrink_ratio=0.4, min_text_size=8, shrink_type='pyclipper') -> None:
        self.shrink_ratio = shrink_ratio
        self.min_text_size = min_text_size
        self.shrink_type = shrink_type

    def __call__(self, image, polygons):
        h,w = image.shape[:2]
        polygons = self.validate_polygons(polygons, h, w)
        shrink = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
            else:
                polygon_shape = Polygon(polygon)
                subject = [tuple(l) for l in polygon]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = []
                # Increase the shrink ratio every time we get multiple polygon returned back
                possible_ratios = np.arange(self.shrink_ratio, 1, self.shrink_ratio)
                np.append(possible_ratios, 1)
                for ratio in possible_ratios:
                    distance = polygon_shape.area * (1 - np.power(ratio, 2)) / polygon_shape.length
                    shrinked = padding.Execute(-distance)
                    if len(shrinked) == 1:
                        break
                
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    continue

                for each_shirnk in shrinked:
                    shirnk = np.array(each_shirnk).reshape(-1, 2)
                    cv2.fillPoly(shrink, [shirnk.astype(np.int32)], 1)
        return image, shrink, mask
    
    def validate_polygons(self, polygons, h, w):
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w-1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h-1)

        for i in range(len(polygons)):
            area = cv2.contourArea(polygons[i])
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons
    

class BorderGenerator(object):
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, image, polygons):
        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        for i in range(len(polygons)):
            self.draw_border_map(polygons[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        return image, canvas, mask

    def draw_border_map(self, polygons, canvas, mask):
        polygons = np.array(polygons)
        assert polygons.ndim == 2
        assert polygons.shape[1] == 2

        polygons_shape = Polygon(polygons)
        if polygons_shape.area <= 0: return
        distance = polygons_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygons_shape.length
        subject = [tuple(l) for l in polygons]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = padding.Execute(distance)
        if len(padded_polygon) == 0: return
        padded_polygon = np.array(padded_polygon[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygons[:, 0] = polygons[:, 0] - xmin
        polygons[:, 1] = polygons[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygons.shape[0], height, width), dtype=np.float32)
        for i in range(polygons.shape[0]):
            j = (i + 1) % polygons.shape[0]
            absolute_distance = self.distance(xs, ys, polygons[i], polygons[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, point_1, point_2):
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)

        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)
        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2