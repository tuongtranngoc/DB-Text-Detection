from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import cv2
import numpy as np
from tqdm import tqdm

from . import *
from src.data.dataset import ICDAR2015Dataset

dataset = ICDAR2015Dataset('Eval')


def validate_polygons(polygons, h, w):
    for polygon in polygons:
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w-1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h-1)

    for i in range(len(polygons)):
        area = cv2.contourArea(polygons[i])
        if area > 0:
            polygons[i] = polygons[i][::-1, :]
    return polygons


for i in tqdm(range(10)):
    image, label = dataset[i]
    image = DataUtils.image_to_numpy(image)
    h, w = image.shape[:2]
    # import ipdb; ipdb.set_trace()
    image = Visualization.draw_polygon(image, label, color=(255, 0, 0))
    polygons = validate_polygons(label, h, w)
    image = Visualization.draw_polygon(image, polygons, color=(0, 0, 255))
    Visualization.save_debug(image, cfg['Debug']['validate_polygons'], f"{i}.png")