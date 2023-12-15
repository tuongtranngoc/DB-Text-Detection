from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *
from tqdm import tqdm
from src.data.dataset import ICDAR2015Dataset
from src.data.label_generator import LabelGenerator, shrink_polygon_py, shrink_polygon_pyclipper

from shapely.geometry import Polygon
import pyclipper
import numpy as np
np.seterr(divide="ignore", invalid="ignore")
import cv2

dataset = ICDAR2015Dataset('Eval')
label_generate = LabelGenerator()
shrink_ratio = 0.4

for i in tqdm(range(0,10), desc="Debug for shrink map..."):
    image, label = dataset[i]
    image = DataUtils.image_to_numpy(image)
    polygon = label[0]

    shrink2 = shrink_polygon_pyclipper(polygon, shrink_ratio)
    image = Visualization.draw_polygon(image, polygon, color=(0, 0, 255))
    image = Visualization.draw_polygon(image, shrink2, color=(255, 0, 0))

    poly = Polygon(polygon)
    distance = poly.area * (1 - np.power(shrink_ratio, 2)) / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    bbox = cv2.minAreaRect(expanded)
    points = cv2.boxPoints(bbox)
    image = Visualization.draw_polygon(image, points, color=(0, 255, 0))
    Visualization.save_debug(image, cfg['Debug']['label_generation'], f'shrink_map_{i}.png')

    # import ipdb; ipdb.set_trace()