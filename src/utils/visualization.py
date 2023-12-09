from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import numpy as np

from . import *

class Visualization:

    @classmethod
    def draw_polygon(cls, image, polygon):
        # polygon: x1, y1, x2, y2, x3, y3, x4, y4
        isClosed = True
        color = (255, 0, 0)
        thickness = 2
        image = DataUtils.image_to_numpy(image)
        image = np.array(image, np.uint8)
        polygon = np.array(polygon, np.int32)
        for pts in polygon:
            pts = pts.reshape((-1, 1, 2))
            image = cv2.polylines(image, [pts], isClosed, color, thickness)
        return image
    
    @classmethod
    def save_debug(cls, image, save_dir, basename):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, basename)
        cv2.imwrite(save_path, image)
