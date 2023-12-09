from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *
import numpy as np


class DataUtils:

    def polygon2xyxy(cls, polygon):
        xyxy = np.array(polygon, dtype=np.float32)
        xyxy = xyxy.reshape((4, 2))
        xmin = xyxy[:, 0].min()
        ymin = xyxy[:, 1].min()
        xmax = xyxy[:, 0].max()
        ymax = xyxy[:, 1].max()

        return [xmin, ymin, xmax, ymax]
