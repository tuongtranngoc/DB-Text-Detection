

# Label generation

## Generation Mechanism

<p align="center">
    <img src="../../images/label_generation.png">
</p>

The label generation for the probability map is inspired by [PSENet](https://arxiv.org/abs/1903.12473). Given a text image, each polygon of its text regions is described by a set of segments:

$$G=\lbrace S_n \rbrace_{k=1}^n$$

$n$ is the number of vertexes (which may be different datasets). Then the positive area is generated by shrinking the polygon $G$ to $G_s$ using offset $D$ of shrinking is computed from perimeter $L$ and area A of the original polygon:

$$D=\frac{A(1-r^2)}{L}$$

where $r$ is the shrink ratio, set to 0.4 empirically.

To generate labels for threshold map. Firstly the text polygon $G$ is dilated with the same offset $D$ to $G_d$. The gap between $G_s$ and $G_d$ as the border of the text regions, where the label of threshold map can be generated by computing the distance to the closet segment in $G$ in the training phase.


## Experiment on the ICDAR2015 dataset

Based on `pyclipper` and `Polygon` libraries, we compute offset $D$ from the perimeter $L$ and area $A$ of the original polygon:

```python
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon

def shrink_polygon_pyclipper(polygon, shrink_ratio):
    polygon_shape = Polygon(polygon)
    # Compute offset D
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    # Compute plygon G_s or G_s
    # G_s: shrinked = padding.Execute(-distance)
    # G_d: shrinked = padding.Execute(distance)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked
```

And some examples from the `Icdar2015` dataset with $n$ = 4 (number of vertexes):

| <p align=center>threshold map</p>| <p align='center'>probability map</p> | 
|--|--|
| <img src="../../images/shrink_map_21.png" width=320> | <img src="../../images/21_gt.png" width=320> |
| <img src="../../images/shrink_map_27.png" width=320> | <img src="../../images/27_gt.png" width=320> |



