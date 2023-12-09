from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *
from tqdm import tqdm
from src.data.dataset import ICDAR2015Dataset

dataset = ICDAR2015Dataset('Train')


for i in tqdm(range(10)):
    image, label = dataset[i]
    image = Visualization.draw_polygon(image, label)
    Visualization.save_debug(image, cfg['Debug']['dataset'], f'{i}.png')
