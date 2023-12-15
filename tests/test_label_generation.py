from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *
from tqdm import tqdm
from src.data.dataset import ICDAR2015Dataset
from src.data.label_generator import LabelGenerator

dataset = ICDAR2015Dataset('Train')
label_generate = LabelGenerator()


for i in tqdm(range(10), desc="Debug for Label Generation ..."):
    image, label = dataset[i]
    image, gt, mask = LabelGenerator()(image, label)
    image = DataUtils.image_to_numpy(image)
    Visualization.save_debug((gt+1)*255, cfg['Debug']['label_generation'], f"{i}_gt.png")
    Visualization.save_debug(mask*255, cfg['Debug']['label_generation'], f"{i}_mask.png")