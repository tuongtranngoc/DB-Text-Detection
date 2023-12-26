from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *
from tqdm import tqdm
from src.data.dataset import ICDAR2015Dataset


dataset = ICDAR2015Dataset('Eval')


for i in tqdm(range(30), desc="Debug for Label Generation ..."):
    image, shrink_map, shrink_mask, border_map, border_mask = dataset[i]
    image = DataUtils.image_to_numpy(image)
    Visualization.save_debug((shrink_map)*255, cfg['Debug']['label_generation'], f"{i}_shrink_map.png")
    Visualization.save_debug(shrink_mask*255, cfg['Debug']['label_generation'], f"{i}_shrink_mask.png")
    Visualization.save_debug((border_map)*255, cfg['Debug']['label_generation'], f"{i}_border_map.png")
    Visualization.save_debug(border_mask*255, cfg['Debug']['label_generation'], f"{i}_boder_mask.png")