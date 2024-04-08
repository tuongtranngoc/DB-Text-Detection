from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *

import os
import cv2
import math
import glob
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class TotalTextDataset(Dataset):
    def __init__(self, mode) -> None:
        super(TotalTextDataset, self).__init__()
        self.mode = mode
        self.img_dir = cfg[self.mode]['dataset']['image_dir']
        self.anno_dir = cfg[self.mode]['dataset']['anno_dir']
        self.ignore_tags = cfg[self.mode]['dataset']['ignore_tags']
        self.is_aug = cfg[self.mode]['dataset']['transforms']['augmentation']
        self.image_size = cfg[self.mode]['dataset']['transforms']['image_shape']
        self.transform = TransformDB()
        self.dataset = self.load_dataset()

    def process_dataset(self, _image, _labels):
        image = _image.clone()
        image = DataUtils.image_to_numpy(image)
        labels = _labels.copy()
        ___, shrink_map, shrink_mask = ShrinkGenerator()(image, labels)
        ___, border_map, border_mask = BorderGenerator()(image, labels)
        return shrink_map, shrink_mask, border_map, border_mask

    def load_dataset(self):
        dataset = []
        for img_path in tqdm(glob.glob(os.path.join(self.img_dir, "*.jpg")), desc=f"Loading dataset for {self.mode}"):
            basename = os.path.basename(img_path)
            anno_path = os.path.join(self.anno_dir, basename + '.txt')
            if not os.path.exists(anno_path): continue
            polygon_list = []
            with open(anno_path, 'r') as f_anno:
                annos = f_anno.readlines()
                for anno in annos:
                    anno = anno.strip().split(',')
                    text = str(anno[-1])
                    if text in self.ignore_tags: continue
                    polygon = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in anno]
                    num_points = math.floor((len(polygon) - 1) / 2) * 2
                    polygon = np.array(list(map(float, polygon[:num_points]))).reshape((-1, 2))
                    box = DataUtils.order_points_clockwise(polygon)
                    if cv2.contourArea(box) <= 0: continue
                    if len(polygon) < 3: continue
                    polygon_list.append(polygon.tolist())
                if len(polygon_list) < 1: continue
            dataset.append([img_path, polygon_list])
        return dataset
    
    def get_image_label(self, img_pth, label, is_aug):
        image = cv2.imread(img_pth)[..., ::-1]
        if is_aug:
            image, label = self.transform.augment(image, label)
        image, label = self.transform.transform(image, label)
        return image, label

    def __len__(self): return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        image, label = self.get_image_label(img_path, label, is_aug=self.is_aug)
        shrink_map, shrink_mask, border_map, border_mask = self.process_dataset(image, label)
        return image, (shrink_map, shrink_mask, border_map, border_mask)

    
    
            
