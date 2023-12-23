from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import *

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class ICDAR2015Dataset(Dataset):
    def __init__(self, mode) -> None:
        super(ICDAR2015Dataset, self).__init__()
        self.mode = mode
        self.img_dir = cfg[self.mode]['dataset']['image_dir']
        self.anno_dir = cfg[self.mode]['dataset']['anno_dir']
        self.ignore_tags = cfg[self.mode]['dataset']['ignore_tags']
        self.is_aug = cfg[self.mode]['dataset']['transforms']['augmentation']
        self.image_size = cfg[self.mode]['dataset']['transforms']['image_shape']
        self.transform = TransformDB()
        self.dataset = self.load_dataset()

    def load_dataset(self):
        dataset = []
        for img_path in tqdm(glob.glob(os.path.join(self.img_dir, "*.jpg")), desc=f"Loading dataset for {self.mode}"):
            basename = os.path.basename(img_path).split('.jpg')[0]
            anno_path = os.path.join(self.anno_dir, 'gt_' + basename + '.txt')
            if not os.path.exists(anno_path): continue
            polygon_list = []
            with open(anno_path, 'r') as f_anno:
                annos = f_anno.readlines()
                for anno in annos:
                    anno = anno.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')[:8]
                    polygon = [int(a) for a in anno]
                    polygon_list.append(polygon)
            dataset.append([img_path, np.array(polygon_list, np.float32).reshape((-1, 4, 2)).tolist()])
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
        return image, label
    

if __name__ == "__main__":
    icdar_dataset = ICDAR2015Dataset()

    
    
            
