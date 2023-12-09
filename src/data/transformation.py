from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import albumentations as A
import albumentations.augmentations.geometric as A_geo
from albumentations.pytorch.transforms import ToTensorV2

from . import cfg

import cv2
import math


class TransformDB(object):
    def __init__(self) -> None:
        self.image_size = cfg['Train']['dataset']['transforms']['image_shape']
        # image transformation
        self.__tranform = A.Compose([
            A.Resize(self.image_size[1], self.image_size[2]),
            A.Normalize(always_apply=True),
            ToTensorV2()],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        # image augmentation
        self.__augment = A.Compose([
            A.SafeRotate(limit=[-5, 5], p=0.3),
            A.Blur(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.MedianBlur(p=0.1, blur_limit=5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.3)
        ])
    
    def transform(self, image, label):
        """Reference: https://github.com/albumentations-team/albumentations/issues/750
        """
        label = np.array(label, np.float32).reshape(-1, 2).tolist()
        label = [val + [i] for i, val in enumerate(label)]
        transformed = self.__tranform(image=image, keypoints=label)
        transformed_image = transformed['image']
        transformed_label = [i[:-1] for i in sorted(transformed['keypoints'], key=lambda x: x[2])]
        transformed_label = [[max(min(x[0], self.image_size[1]-1), 0), max(min(x[1], self.image_size[1]-1), 0)] for x in transformed_label]
        transformed_label = np.array(transformed_label, np.float32).reshape(-1, 4, 2)
        return transformed_image, transformed_label
        
    def augment(self, image):
        augmented = self.__augment(image=image)
        augmented_image = augmented['image']
        return augmented_image