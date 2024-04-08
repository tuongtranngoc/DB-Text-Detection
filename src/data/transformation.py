from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from . import cfg


class TransformDB(object):
    """Using augmentation of keypoints from albumentation
    Reference: https://albumentations.ai/docs/examples/example_keypoints/
    """
    def __init__(self) -> None:
        self.image_size = cfg['Train']['dataset']['transforms']['image_shape']
        # image transformation
        self.__tranform = A.Compose([
            A.Resize(self.image_size[1], self.image_size[2]),
            A.Normalize(always_apply=True),
            ToTensorV2()],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        # image augmentation
        self.__augment = A.Compose(transforms=[
            A.ToGray(p=0.1),
            A.HorizontalFlip(p=0.3),
            A.Affine(p=0.3, rotate=15),
            A.Blur(p=0.3, blur_limit=5),
            A.RandomBrightnessContrast(p=0.3),
            A.MedianBlur(p=0.1, blur_limit=5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.RandomCrop(height=320, width=320, p=0.3)
            ], p=0.55,
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def reconstruct_polygon(self, points, len_labels):
        new_points = []
        st = 0
        for len_l in len_labels:
            new_points.append(np.array(points[st: st+len_l], dtype=np.int32))
            st += len_l
        return new_points
    
    def transform(self, image, labels):
        """Reference: https://github.com/albumentations-team/albumentations/issues/750
        """
        labels = [list(label) for label in labels]
        len_labels = [len(label) for label in labels]
        labels = sum(labels, [])
        labels = [list(val) + [i] for i, val in enumerate(labels)]
        transformed = self.__tranform(image=image, keypoints=labels)
        transformed_image = transformed['image']
        transformed_label = [list(i[:-1]) for i in sorted(transformed['keypoints'], key=lambda x: x[2])]
        transformed_label = self.reconstruct_polygon(transformed_label, len_labels)
        
        return transformed_image, transformed_label
    
    def augment(self, image, labels):
        """Reference: https://github.com/albumentations-team/albumentations/issues/750
        """
        labels = [list(label) for label in labels]
        len_labels = [len(label) for label in labels]
        labels = sum(labels, [])
        labels = [list(val) + [i] for i, val in enumerate(labels)]
        augmented = self.__augment(image=image, keypoints=labels)
        augmented_image = augmented['image']
        augmented_label = [list(i[:-1]) for i in sorted(augmented['keypoints'], key=lambda x: x[2])]
        augmented_label = self.reconstruct_polygon(augmented_label, len_labels)

        return augmented_image, augmented_label