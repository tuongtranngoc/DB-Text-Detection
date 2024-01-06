from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg

import os
import argparse
from tqdm import tqdm

from src.utils.logger import Logger
from src.utils.data_utils import DataUtils
from src.data.dataset import ICDAR2015Dataset
from src.models.diff_binarization import DiffBinarization
from src.models.losses.db_loss import DiffBinarizationLoss



class Evaluator:
    def __init__(self) -> None:
        pass

    def eval(self):
        pass