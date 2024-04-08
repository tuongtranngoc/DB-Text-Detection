from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import torch
import argparse
import torch.nn.functional as F

from . import *

logger = Logger.get_logger("EXPORT")


class Exporter:
    def __init__(self, args):
        self.args = args
        self.model = DiffBinarization(pretrained=False)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)['model'])
        self.sample = torch.ones(size=cfg['Train']['dataset']['transforms']['image_shape']).unsqueeze(0).to(self.args.device)
        self.model.to(self.args.device)
        self.model.eval()
    
    def export_torchscript(self):
        """https://pytorch.org/docs/stable/jit.html
        """
        f = str(self.args.model_path).replace('.pth', f'.torchscript')
        logger.info(f'Starting export with torch {torch.__version__}...')
        ts = torch.jit.trace(self.model, self.sample, strict=False)
        logger.info(f'Optimizing for mobile...')
        ts.save(f)
        return f
    
    def __call__(self):
        logger.info("Begining export model ...")
        self.export_torchscript()
        
    
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'], help="Path to pytorch model")
    parser.add_argument("--device", type=str, default='cuda', help="Select device for export format")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    exporter = Exporter(args)
    exporter()